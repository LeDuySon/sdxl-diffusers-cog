import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
import huggingface_hub
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from huggingface_hub import snapshot_download
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils import load_image

from weights import WeightsDownloadCache

os.environ['HF_HUB_CACHE'] = "/src/model-cache"
os.environ['HF_HOME'] = "/src/model-cache"

print("HAHAHA")

SDXL_MODEL_CACHE = "/src/model-cache"
LORA_MODEL_CACHE = "/src/model-cache"
MODEL_ID = "cagliostrolab/animagine-xl-3.0"
DEFAULT_LORA_MODEL = "galverse/mama-1.5"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def load_lora_adapter(self, lora_id: str, pipe):
        prev_tuned_weights = self.tuned_weights
        
        # weights can be a URLPath, which behaves in unexpected ways
        lora_id = str(lora_id)
        if self.tuned_weights == lora_id:
            print("skipping loading .. weights already loaded")
            return

        # predictions can be cancelled while in this function, which
        # interrupts this finishing.  To protect against odd states we
        # set tuned_weights to a value that lets the next prediction
        # know if it should try to load weights or if loading completed
        self.tuned_weights = 'loading'

        try:
            local_weights_cache = self.weights_cache.ensure(lora_id)
        except Exception as e:
            traceback.print_exc()
            # back to use base model
            print(f"Lora id {lora_id} is not valid: {e}. Fallback to {self.tuned_weights} lora")
            self.tuned_weights = prev_tuned_weights
            return 
        
        # first we will get list of current activate adapter and remove them from memory
        activate_adapters = pipe.get_active_adapters()
        if(len(activate_adapters) > 0):
            print(f"Remove adapters {activate_adapters} from memory")
            pipe.delete_adapters(activate_adapters)

        print(f"loading lora {lora_id} from {local_weights_cache}")
        pipe.load_lora_weights(lora_id)

        self.tuned_weights = lora_id
        self.tuned_model = True

    def unload_lora_adapter(self, pipe: DiffusionPipeline):
        print(f"Unload lora {self.tuned_weights} from model")
        pipe.unload_lora_weights()

        self.tuned_weights = None
        self.tuned_model = False

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None

        self.weights_cache = WeightsDownloadCache(base_dir=LORA_MODEL_CACHE)

        if not os.path.exists(SDXL_MODEL_CACHE):
            os.makedirs(SDXL_MODEL_CACHE)
        
        is_model_exists = self.weights_cache.check_model_exist_in_cache(MODEL_ID)
        if(not is_model_exists):
            snapshot_download(repo_id=MODEL_ID, cache_dir=SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # load default lora model
        self.load_lora_adapter(DEFAULT_LORA_MODEL, self.txt2img_pipe)

        self.txt2img_pipe.to("cuda")

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")
        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPMSolverMultistep",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        lora_id: str = Input(
            description="Path to the lora model on huggingface. Leave blank to use the default weights.",
            default="",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if lora_id and lora_id != "":
            self.is_lora = True
            self.load_lora_adapter(lora_id, self.txt2img_pipe)
        elif self.tuned_model:
            self.is_lora = False
            self.unload_lora_adapter(self.txt2img_pipe)
        else:
            print("No lora to unload")

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        print(f"Prompt: {prompt}")
        if image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        print(f"Setting up scheduler {scheduler}")
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}
        
        print(f"Running prediction... with parameters: {common_args}\n {sdxl_kwargs}")
        try:
            output = pipe(**common_args, **sdxl_kwargs)
        except Exception as e:
            print(f"Error when running prediction: {e}")
            traceback.print_exc()
            return []
        
        print("Apply watermark...")
        if not apply_watermark:
            pipe.watermark = watermark_cache

        print("Get image outputs...")
        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
            
        print(f"Gathered {len(output_paths)} images.")

        return output_paths
