from diffusers import DiffusionPipeline
import torch
import os 

os.environ['HF_HUB_CACHE'] = "/src/model-cache"
os.environ['HF_HOME'] = "/src/model-cache"

from huggingface_hub import scan_cache_dir, snapshot_download

# data = scan_cache_dir("/models/huggingface/hub")

# for repo in data.repos:
#     print(repo.repo_id)
#     print(str(repo.repo_path))

# print("Downloading weights: cagliostrolab/animagine-xl-3.0")
# snapshot_download(repo_id="cagliostrolab/animagine-xl-3.0", cache_dir="/models/huggingface/hub")

base_model_path = "cagliostrolab/animagine-xl-3.0"
lora_model_path = "galverse/mama-1.5"

print("Loading model...")
pipe = DiffusionPipeline.from_pretrained(base_model_path, 
                                         torch_dtype=torch.float16,
                                         use_safetensors=True)
pipe.to("cuda")
pipe.load_lora_weights(lora_model_path)

generator = torch.Generator("cuda").manual_seed(1024)
config = {'prompt': ['a girl'], 'negative_prompt': ['bad'], 'guidance_scale': 7.5, 'generator': generator, 'num_inference_steps': 30}
sdxl_params = {'width': 1024, 'height': 1024, 'cross_attention_kwargs': {'scale': 1.0}}

image = pipe(**config, **sdxl_params).images[0]
image.save("naruto.png")