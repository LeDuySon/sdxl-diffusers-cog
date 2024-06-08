# Cog-SDXL (https://github.com/replicate/cog-sdxl)
This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

for prediction,

```bash
cog predict -i prompt="a photo of TOK"
```

```bash
cog train -i input_images=@example_datasets/__data.zip -i use_face_detection_instead=True
```

```bash
cog run -p 5000 python -m cog.server.http
```
