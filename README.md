# Cog-SDXL (https://github.com/replicate/cog-sdxl)
This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

- for prediction in local,

```bash
cog predict -i prompt="a photo of TOK"
```

```bash
cog run -p 5000 python -m cog.server.http
```

- Build and push to replicate 
```
cog build -t cog-sdxl:v1.0.0
```

- Push to replicate (You need to create a model in replicate https://replicate.com/create)
```
cog push {MODEL_REPLICATE_URL}
```
