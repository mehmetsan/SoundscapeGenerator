import os

from diffusers import StableDiffusionPipeline

model_cache_dir = '/ext/sanisoglum/checkpoints/caches'

if not os.path.exists(os.path.join(model_cache_dir, 'models--riffusion--riffusion-model-v1')):
    print('A cache already exists, either resuming download or skipping download')

os.makedirs(model_cache_dir, exist_ok=True)
pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1",
                                                   cache_dir=model_cache_dir,
                                                   resume_download=True)

print('Model is loaded')
