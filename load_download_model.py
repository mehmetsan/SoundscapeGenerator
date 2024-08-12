import os
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from env_variables import model_cache_path

if os.path.exists(os.path.join(model_cache_path, 'models--riffusion--riffusion-model-v1')):
    print('A cache already exists, either resuming download or skipping download')
else:
    print('Downloading model from scratch')
    os.makedirs(model_cache_path, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1",
                                                   cache_dir=model_cache_path,
                                                   resume_download=True)

model = pipeline.unet
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer

print(model)
print('Model is loaded')
