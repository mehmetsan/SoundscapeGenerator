import os
import torch.nn as nn
from diffusers import StableDiffusionPipeline

model_cache_dir = '/ext/sanisoglum/checkpoints/'

pipeline = StableDiffusionPipeline.from_pretrained(model_cache_dir,
                                                   cache_dir=model_cache_dir,
                                                   resume_download=True)

model = pipeline.unet
model = nn.DataParallel(model)
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer

print(model)
print('Model is loaded')
