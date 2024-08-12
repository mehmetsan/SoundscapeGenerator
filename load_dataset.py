import os

import torch
import torch.nn as nn
import wandb
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader

from env_variables import model_cache_path
from utils.training_utils import CustomImageDataset

# Model loading
try:
    pipeline = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1",
                                                 cache_dir=model_cache_path,
                                                 resume_download=True)

    print('Model is loaded')
    # Extract model components
    model = pipeline.unet
    model = nn.DataParallel(model)
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

except Exception as e:
    raise Exception(f"Model loading failed due to {e}")

# Dataset loading
try:
    dataset = CustomImageDataset(root_dir='categorized_spectrograms')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print('Dataset loading successful')
except Exception as e:
    raise Exception(f"Dataset loading failed due to {e}")

print("wow lets go")

# Move model to device
device = "cuda"
model.to(device)
text_encoder.to(device)

print(f"\nThere are {torch.cuda.device_count()} available devices")

# Training settings
num_epochs = 1
learning_rate = 1e-4
scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=500)
optimizer = AdamW(model.parameters(), lr=learning_rate)