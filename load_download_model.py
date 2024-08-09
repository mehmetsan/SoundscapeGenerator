import os
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.training_utils import CustomImageDataset

model_cache_dir = '/ext/sanisoglum/checkpoints/caches'

if os.path.exists(os.path.join(model_cache_dir, 'models--riffusion--riffusion-model-v1')):
    print('A cache already exists, either resuming download or skipping download')
else:
    print('Downloading model from scratch')
    os.makedirs(model_cache_dir, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1",
                                                   cache_dir=model_cache_dir,
                                                   resume_download=True)

model = pipeline.unet
model = nn.DataParallel(model)
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer

print('Model is loaded')

# Try training
try:
    if not torch.cuda.is_available():
        raise Exception("Cuda is not available, aborted training")

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

    print('Started training')
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for images, labels in dataloader:
            images = images.to(device)

            # Generate text embeddings
            text_inputs = [f"sndscp {dataset.classes[label]}" for label in labels]
            text_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).input_ids
            text_embeddings = text_encoder(text_inputs.to(device)).last_hidden_state

            # Forward pass
            noise = torch.randn_like(images)
            time_steps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],),
                                       device=images.device).long()
            noisy_images = scheduler.add_noise(images, noise, time_steps)

            model_output = model(noisy_images, time_steps, text_embeddings)["sample"]

            print(f"Model output is {model_output}")
            # Compute loss (simplified)
            loss = torch.nn.functional.mse_loss(model_output, noise)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

    print('Finished training')
except Exception as e:
    raise Exception(f"Training failed due to {e}")