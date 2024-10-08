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

if not os.path.exists('categorized_spectrograms'):
    print('Categorized spectrograms are missing, run the categorize_spectrograms.py script first')
else:

    try:
        # Logging in to wandb
        try:
            wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
            print('Wandb login successful')
            run = wandb.init(
                project="SoundscapeGenerator",
            )
        except Exception as e:
            raise Exception(f"Wandb login failed due to {e}")

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

                    # Compute loss (simplified)
                    loss = torch.nn.functional.mse_loss(model_output, noise)
                    wandb.log({"loss": loss.item()})

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

            print('Finished training')
        except Exception as e:
            raise Exception(f"Training failed due to {e}")

    except Exception as e:
        print(e)

