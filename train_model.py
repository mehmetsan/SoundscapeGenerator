import os

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.training_utils import CustomImageDataset

if not os.path.exists('categorized_spectrograms'):
    print('Categorized spectrograms are missing, run the categorize_spectrograms.py script first')
else:
    dataset = CustomImageDataset(root_dir='categorized_spectrograms')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load the checkpoint
    os.makedirs('/ext/sanisoglum/checkpoints/caches', exist_ok=True)
    pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", cache_dir='/ext/sanisoglum/checkpoints/caches')

    # Extract model components
    model = pipeline.unet
    model = nn.DataParallel(model)
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

    # Move model to device
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        text_encoder.to(device)

        # Training settings
        num_epochs = 1
        learning_rate = 1e-4
        scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=500)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

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
                time_steps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
                noisy_images = scheduler.add_noise(images, noise, time_steps)

                model_output = model(noisy_images, time_steps, text_embeddings)["sample"]

                # Compute loss (simplified)
                loss = torch.nn.functional.mse_loss(model_output, noise)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

        # Save the fine-tuned model
        model.save_pretrained('fine-tuned-riffusion')

        print('Finished training')
    else:
        print("Cuda wasn't available, terminating the training")
