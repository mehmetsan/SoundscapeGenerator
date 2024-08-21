import torch
import wandb
import torch.nn as nn

from env_variables import model_cache_path
from utils.riffusion_pipeline import RiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def add_extra_channel(images):
    extra_channel = torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device)
    return torch.cat((images, extra_channel), dim=1)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

try:
    wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
    print('Wandb login successful')
    run = wandb.init(
        project="SoundscapeGenerator",
    )
except Exception as e:
    raise Exception(f"Wandb login failed due to {e}")


# Load dataset
dataset = datasets.ImageFolder(root='categorized_spectrograms', transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print('Dataset ready')

device = "cuda"

# Load the RiffusionPipeline
pipeline = RiffusionPipeline.from_pretrained(pretrained_model_name_or_path="riffusion/riffusion-model-v1",
                                             cache_dir=model_cache_path,
                                             resume_download=True)
print('Model is loaded')

pipeline.to(device)  # Move the pipeline to the GPU if available

# Assuming the pipeline has a model attribute that is trainable
unet = pipeline.unet
unet = nn.DataParallel(unet)
unet.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

print('Started training')
# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    unet.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU if available

        images = add_extra_channel(images)

        timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()  # Example timesteps
        encoder_hidden_states = torch.randn(images.size(0), 512, device=device)  # Example encoder hidden states

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = unet(images, timesteps, encoder_hidden_states)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss

        wandb.log({"loss": loss.item()})

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the parameters

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
print('Finished training')
# Save the trained model
unet.save_pretrained('path/to/save/model')