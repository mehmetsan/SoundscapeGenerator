import torch
import wandb
import torch.nn as nn
from env_variables import model_cache_path
from utils.riffusion_pipeline import RiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Emotion embedding layer
class EmotionEmbedding(nn.Module):
    def __init__(self, num_classes=12, embedding_dim=768):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, labels):
        return self.embedding(labels)


# Add an extra channel
def add_extra_channel(images):
    extra_channel = torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device)
    return torch.cat((images, extra_channel), dim=1)


# Define transformations
transform = transforms.Compose([
    transforms.Pad((0, 0, 11, 0)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Initialize WandB
try:
    wandb.login(key="your_wandb_key_here")
    run = wandb.init(project="SoundscapeGenerator")
except Exception as e:
    raise Exception(f"Wandb login failed due to {e}")

# Load dataset
dataset = datasets.ImageFolder(root='categorized_spectrograms', transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the RiffusionPipeline
pipeline = RiffusionPipeline.from_pretrained(pretrained_model_name_or_path="riffusion/riffusion-model-v1",
                                             cache_dir=model_cache_path,
                                             resume_download=True)
unet = pipeline.unet
unet.to(device)  # Move the model to GPU

# Define optimizer and loss function
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# Emotion embedding layer
emotion_embedding_layer = EmotionEmbedding(num_classes=12).to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    unet.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in dataloader:
        images, labels = batch

        # Move the batch to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Add the extra channel
        images = add_extra_channel(images)

        # Generate the timesteps and encoder hidden states
        timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()
        encoder_hidden_states = torch.randn(images.size(0), 1, 768, device=device)

        # Get emotion embeddings and concatenate with encoder hidden states
        emotion_embedding = emotion_embedding_layer(labels)
        encoder_hidden_states_with_emotion = torch.cat((encoder_hidden_states, emotion_embedding.unsqueeze(1)), dim=2)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        try:
            outputs = unet(images, timesteps, encoder_hidden_states_with_emotion).sample
        except RuntimeError as e:
            print(f"Out of memory during forward pass: {e}")
            torch.cuda.empty_cache()
            continue  # Skip this batch if out of memory

        # Compute the loss (denoising loss)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log loss with WandB
        wandb.log({"loss": loss.item()})

        running_loss += loss.item()

        # Clear cached memory to avoid OOM
        torch.cuda.empty_cache()

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

print('Finished training')

# Save the trained model
unet.save_pretrained('path/to/save/model')
