import torch
import wandb
import torch.nn as nn
from env_variables import model_cache_path, model_save_path
from utils.riffusion_pipeline import RiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Emotion embedding layer
class EmotionEmbedding(nn.Module):
    def __init__(self, num_classes=12, embedding_dim=768):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, labels):
        return self.embedding(labels)



# Custom dataset to handle corrupted images
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (OSError, IOError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            return None


# Add an extra channel
def add_extra_channel(images):
    extra_channel = torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device)
    return torch.cat((images, extra_channel), dim=1)


# Define transformations
transform = transforms.Compose([
    transforms.Pad((0, 0, 11, 0)),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# Initialize WandB
try:
    wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
    run = wandb.init(project="SoundscapeGenerator")
except Exception as e:
    raise Exception(f"Wandb login failed due to {e}")

BATCH_SIZE = 1

# Load dataset with SafeImageFolder to handle corrupted images
dataset = SafeImageFolder(root='categorized_spectrograms', transform=transform)

# Create DataLoader that skips corrupted images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Filter out None entries
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

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
num_epochs = 1
total_batches = len(dataloader)

scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 2

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    unet.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue  # Skip the batch if it was filtered out due to corrupted images
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
        encoder_hidden_states_with_emotion = encoder_hidden_states + emotion_embedding.unsqueeze(1)

        # Zero the gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Forward pass
            outputs = unet(images, timesteps, encoder_hidden_states_with_emotion).sample
            # Compute the loss (denoising loss)
            loss = criterion(outputs, images)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Backward pass and optimization
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients after updating

        # Log loss with WandB
        wandb.log({"loss": loss.item()})

        running_loss += loss.item()

        # Print the current batch number and total batches
        print(f"Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item()}")

        # Clear cached memory to avoid OOM
        torch.cuda.empty_cache()

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

print('Finished training')

# Save the trained model
unet.save_pretrained(model_save_path)
