import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from PIL import ImageFile
from env_variables import model_cache_path, model_save_path
from utils.riffusion_pipeline import RiffusionPipeline
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

# Function to initialize the distributed process
def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

# Collate function for skipping None batches (from corrupted images)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Filter out None entries
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_model(rank, world_size):
    try:
        setup(rank, world_size)

        # Set the device for this process
        device = torch.device(f'cuda:{rank}')

        # Initialize WandB only on the main process (rank 0)
        if rank == 0:
            try:
                wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
                run = wandb.init(project="SoundscapeGenerator")
            except Exception as e:
                raise Exception(f"Wandb login failed due to {e}")

        BATCH_SIZE = 1
        EPOCHS = 1

        # Load dataset with SafeImageFolder to handle corrupted images
        dataset = SafeImageFolder(root='categorized_spectrograms', transform=transform)

        # Use DistributedSampler to split data across GPUs
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)


        # Create DataLoader that skips corrupted images
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)

        # Load the RiffusionPipeline
        pipeline = RiffusionPipeline.from_pretrained(pretrained_model_name_or_path="riffusion/riffusion-model-v1",
                                                     cache_dir=model_cache_path,
                                                     resume_download=True)
        unet = pipeline.unet

        # Move the model to the correct device and wrap it with DDP
        unet.to(device)
        unet = DDP(unet, device_ids=[rank])

        dist.barrier()

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
        criterion = nn.MSELoss()

        # Emotion embedding layer
        emotion_embedding_layer = EmotionEmbedding(num_classes=12).to(device)

        # Training loop
        scaler = torch.cuda.amp.GradScaler()
        accumulation_steps = 2

        for epoch in range(EPOCHS):
            try:
                unet.train()
                running_loss = 0.0

                for batch_idx, batch in enumerate(dataloader):
                    if batch is None:  # Skip empty batches caused by filtering out corrupted images
                        continue
                    print(f"Rank {rank} is processing batch {batch_idx} with data indices: {sampler.indices[:10]}")

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

                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    running_loss += loss.item()
                    torch.cuda.empty_cache()

                if rank == 0:
                    wandb.log({"loss": running_loss / len(dataloader)})

            except Exception as e:
                print(f"Error during training loop on rank {rank}: {e}")
                raise e

        # Save the trained model (only rank 0)
        if rank == 0:
            unet.module.save_pretrained(model_save_path)

        cleanup()

    except Exception as e:
        print(f"Error during initialization on rank {rank}: {e}")
        raise e


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
