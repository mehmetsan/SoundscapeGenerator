from diffusers import DiffusionPipeline
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

