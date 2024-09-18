from utils.riffusion_pipeline import RiffusionPipeline
from env_variables import  model_save_path

# Load the RiffusionPipeline
pipeline = RiffusionPipeline.from_pretrained(pretrained_model_name_or_path="riffusion/riffusion-model-v1",
                                             resume_download=True)

unet = pipeline.unet

unet.save_pretrained(model_save_path)
