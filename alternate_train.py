from transformers import Trainer, TrainingArguments, AutoModelForImageClassification
from utils.training_utils import CustomImageDataset
import os


model = AutoModelForImageClassification.from_pretrained('/ext/sanisoglum/checkpoints')

print('model loaded')

os.makedirs('/ext/sanisoglum/outputs', exist_ok=True)

training_args = TrainingArguments(
    output_dir='/ext/sanisoglum/outputs',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=CustomImageDataset,
)

print('Trainer defined')

trainer.train()

print('Training completed')