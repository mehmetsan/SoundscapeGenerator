import wandb
from env_variables import model_save_path


try:
    wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
    run = wandb.init(project="SoundscapeGenerator")
except Exception as e:
    raise Exception(f"Wandb login failed due to {e}")

try:
    artifact = wandb.Artifact("riffusion_fine_tuned", type="model")
    artifact.add_file(model_save_path)
    run.log_artifact(artifact)
except:
    print("")