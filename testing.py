import torch
print("Is cuda available?")
print("Yes" if torch.cuda.is_available() else "No")

torch.cuda.device_count()

torch.cuda.current_device()

torch.cuda.device(0)

torch.cuda.get_device_name(0)