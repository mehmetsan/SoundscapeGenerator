import torch
print("Is cuda available?")
print("Yes" if torch.cuda.is_available() else "No")

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))