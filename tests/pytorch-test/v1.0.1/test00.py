import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")

print("Creating random matrix:")
x = torch.rand(5,3)
print(x)

