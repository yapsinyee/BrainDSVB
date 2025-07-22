import torch

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if MPS (Metal Performance Shaders) is available
print(f"Is MPS available: {torch.backends.mps.is_available()}")

# Check if MPS is built with PyTorch
print(f"Is MPS built: {torch.backends.mps.is_built()}")

# Example of moving a tensor to MPS device
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(5, device=mps_device)
    print(f"Tensor on MPS device: {x}")
else:
    print("MPS is not available. PyTorch will use CPU.")

