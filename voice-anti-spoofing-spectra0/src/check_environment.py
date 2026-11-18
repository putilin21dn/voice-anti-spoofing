import sys, platform
import torch

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    a = torch.randn((1024, 1024), device="cuda")
    _ = a @ a
    torch.cuda.synchronize()
    print("CUDA matrix test: OK")
else:
    print("WARNING: CUDA is unavailable; CPU inference remains possible.")
