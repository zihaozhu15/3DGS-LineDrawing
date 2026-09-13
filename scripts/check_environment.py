"""Check the dedicated environment with actual CUDA forward/backward work."""
import json
import sys

import torch
import torchvision

assert torch.cuda.is_available(), "CUDA is unavailable"
torch.manual_seed(0)
x = torch.randn(128, 128, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss).item()
assert x.grad is not None and torch.isfinite(x.grad).all().item()
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "cuda_forward_backward": "passed",
    "torchvision_cuda_nms": "passed",
}, indent=2))
