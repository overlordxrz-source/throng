# Scope: Milestone 1 — Environment Setup & Differentiable Wrapper

## Architecture
Verify the remote GPU and package environment on `dragonbg` server. Build a fully differentiable PyTorch wrapper around the TRIBE v2 `FmriEncoder` (`tribe_grad.py`).

## Interface Contract
`tribe_grad.py` must expose:
- `DifferentiableTribeEncoder(device='cuda')`
- `forward(self, x: torch.Tensor) -> torch.Tensor`
  - Input `x`: float torch.Tensor of shape `(B, 3, H, W)` normalized to `[0, 1]` with `requires_grad=True`.
  - Output: torch.Tensor of shape `(B, 20484)` representing fsaverage5 vertex activations, with working gradients back to `x`.

## Verification Criteria
- A unit test script `test_tribe_grad.py` must be written.
- It must run on `dragonbg`.
- It must perform a backward pass from the output of the wrapper to a dummy input tensor, verifying `x.grad is not None` and gradient elements are non-zero.

## Tasks
1. Connect via SSH to `dragonbg`.
2. Check CUDA/GPU (nvidia-smi) and python packages.
3. Locate the installed TRIBE v2 `FmriEncoder` codebase or install dependencies.
4. Implement `tribe_grad.py` bypassing non-differentiable preprocessing (like PIL image conversion) inside TRIBE's encoder.
5. Verify gradients flow using the test script.
