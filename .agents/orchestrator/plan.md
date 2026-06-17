# Implementation Plan — InverseTribe

This plan outlines the specific delegation of each milestone to subagents and how verification will be conducted.

## Milestone Delegation Roadmap

### M1. Environment & Differentiable Wrapper
- **Agent Type**: `teamwork_preview_worker`
- **Role**: `TribeGrad Specialist`
- **Tasks**:
  - Check the remote `dragonbg` environment (available python packages, GPU, CUDA, TRIBE v2 repository/installation).
  - Draft and verify `tribe_grad.py` to wrap `FmriEncoder` cleanly.
  - Implement a PyTorch unit test verifying `x.grad is not None` and non-zero gradient flow back to pixel space.
- **Verification**: Run units tests on `dragonbg` showing successful gradient backpropagation.

### M2. Region Atlas & Baseline Cache
- **Agent Type**: `teamwork_preview_worker`
- **Role**: `BrainAtlas Specialist`
- **Tasks**:
  - Implement `roi_atlas.py` utilizing Nilearn to map standard brain regions to fsaverage5 vertices.
  - Ensure the 60% coverage requirement is met.
  - Implement `baseline_cache.py` to download 500 COCO validation images, feed them to `tribe_grad.py`, and cache the baseline.
- **Verification**: Script check verifying vertex coordinates, coverage percentage, and cached baseline properties.

### M3. SDXL Guidance Loop
- **Agent Type**: `teamwork_preview_worker`
- **Role**: `Diffusion Specialist`
- **Tasks**:
  - Implement `generate.py` incorporating Stable Diffusion XL and differentiable brain guidance.
  - Incorporate customizable `guidance_strength` and `steps`.
  - Save results and verify brain loss trends down.
- **Verification**: Script runs successfully on `dragonbg` without OOM, logs loss progression, and verifies targeted ROI activation.

### M4. FastAPI Backend
- **Agent Type**: `teamwork_preview_worker`
- **Role**: `Backend Engineer`
- **Tasks**:
  - Implement `server.py` with necessary endpoints: `/rois`, `/preview`, `/generate`.
  - Support WebSocket streaming for progress and preview images.
- **Verification**: Server start confirmation and endpoint smoke tests.

### M5. React UI
- **Agent Type**: `teamwork_preview_worker`
- **Role**: `Frontend Developer`
- **Tasks**:
  - Set up a React app inside `ui/` locally on the dev machine.
  - Build UI layout with sliders for ROIs, preview/generate status visualization, and image rendering.
- **Verification**: Build/compile checks and mock tests.

### M6. E2E Integration & Verification
- **Agent Type**: `teamwork_preview_reviewer` / `teamwork_preview_auditor`
- **Tasks**:
  - Run the full system: UI -> Backend -> generate loop -> tribe wrapper.
  - Audit implementation integrity.
- **Verification**: Verified results on both remote and local.

## Workflow Gating & Review
- Each milestone must be gated by code reviews and testing.
- Prior to launching worker tasks, we must perform an exploration step to understand `dragonbg` server access and TRIBE v2 setup.
