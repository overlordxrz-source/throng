# Project: InverseTribe

## Architecture
InverseTribe consists of a pipeline that optimizes generated image latents using gradients backpropagated from a brain model.
- **`tribe_grad.py`**: A fully differentiable PyTorch wrapper around the TRIBE v2 `FmriEncoder`.
- **`roi_atlas.py`**: Maps standard ROIs (e.g. V1, FFA, Amygdala) to fsaverage5 vertices (20,484 vertices).
- **`baseline_cache.py`**: Downloads 500 COCO validation images, encodes them, and caches mean activations.
- **`generate.py`**: Gradient-guided SDXL loop that applies guidance from the differentiable brain model to latents.
- **`server.py`**: FastAPI backend exposing `/rois`, `/preview`, and `/generate` (WebSocket streaming).
- **`ui/`**: React frontend providing interactive region selection, targets, and progress visualization.

```
React UI  <-- (WebSocket / HTTP) -->  FastAPI Server
                                            |
                                            v
                                     generate.py (SDXL + Differentiable fMRI)
                                            |
                                            v
                                     tribe_grad.py (TRIBE v2 wrapper)
```

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | Environment & Wrapper | Set up dependencies on `dragonbg`; build differentiable PyTorch wrapper `tribe_grad.py` | None | IN_PROGRESS (Conv: 3d64c6cf-cda7-4244-a25e-faa8d5255153) |
| 2 | Region Atlas & Baseline | Build ROI vertices registry (`roi_atlas.py`) and compute cached baseline activations (`baseline_cache.py`) | M1 | PLANNED |
| 3 | SDXL Guidance Loop | Implement gradient-guided diffusion generation loop (`generate.py`) w/ customizable strength | M1, M2 | PLANNED |
| 4 | FastAPI Backend | Implement server endpoints (`server.py`) and WebSocket stream for progress updates | M3 | PLANNED |
| 5 | React UI | Build the browser dashboard frontend in `ui/` | M4 | PLANNED |
| 6 | E2E Integration & Verification | End-to-end testing, validation of acceptance criteria, and auditing | M5 | PLANNED |

*Note: E2E Testing Track is running in parallel (Conv: 2622d0b8-a2d6-48d3-bbbe-043d93d42331).*

## Interface Contracts

### 1. `DifferentiableTribeEncoder` (`tribe_grad.py`)
- **Constructor**: `DifferentiableTribeEncoder(device='cuda')`
- **Method**: `forward(self, x: torch.Tensor) -> torch.Tensor`
  - Input `x`: torch.Tensor of shape `(B, 3, H, W)` containing images normalized to `[0, 1]`. Requires gradients.
  - Output: torch.Tensor of shape `(B, 20484)` representing fsaverage5 vertex activations. Differentiable back to `x`.

### 2. ROI Atlas (`roi_atlas.py`)
- **Function**: `get_vertices(roi_name: str) -> np.ndarray`
  - Input: ROI name (e.g., `"V1"`, `"FFA"`, `"amygdala"`).
  - Output: 1D array of vertex indices (integers, all `< 20484`).
- **Function**: `get_all_supported_rois() -> List[str]`
- **Constraint**: The union of all supported ROIs must cover at least 60% of fsaverage5 vertices.

### 3. Baseline Cache (`baseline_cache.py`)
- **Cache File**: `baseline.pt`
  - Saved via `torch.save()`.
  - Format: torch.Tensor of shape `(20484,)`.
  - Content: Average activations over 500 COCO validation images.
  - Requirements: Finite values, non-zero per-ROI variance.

### 4. Generation Loop (`generate.py`)
- **Function**: `generate_guided_image(prompt: str, targets: Dict[str, float], guidance_strength: float, steps: int = 50) -> Image.Image`
  - Minimizes brain loss w.r.t SDXL latent during denoising.
  - Brain loss: `MSE(predicted_activations[roi_vertices], target_activations)`.

### 5. Backend Server API (`server.py`)
- `GET /rois`: Returns list of support region names.
- `GET /preview`: Returns active targets preview.
- `POST /generate`: Trigger image generation or stream via WebSocket.
- `WS /ws/generate`: Dual-way WebSocket for streaming generation logs and preview images.

## Code Layout
- `tribe_grad.py`
- `roi_atlas.py`
- `baseline_cache.py`
- `generate.py`
- `server.py`
- `ui/`
  - `src/`
    - `App.js` or `App.tsx`
    - `components/`
      - `RoiSelector.jsx`
      - `BrainOutput.jsx`
