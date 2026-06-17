# Original User Request

## Initial Request — 2026-06-15T22:44:01Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

A system where you pick brain regions, set how active you want them (e.g. "V1 at 80%, amygdala at −30%"), and the system generates an image that — when viewed by a human — should produce exactly that pattern of brain activity.

Working directory (Local): `~/teamwork_projects/inversetribe`
Working directory (Remote): `~/inversetribe` on `dragonbg`
Integrity mode: benchmark

## Requirements

### R1. Differentiable TRIBE v2 Wrapper (`tribe_grad.py`)
Implement a fully differentiable PyTorch wrapper around the TRIBE v2 `FmriEncoder`. Bypass non-differentiable preprocessing steps (like PIL image loading) so gradients flow from cortical activations back to pixel space. Do not modify TRIBE v2's source code; wrap it from the outside.

### R2. Region Mapping & Baseline Cache (`roi_atlas.py`, `baseline_cache.py`)
Map standard brain regions (e.g., V1, FFA, amygdala) to fsaverage5 vertices using NiLearn atlases (HCP MMP1.0 / Destrieux and Wang retinotopic). Write a script to download a sample of 500 natural images from COCO val2017, then compute and cache baseline activations for these vertices over the images to establish "100% activation" targets.

### R3. Gradient-Guided Generation Loop (`generate.py`)
Implement a denoising loop using StableDiffusionXLPipeline that applies gradient guidance from the TRIBE v2 encoder at each step. Modify latents to minimize the MSE between predicted brain activations and target percentages for selected ROIs. Keep `guidance_strength` adjustable.

### R4. FastAPI Backend & React Frontend (`server.py`, `ui/`)
Build a FastAPI server with `/rois`, `/preview`, and `/generate` (WebSocket for streaming progress). Build React frontend (Phase 1) allowing users to select regions, set target percentages, run generation, and view the output image with predicted vs. target activations.

### R5. Execution & Environment
You must connect to the `dragonbg` GPU server via SSH to execute and verify the backend code. The frontend should run locally on the dev machine. Maintain synchronization between the local codebase and the remote execution environment.

## Acceptance Criteria

### Differentiable Wrapper
- [ ] Running a backward pass from the output of `DifferentiableTribeEncoder` to a dummy pixel tensor input results in a non-None gradient (`x.grad is not None`).
- [ ] Gradient magnitude is non-zero for at least some vertices.

### Brain Region Mapping & Baseline
- [ ] `get_vertices("V1")` returns an array of integers, all < 20484.
- [ ] The union of all supported ROI vertices covers at least 60% of the total 20,484 fsaverage5 vertices.
- [ ] `baseline.pt` is successfully generated, has shape `(20484,)`, contains finite values, and per-ROI variance is non-zero.

### Generation & Guidance Loop
- [ ] The generation loop completes without out-of-memory (OOM) errors.
- [ ] Brain loss decreases monotonically (or trends down) over the diffusion steps.
- [ ] Final `brain_pred_final[get_vertices("FFA")]` is demonstrably higher than the baseline when FFA is targeted at >100%.

### End-to-End System
- [ ] The FastAPI server starts successfully and all endpoints respond correctly.
- [ ] The full end-to-end flow from the browser UI to the generated image works as expected.

---

## Reference Material: InverseTribe — Full Spec & Implementation Plan

### 1. What Is This Project?
A system where you pick brain regions, set how active you want them (e.g. "V1 at 80%, amygdala at −30%"), and the system generates an image that — when viewed by a human — should produce exactly that pattern of brain activity. This is a research prototype combining TRIBE v2 (Meta FAIR, March 2026) and Stable Diffusion XL.

### 2. Background Context
TRIBE v2 solves the forward problem (stimulus → brain response). We want to solve the inverse problem (desired brain response → stimulus) via activation-targeted stimulus synthesis. Because TRIBE v2 is implemented in PyTorch and is differentiable end-to-end, we can backpropagate gradients from the brain activation output all the way back to pixel space, optimizing an image to produce any target brain state (building upon the BrainDiVE and BrainACTIV precedents).

### 3. System Architecture
Hardware Setup:
- **dragonbg**: GPU server. Runs the backend. Both TRIBE v2 and SDXL load here.
- **Dev machine (MacBook M5)**: Runs the React frontend.
- **Communication**: FastAPI with a WebSocket endpoint for streaming generation progress.

### 4. Module Implementation Plan
- **Phase 0: Environment Setup** (On dragonbg: install tribev2, SDXL, verify VRAM).
- **Module 1: `tribe_grad.py`** (Differentiable TRIBE v2 Wrapper). Must use PyTorch equivalents instead of PIL/numpy.
- **Module 2: `roi_atlas.py`** (Brain Region Registry). Map ~50 human-readable region names to lists of fsaverage5 vertex indices.
- **Module 3: `baseline_cache.py`** (Baseline Activation Precomputation). Download 500 COCO images, run through encoder, compute mean activation per vertex.
- **Module 4: `generate.py`** (Gradient-Guided Generation Loop). Build target vector, compute brain loss, apply gradient w.r.t latent during SDXL denoising steps.
- **Module 5: `server.py`** (FastAPI Backend). Endpoints: `/rois`, `/generate` (POST, WebSocket stream), `/preview`.
- **Module 6: `ui/`** (React Frontend). Phase 1: RoiSelector, App, BrainOutput.

### 5. Known Risks & Mitigations
- **Risk 1: VRAM.** TRIBE v2 + SDXL might OOM. Mitigations: fp16, empty_cache, CPU offload for TRIBE, or use SD 1.5.
- **Risk 2: Non-differentiable TRIBE v2 preprocessing.** Identify and replace PIL/numpy ops with `torchvision.transforms.functional` equivalents. If backbone has non-diff ops, fallback to a surrogate encoder (e.g. CLIP ViT-L/14) for guidance, use TRIBE only for preview.
- **Risk 3: Gradient vanishing.** Check gradient magnitude, normalize if needed.
- **Risk 4: Adversarial noise.** Lower guidance_strength, add perceptual/aesthetic regularizer, increase steps.

### 6. Implementation Order
1. Environment setup on dragonbg
2. tribe_grad.py — differentiable wrapper
3. roi_atlas.py — ROI registry
4. baseline_cache.py — precompute baseline
5. generate.py — gradient guidance loop
6. server.py — FastAPI + WebSocket
7. ui/ Phase 1 — React dropdown + sliders

### 7. Coding Agent Instructions
- Work in the order specified in Section 6. Do not skip steps.
- After completing each module, run its success criteria check.
- Check Risk #1 (VRAM) and Risk #2 (gradient flow) first before building the rest.
- The `guidance_strength` parameter is scientifically important. Make it easy to tune.

## Follow-up — 2026-06-15T23:00:26Z

Urgent Update from User regarding R5 (Execution & Environment): 

The 'dragonbg' GPU server is actually hosted on Modal. The standard SSH commands will not work. To connect to it, the correct command is:
`modal shell ta-01KV6PEESNZMDNR82XFKR143JX`

Please update your execution strategy across the team. Since `modal shell` is interactive, you may want to adapt your bash commands to work with it, or alternatively, write the code locally and use `modal run` to execute scripts on the remote GPU. Please course-correct your worker agents who are currently attempting standard SSH connections.
