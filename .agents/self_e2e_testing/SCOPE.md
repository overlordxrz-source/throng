# Scope: E2E Testing Track

## Architecture
The E2E Testing Track designs and implements a comprehensive, opaque-box test suite derived from the InverseTribe requirements. It is decoupled from implementation details and exercises the system through public interfaces.

## Feature Inventory
1. **Differentiable Encoder**: Backpropagation from brain regions back to pixels.
2. **Region mapping**: ROI vertex registration (V1, FFA, amygdala) and >= 60% fsaverage5 vertex coverage.
3. **Baseline caching**: Precomputed mean activations vector cached in `baseline.pt` with non-zero variance.
4. **Gradient guidance**: SDXL denoising loop that guides latents to match target ROI activations.
5. **FastAPI backend**: `/rois`, `/preview`, `/generate` endpoints and WebSocket streaming.
6. **React UI**: Interactive ROI selection, target inputs, progress visualization, and image rendering.

## Test Suite Plan (Dual Track)
We will design test cases matching the 4-tier methodology:
- **Tier 1 - Feature Coverage**: Happy-path tests for each core feature (>=5 per feature).
- **Tier 2 - Boundary & Corner Cases**: Edge inputs, extreme target values, empty lists (>=5 per feature).
- **Tier 3 - Cross-Feature Combinations**: Interactive tests combining multiple features (e.g. FFA guidance + Amygdala guidance, WebSocket stream under network jitter).
- **Tier 4 - Real-World Application Scenarios**: End-to-end user journeys (e.g. generating an image with targeted high FFA and low amygdala, viewing progress, and receiving final output).

## Output
Upon completion, write `TEST_READY.md` at project root specifying the test runner command and coverage metrics.
