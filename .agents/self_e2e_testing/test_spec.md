# InverseTribe E2E Testing Suite Specification

This document details the test cases to be implemented in the `tests/` directory.

## 1. ROI Atlas Tests (`tests/test_roi_atlas.py`)
This file tests the brain region mapping logic defined in `roi_atlas.py`.

### Tier 1 - Feature Coverage:
1. `test_get_vertices_v1`: Verify `get_vertices("V1")` returns a 1D numpy array of integers, all < 20484.
2. `test_get_vertices_ffa`: Verify `get_vertices("FFA")` returns a 1D numpy array of integers, all < 20484.
3. `test_get_vertices_amygdala`: Verify `get_vertices("amygdala")` returns a 1D numpy array of integers, all < 20484.
4. `test_get_all_supported_rois`: Verify `get_all_supported_rois()` returns a list containing at least `"V1"`, `"FFA"`, and `"amygdala"`.
5. `test_vertex_coverage_threshold`: Verify that the union of all vertices from all supported ROIs covers at least 60% of the total 20,484 fsaverage5 vertices (i.e., `len(union_vertices) >= 12290`).
6. `test_roi_case_insensitivity`: Verify `get_vertices` works case-insensitively (e.g. `"v1"` vs `"V1"` vs `"V1"`).

### Tier 2 - Boundary & Corner Cases:
1. `test_get_vertices_invalid_roi`: Verify calling `get_vertices` with an unsupported region name raises `ValueError`.
2. `test_get_vertices_empty_string`: Verify calling `get_vertices` with an empty string `""` raises `ValueError`.
3. `test_vertex_indices_bounds`: Verify all vertex indices returned for all supported ROIs are strictly in the range `[0, 20483]`.
4. `test_no_duplicate_vertices_per_roi`: Verify that there are no duplicate vertex indices within the array returned for any single ROI.
5. `test_large_random_input_roi`: Verify passing a very long string as ROI name raises `ValueError` cleanly.

### Tier 3 - Cross-Feature Combinations:
1. `test_roi_overlap_properties`: Verify overlap properties between ROIs (e.g. check if V1 and FFA overlap or are disjoint, validating anatomical distinction).

---

## 2. Baseline Cache Tests (`tests/test_baseline_cache.py`)
This file tests the baseline precomputation and caching behavior defined in `baseline_cache.py`.

### Tier 1 - Feature Coverage:
1. `test_baseline_file_exists`: Verify that `baseline.pt` is generated (or mock its existence) in the correct path.
2. `test_baseline_shape`: Load `baseline.pt` using `torch.load` and verify it has shape `(20484,)`.
3. `test_baseline_values_finite`: Verify all values in `baseline.pt` are finite (no NaN or Inf).
4. `test_baseline_variance`: Verify the baseline activations have a non-zero variance.
5. `test_baseline_roi_averages`: Verify that slicing `baseline.pt` using vertices from support ROIs yields valid mean activations.

### Tier 2 - Boundary & Corner Cases:
1. `test_baseline_missing_behavior`: Test behavior when `baseline.pt` is missing (verify it triggers computation or downloads, or raises a structured exception that is handled).
2. `test_baseline_corrupt_file`: Verify that loading a corrupted `baseline.pt` file is detected and handled (e.g., triggers regeneration).
3. `test_baseline_all_zeros_prevention`: Verify baseline is not a trivial all-zeros tensor (variance > 0, sum > 0).
4. `test_baseline_value_range`: Verify baseline activations fall within reasonable expected scaling ranges (e.g., standard activations are non-negative or normalized).
5. `test_baseline_read_only`: Verify handling when the directory containing `baseline.pt` is read-only (permission issues).

---

## 3. Differentiable Encoder Tests (`tests/test_tribe_encoder.py`)
This file tests the `DifferentiableTribeEncoder` wrapper in `tribe_grad.py`.

### Tier 1 - Feature Coverage:
1. `test_encoder_initialization`: Verify `DifferentiableTribeEncoder` initializes correctly on CPU and CUDA (mocked if CUDA is unavailable).
2. `test_encoder_forward_shape`: Pass a dummy image tensor of shape `(B, 3, H, W)` and verify output shape is `(B, 20484)`.
3. `test_encoder_backward_gradients`: Verify running a backward pass from output activations to input image tensor results in `x.grad is not None`.
4. `test_encoder_gradient_magnitude`: Verify gradient `x.grad` contains non-zero elements.
5. `test_encoder_batch_processing`: Verify forward and backward pass work with batches of size > 1.

### Tier 2 - Boundary & Corner Cases:
1. `test_encoder_empty_batch`: Verify passing an empty batch (size 0) raises a clean exception or is handled gracefully.
2. `test_encoder_invalid_image_dims`: Verify passing images with invalid dimensions or channels raises an error.
3. `test_encoder_extreme_pixel_values`: Verify passing pixels with values out of `[0, 1]` (e.g. negative or > 1) is handled or raises error.
4. `test_encoder_nan_inf_inputs`: Verify handling of inputs containing NaN or Inf values.
5. `test_encoder_requires_grad_false`: Verify that if input `x` has `requires_grad=False`, trying to compute gradient raises a PyTorch runtime error or is handled.

---

## 4. Denoising Guidance Loop Tests (`tests/test_generate_guidance.py`)
This file tests the gradient-guided diffusion generation loop in `generate.py`.

### Tier 1 - Feature Coverage:
1. `test_guidance_loop_runs`: Verify `generate_guided_image` runs and returns a valid PIL Image without errors.
2. `test_brain_loss_decreasing`: Verify that computed brain loss decreases or trends downward over diffusion steps.
3. `test_guidance_strength_effect`: Verify that higher guidance strength leads to faster or larger reduction in brain loss compared to low strength.
4. `test_target_activation_matching`: Verify final image activations for targeted ROI (e.g. FFA) are higher than baseline when targeted > 100%.
5. `test_output_image_properties`: Verify final returned output is a PIL Image of correct dimensions.

### Tier 2 - Boundary & Corner Cases:
1. `test_zero_guidance_strength`: Verify setting guidance strength to `0.0` works (reverts to standard SDXL denoising).
2. `test_negative_guidance_strength`: Verify setting negative guidance strength works (repels activations from targets).
3. `test_extreme_steps`: Verify running with 1 step vs 100 steps behaves stably.
4. `test_empty_targets`: Verify passing empty target dict `{}` works (no guidance applied).
5. `test_invalid_target_rois`: Verify passing unsupported ROI in target dict raises `ValueError`.

---

## 5. Backend Server Tests (`tests/test_server.py`)
This file tests the FastAPI server endpoints in `server.py`.

### Tier 1 - Feature Coverage:
1. `test_get_rois_endpoint`: Verify `GET /rois` returns 200 OK and lists supported regions.
2. `test_get_preview_endpoint`: Verify `GET /preview` returns 200 OK and target schema.
3. `test_post_generate_endpoint`: Verify `POST /generate` with valid JSON payload returns 200 or 202 status code.
4. `test_websocket_generate_endpoint`: Verify connecting to `WS /ws/generate` streams generation progress messages (log events, step numbers).
5. `test_server_concurrent_requests`: Verify server can handle multiple HTTP queries concurrently.

### Tier 2 - Boundary & Corner Cases:
1. `test_post_generate_invalid_json`: Verify posting invalid JSON to `/generate` returns 422 Unprocessable Entity.
2. `test_post_generate_unsupported_roi`: Verify posting target dict containing unsupported ROI returns 400 Bad Request or 422.
3. `test_websocket_abrupt_disconnect`: Verify WebSocket connection handles abrupt client disconnection without server crash.
4. `test_post_generate_empty_payload`: Verify posting empty target payload is handled gracefully.
5. `test_generate_invalid_types`: Verify posting non-numeric values for target percentages or guidance strength returns validation error.

---

## 6. Tier 4 Real-World Application Scenarios
To be implemented as integrated flows inside `tests/test_integration.py` or within endpoint tests:
1. **Scenario 1**: User targets high FFA (> 120%) and low Amygdala (< 30%). Runs optimization. Verifies final activations match trend.
2. **Scenario 2**: Full WebSocket optimization stream. Client connects, starts generation, receives progressive logs, and downloads output.
3. **Scenario 3**: Baseline recovery. Delete `baseline.pt`, run generation, verify it automatically regenerates baseline and completes optimization.
