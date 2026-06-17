# E2E Test Infra: InverseTribe

## Test Philosophy
- **Opaque-box, requirement-driven**: Tests derive from user requirements, exercising features through public Python APIs, HTTP endpoints, and WebSockets.
- **Methodology**: 4-Tier Test Suite structure ensuring feature coverage, boundary conditions, cross-feature interactions, and real-world application workloads.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Differentiable Encoder | ORIGINAL_REQUEST §R1 | 5      | 5      | ✓      |
| 2 | Region Mapping | ORIGINAL_REQUEST §R2 | 5      | 5      | ✓      |
| 3 | Baseline Caching | ORIGINAL_REQUEST §R2 | 5      | 5      | ✓      |
| 4 | Gradient Guidance | ORIGINAL_REQUEST §R3 | 5      | 5      | ✓      |
| 5 | FastAPI Backend | ORIGINAL_REQUEST §R4 | 5      | 5      | ✓      |
| 6 | React UI | ORIGINAL_REQUEST §R4 | 5      | 5      | ✓      |

## Test Architecture
- **Test Runner**: Pytest (`pytest`) for Python unit/integration tests. Jest/React Testing Library for React UI tests.
- **Invocation**: 
  - Backend: `pytest tests/`
  - Frontend: `npm run test` or `yarn test` within the `ui/` directory.
- **Test Case Format**: Pytest test cases using standard assertions.
- **Directory Layout**:
  ```
  tests/
  ├── conftest.py               # Shared test fixtures (mock models, server client)
  ├── test_tribe_encoder.py     # Feature 1 tests
  ├── test_roi_atlas.py         # Feature 2 tests
  ├── test_baseline_cache.py    # Feature 3 tests
  ├── test_generate_guidance.py # Feature 4 tests
  └── test_server.py            # Feature 5 tests
  ui/src/__tests__/
  └── test_ui.js                # Feature 6 tests (React UI)
  ```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | High FFA / Low Amygdala Generation Journey | F1, F2, F3, F4, F5 | High |
| 2 | ROI Activation Sweep & Preview Validation | F2, F3, F5 | Medium |
| 3 | WebSocket generation with progress feedback | F4, F5, F6 | High |
| 4 | Offline Baseline Recovery & Run | F3, F4 | Medium |
| 5 | End-to-end user image custom optimization | F1, F2, F3, F4, F5, F6 | High |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥5 tests per feature (30 total). Must pass 100%.
- **Tier 2 (Boundary & Corner Cases)**: ≥5 tests per feature (30 total). Must pass 100%.
- **Tier 3 (Cross-Feature Combinations)**: ≥6 tests covering major interactions. Must pass 100%.
- **Tier 4 (Real-World Application)**: ≥5 end-to-end integration workflows. Must pass 100%.
