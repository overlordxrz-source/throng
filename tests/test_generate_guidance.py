import pytest
import torch
import numpy as np
from PIL import Image

try:
    import generate
    HAS_GENERATE = True
except ImportError:
    HAS_GENERATE = False

pytestmark = pytest.mark.skipif(not HAS_GENERATE, reason="generate.py is not present")

@pytest.fixture(autouse=True)
def setup_encoder_for_generation(monkeypatch, mock_tribe_encoder):
    # Ensure tribe_grad is monkeypatched to use CPU mock encoder if CUDA is not available
    try:
        import tribe_grad
        if not torch.cuda.is_available():
            monkeypatch.setattr(tribe_grad, "DifferentiableTribeEncoder", mock_tribe_encoder)
    except ImportError:
        pass

# Tier 1 - Feature Coverage

def test_guidance_loop_runs():
    img = generate.generate_guided_image(
        prompt="a face",
        targets={"FFA": 1.2},
        guidance_strength=1.0,
        steps=5
    )
    assert isinstance(img, Image.Image)

def test_brain_loss_decreasing(monkeypatch):
    import tribe_grad
    import roi_atlas
    import baseline_cache
    
    activations = []
    original_forward = tribe_grad.DifferentiableTribeEncoder.forward
    
    def wrapped_forward(self, x):
        out = original_forward(self, x)
        activations.append(out.detach().cpu())
        return out
        
    monkeypatch.setattr(tribe_grad.DifferentiableTribeEncoder, "forward", wrapped_forward)
    
    targets = {"FFA": 1.5}
    generate.generate_guided_image(
        prompt="a face",
        targets=targets,
        guidance_strength=10.0,
        steps=5
    )
    
    baseline = baseline_cache.load_baseline()
    ffa_idx = roi_atlas.get_vertices("FFA")
    target_val = 1.5 * baseline[ffa_idx]
    
    losses = []
    for act in activations:
        step_loss = torch.mean((act[0, ffa_idx] - target_val) ** 2).item()
        losses.append(step_loss)
        
    assert len(losses) >= 2
    # The brain loss should trend downward
    assert losses[-1] < losses[0]

def test_guidance_strength_effect(monkeypatch):
    import tribe_grad
    import roi_atlas
    import baseline_cache
    
    activations = []
    original_forward = tribe_grad.DifferentiableTribeEncoder.forward
    
    def wrapped_forward(self, x):
        out = original_forward(self, x)
        activations.append(out.detach().cpu())
        return out
        
    monkeypatch.setattr(tribe_grad.DifferentiableTribeEncoder, "forward", wrapped_forward)
    
    # Run with low strength
    generate.generate_guided_image(
        prompt="a face",
        targets={"FFA": 1.5},
        guidance_strength=0.1,
        steps=5
    )
    low_strength_activations = list(activations)
    
    # Run with high strength
    activations.clear()
    generate.generate_guided_image(
        prompt="a face",
        targets={"FFA": 1.5},
        guidance_strength=10.0,
        steps=5
    )
    high_strength_activations = list(activations)
    
    baseline = baseline_cache.load_baseline()
    ffa_idx = roi_atlas.get_vertices("FFA")
    target_val = 1.5 * baseline[ffa_idx]
    
    def calc_losses(acts):
        return [torch.mean((act[0, ffa_idx] - target_val) ** 2).item() for act in acts]
        
    low_losses = calc_losses(low_strength_activations)
    high_losses = calc_losses(high_strength_activations)
    
    low_reduction = low_losses[0] - low_losses[-1]
    high_reduction = high_losses[0] - high_losses[-1]
    
    # High guidance strength should result in a larger reduction in loss
    assert high_reduction > low_reduction

def test_target_activation_matching(monkeypatch):
    import tribe_grad
    import roi_atlas
    import baseline_cache
    
    activations = []
    original_forward = tribe_grad.DifferentiableTribeEncoder.forward
    
    def wrapped_forward(self, x):
        out = original_forward(self, x)
        activations.append(out.detach().cpu())
        return out
        
    monkeypatch.setattr(tribe_grad.DifferentiableTribeEncoder, "forward", wrapped_forward)
    
    generate.generate_guided_image(
        prompt="a face",
        targets={"FFA": 1.5},
        guidance_strength=10.0,
        steps=5
    )
    
    baseline = baseline_cache.load_baseline()
    ffa_idx = roi_atlas.get_vertices("FFA")
    
    final_ffa_mean = activations[-1][0, ffa_idx].mean().item()
    baseline_ffa_mean = baseline[ffa_idx].mean().item()
    
    # Activating FFA should yield mean activations higher than the baseline mean
    assert final_ffa_mean > baseline_ffa_mean

def test_output_image_properties():
    img = generate.generate_guided_image(
        prompt="test",
        targets={"V1": 1.0},
        guidance_strength=1.0,
        steps=3
    )
    assert isinstance(img, Image.Image)
    assert img.size in [(512, 512), (1024, 1024)]

# Tier 2 - Boundary & Corner Cases

def test_zero_guidance_strength():
    img = generate.generate_guided_image(
        prompt="test",
        targets={"FFA": 1.5},
        guidance_strength=0.0,
        steps=3
    )
    assert isinstance(img, Image.Image)

def test_negative_guidance_strength():
    img = generate.generate_guided_image(
        prompt="test",
        targets={"FFA": 1.5},
        guidance_strength=-5.0,
        steps=3
    )
    assert isinstance(img, Image.Image)

def test_extreme_steps():
    img1 = generate.generate_guided_image(
        prompt="test",
        targets={"FFA": 1.2},
        guidance_strength=1.0,
        steps=1
    )
    assert img1 is not None
    
    img10 = generate.generate_guided_image(
        prompt="test",
        targets={"FFA": 1.2},
        guidance_strength=1.0,
        steps=10
    )
    assert img10 is not None

def test_empty_targets():
    img = generate.generate_guided_image(
        prompt="test",
        targets={},
        guidance_strength=1.0,
        steps=3
    )
    assert img is not None

def test_invalid_target_rois():
    with pytest.raises(ValueError):
        generate.generate_guided_image(
            prompt="test",
            targets={"INVALID_ROI": 1.5},
            guidance_strength=1.0,
            steps=3
        )

# Tier 3 - Cross-Feature Combinations

def test_multi_roi_conflict_resolution(monkeypatch):
    import tribe_grad
    import roi_atlas
    import baseline_cache
    
    # Target high FFA and low Amygdala simultaneously
    targets = {"FFA": 1.3, "amygdala": 0.2}
    
    activations = []
    original_forward = tribe_grad.DifferentiableTribeEncoder.forward
    
    def wrapped_forward(self, x):
        out = original_forward(self, x)
        activations.append(out.detach().cpu())
        return out
        
    monkeypatch.setattr(tribe_grad.DifferentiableTribeEncoder, "forward", wrapped_forward)
    
    generate.generate_guided_image(
        prompt="a face",
        targets=targets,
        guidance_strength=10.0,
        steps=5
    )
    
    baseline = baseline_cache.load_baseline()
    ffa_idx = roi_atlas.get_vertices("FFA")
    amygdala_idx = roi_atlas.get_vertices("amygdala")
    
    final_ffa = activations[-1][0, ffa_idx].mean().item()
    final_amygdala = activations[-1][0, amygdala_idx].mean().item()
    
    # Check that they follow the correct relative trends matching target directions
    assert final_ffa > baseline[ffa_idx].mean().item() * 0.9
    assert final_amygdala < baseline[amygdala_idx].mean().item() * 1.1
