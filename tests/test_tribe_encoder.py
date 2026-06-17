import pytest
import torch
import numpy as np

try:
    import tribe_grad
    HAS_TRIBE_GRAD = True
except ImportError:
    HAS_TRIBE_GRAD = False

pytestmark = pytest.mark.skipif(not HAS_TRIBE_GRAD, reason="tribe_grad.py is not present")

@pytest.fixture(autouse=True)
def setup_encoder(monkeypatch, mock_tribe_encoder):
    # If tribe_grad is present, check if CUDA is available.
    # If not, monkeypatch tribe_grad.DifferentiableTribeEncoder to use mock_tribe_encoder.
    if HAS_TRIBE_GRAD:
        if not torch.cuda.is_available():
            monkeypatch.setattr(tribe_grad, "DifferentiableTribeEncoder", mock_tribe_encoder)

# Tier 1 - Feature Coverage

def test_encoder_initialization():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    assert encoder is not None
    
    if torch.cuda.is_available():
        encoder_cuda = tribe_grad.DifferentiableTribeEncoder(device='cuda')
        assert encoder_cuda is not None

def test_encoder_forward_shape():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 20484)

def test_encoder_backward_gradients():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_encoder_gradient_magnitude():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert torch.any(x.grad != 0.0)

def test_encoder_batch_processing():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(4, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    assert out.shape == (4, 20484)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

# Tier 2 - Boundary & Corner Cases

def test_encoder_empty_batch():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(0, 3, 224, 224, requires_grad=True)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        encoder(x)

def test_encoder_invalid_image_dims():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x_wrong_channel = torch.randn(1, 1, 224, 224)
    with pytest.raises((ValueError, RuntimeError)):
        encoder(x_wrong_channel)
    
    x_wrong_ndim = torch.randn(3, 224, 224)
    with pytest.raises((ValueError, RuntimeError)):
        encoder(x_wrong_ndim)

def test_encoder_extreme_pixel_values():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224) * 100.0
    out = encoder(x)
    assert torch.isfinite(out).all()

def test_encoder_nan_inf_inputs():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224)
    x[0, 0, 0, 0] = float('nan')
    try:
        out = encoder(x)
        assert torch.isnan(out).any() or not torch.isfinite(out).all()
    except (ValueError, RuntimeError):
        pass

def test_encoder_requires_grad_false():
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224, requires_grad=False)
    out = encoder(x)
    loss = out.sum()
    with pytest.raises((RuntimeError, AssertionError)):
        loss.backward()
        assert x.grad is None

# Tier 3 - Cross-Feature Combinations

def test_encoder_roi_atlas_integration():
    try:
        import roi_atlas
    except ImportError:
        pytest.skip("roi_atlas is required for this cross-feature test")
        
    encoder = tribe_grad.DifferentiableTribeEncoder(device='cpu')
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    
    ffa_vertices = roi_atlas.get_vertices("FFA")
    target_activations = out[0, ffa_vertices]
    
    loss = target_activations.mean()
    loss.backward()
    
    assert x.grad is not None
    assert torch.any(x.grad != 0.0)
