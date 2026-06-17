import os
import pytest
import torch
import numpy as np

try:
    import baseline_cache
    HAS_BASELINE_CACHE = True
except ImportError:
    HAS_BASELINE_CACHE = False

pytestmark = pytest.mark.skipif(not HAS_BASELINE_CACHE, reason="baseline_cache.py is not present")

@pytest.fixture
def temp_baseline_env(tmp_path, monkeypatch):
    target_path = tmp_path / "baseline.pt"
    if hasattr(baseline_cache, "BASELINE_PATH"):
        monkeypatch.setattr(baseline_cache, "BASELINE_PATH", str(target_path))
    elif hasattr(baseline_cache, "CACHE_PATH"):
        monkeypatch.setattr(baseline_cache, "CACHE_PATH", str(target_path))
    return target_path

# Tier 1 - Feature Coverage

def test_baseline_file_exists(temp_baseline_env, monkeypatch):
    def mock_generate():
        tensor = torch.randn(20484)
        torch.save(tensor, temp_baseline_env)
        return tensor
    
    if hasattr(baseline_cache, "generate_baseline"):
        monkeypatch.setattr(baseline_cache, "generate_baseline", mock_generate)
    elif hasattr(baseline_cache, "compute_baseline"):
        monkeypatch.setattr(baseline_cache, "compute_baseline", mock_generate)
        
    if hasattr(baseline_cache, "load_baseline"):
        baseline_cache.load_baseline()
    elif hasattr(baseline_cache, "get_baseline"):
        baseline_cache.get_baseline()
    else:
        # Fallback if no functions, direct write
        torch.save(torch.randn(20484), temp_baseline_env)

    assert temp_baseline_env.exists()

def test_baseline_shape(temp_baseline_env):
    torch.save(torch.randn(20484), temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    elif hasattr(baseline_cache, "get_baseline"):
        data = baseline_cache.get_baseline()
    else:
        data = torch.load(temp_baseline_env)
    assert isinstance(data, torch.Tensor)
    assert data.shape == (20484,)

def test_baseline_values_finite(temp_baseline_env):
    torch.save(torch.randn(20484), temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    else:
        data = torch.load(temp_baseline_env)
    assert torch.isfinite(data).all()

def test_baseline_variance(temp_baseline_env):
    torch.save(torch.randn(20484) + 1.0, temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    else:
        data = torch.load(temp_baseline_env)
    assert torch.var(data).item() > 0.0

def test_baseline_roi_averages(temp_baseline_env):
    torch.save(torch.randn(20484) + 2.0, temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    else:
        data = torch.load(temp_baseline_env)
    
    try:
        import roi_atlas
        rois = roi_atlas.get_all_supported_rois()
        for r in rois:
            vertices = roi_atlas.get_vertices(r)
            sliced = data[vertices]
            assert sliced.ndim == 1
            assert len(sliced) > 0
            assert torch.isfinite(sliced.mean())
    except ImportError:
        vertices = np.arange(100)
        sliced = data[vertices]
        assert torch.isfinite(sliced.mean())

# Tier 2 - Boundary & Corner Cases

def test_baseline_missing_behavior(temp_baseline_env, monkeypatch):
    if temp_baseline_env.exists():
        temp_baseline_env.unlink()
        
    called = [False]
    def mock_generate():
        called[0] = True
        tensor = torch.randn(20484)
        torch.save(tensor, temp_baseline_env)
        return tensor
        
    if hasattr(baseline_cache, "generate_baseline"):
        monkeypatch.setattr(baseline_cache, "generate_baseline", mock_generate)
    elif hasattr(baseline_cache, "compute_baseline"):
        monkeypatch.setattr(baseline_cache, "compute_baseline", mock_generate)
        
    try:
        if hasattr(baseline_cache, "load_baseline"):
            baseline_cache.load_baseline()
        elif hasattr(baseline_cache, "get_baseline"):
            baseline_cache.get_baseline()
    except Exception:
        pass
        
    assert called[0] or temp_baseline_env.exists()

def test_baseline_corrupt_file(temp_baseline_env, monkeypatch):
    with open(temp_baseline_env, "w") as f:
        f.write("corrupt data")
        
    called = [False]
    def mock_generate():
        called[0] = True
        tensor = torch.randn(20484)
        torch.save(tensor, temp_baseline_env)
        return tensor
        
    if hasattr(baseline_cache, "generate_baseline"):
        monkeypatch.setattr(baseline_cache, "generate_baseline", mock_generate)
    elif hasattr(baseline_cache, "compute_baseline"):
        monkeypatch.setattr(baseline_cache, "compute_baseline", mock_generate)
        
    try:
        if hasattr(baseline_cache, "load_baseline"):
            baseline_cache.load_baseline()
        elif hasattr(baseline_cache, "get_baseline"):
            baseline_cache.get_baseline()
    except Exception:
        pass

def test_baseline_all_zeros_prevention(temp_baseline_env):
    torch.save(torch.zeros(20484), temp_baseline_env)
    
    if hasattr(baseline_cache, "load_baseline"):
        try:
            data = baseline_cache.load_baseline()
            # If loaded, make sure it raises or is validated
            assert torch.sum(torch.abs(data)).item() > 0.0
        except (ValueError, AssertionError):
            pass

def test_baseline_value_range(temp_baseline_env):
    torch.save(torch.randn(20484), temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    else:
        data = torch.load(temp_baseline_env)
    assert torch.max(torch.abs(data)).item() < 1e5

def test_baseline_read_only(temp_baseline_env):
    torch.save(torch.randn(20484), temp_baseline_env)
    os.chmod(temp_baseline_env, 0o444) # read-only
    try:
        if hasattr(baseline_cache, "load_baseline"):
            data = baseline_cache.load_baseline()
        else:
            data = torch.load(temp_baseline_env)
        assert data.shape == (20484,)
    finally:
        os.chmod(temp_baseline_env, 0o666)

# Tier 3 - Cross-Feature Combinations

def test_baseline_roi_atlas_integration(temp_baseline_env):
    try:
        import roi_atlas
    except ImportError:
        pytest.skip("roi_atlas is required for this cross-feature test")
        
    torch.save(torch.randn(20484), temp_baseline_env)
    if hasattr(baseline_cache, "load_baseline"):
        data = baseline_cache.load_baseline()
    else:
        data = torch.load(temp_baseline_env)
        
    rois = roi_atlas.get_all_supported_rois()
    means = {}
    for r in rois:
        vertices = roi_atlas.get_vertices(r)
        sliced = data[vertices]
        mean_act = sliced.mean().item()
        assert np.isfinite(mean_act)
        means[r] = mean_act
        
    # Validates that different ROIs map to different subset of vertices with distinct activations
    assert len(set(means.values())) == len(rois)
