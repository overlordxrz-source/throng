import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Mock DifferentiableTribeEncoder
class MockDifferentiableTribeEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Project from pooled 3x64x64 to 20484 vertices
        self.fc = nn.Linear(3 * 64 * 64, 20484)
        # Initialize weight to non-zero values so gradients are non-zero
        nn.init.normal_(self.fc.weight, std=0.01)
        self.to(device)

    def forward(self, x):
        # Input shape: (B, 3, H, W)
        # Output shape: (B, 20484)
        B = x.shape[0]
        # Differentiable resize/interpolate to (64, 64)
        x_resized = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x_flat = x_resized.view(B, -1)
        out = self.fc(x_flat)
        return out

@pytest.fixture
def mock_tribe_encoder():
    return MockDifferentiableTribeEncoder

# 2. Mock Stable Diffusion Pipeline
class MockSDXLImg2ImgPipeline:
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cpu")
        # Add basic attributes that might be used by generate.py
        self.vae = MagicMock()
        self.unet = MagicMock()
        self.scheduler = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, prompt=None, image=None, strength=0.0, num_inference_steps=50, generator=None, **kwargs):
        # Returns a mock output containing PIL images.
        img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        
        class PipelineOutput:
            def __init__(self, images):
                self.images = images
        
        return PipelineOutput([img])

# Mock the diffusers module if it doesn't exist
if "diffusers" not in sys.modules:
    mock_diffusers = MagicMock()
    mock_diffusers.StableDiffusionXLImg2ImgPipeline = MockSDXLImg2ImgPipeline
    mock_diffusers.StableDiffusionXLPipeline = MockSDXLImg2ImgPipeline
    sys.modules["diffusers"] = mock_diffusers

# Mock tribe_grad or dragonbg modules if they don't exist to prevent import errors in generate.py
if "dragonbg" not in sys.modules:
    sys.modules["dragonbg"] = MagicMock()

@pytest.fixture
def mock_sd_pipeline():
    return MockSDXLImg2ImgPipeline

# 3. TestClient fixture for FastAPI (server.py)
@pytest.fixture
def test_client():
    try:
        from server import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        return None
