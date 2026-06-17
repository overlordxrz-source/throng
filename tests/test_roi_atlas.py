import pytest
import numpy as np

try:
    import roi_atlas
    HAS_ROI_ATLAS = True
except ImportError:
    HAS_ROI_ATLAS = False

pytestmark = pytest.mark.skipif(not HAS_ROI_ATLAS, reason="roi_atlas.py is not present")

# Tier 1 - Feature Coverage

def test_get_vertices_v1():
    vertices = roi_atlas.get_vertices("V1")
    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 1
    assert np.issubdtype(vertices.dtype, np.integer)
    assert np.all(vertices < 20484)

def test_get_vertices_ffa():
    vertices = roi_atlas.get_vertices("FFA")
    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 1
    assert np.issubdtype(vertices.dtype, np.integer)
    assert np.all(vertices < 20484)

def test_get_vertices_amygdala():
    vertices = roi_atlas.get_vertices("amygdala")
    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 1
    assert np.issubdtype(vertices.dtype, np.integer)
    assert np.all(vertices < 20484)

def test_get_all_supported_rois():
    rois = roi_atlas.get_all_supported_rois()
    assert isinstance(rois, list)
    normalized_rois = [r.lower() for r in rois]
    assert "v1" in normalized_rois
    assert "ffa" in normalized_rois
    assert "amygdala" in normalized_rois

def test_vertex_coverage_threshold():
    rois = roi_atlas.get_all_supported_rois()
    union_vertices = set()
    for r in rois:
        vertices = roi_atlas.get_vertices(r)
        union_vertices.update(vertices.tolist())
    # The union of all supported ROIs must cover at least 60% of fsaverage5 vertices (12290 vertices)
    assert len(union_vertices) >= 12290

def test_roi_case_insensitivity():
    v1_upper = roi_atlas.get_vertices("V1")
    v1_lower = roi_atlas.get_vertices("v1")
    v1_mixed = roi_atlas.get_vertices("V1")
    np.testing.assert_array_equal(v1_upper, v1_lower)
    np.testing.assert_array_equal(v1_upper, v1_mixed)

# Tier 2 - Boundary & Corner Cases

def test_get_vertices_invalid_roi():
    with pytest.raises(ValueError):
        roi_atlas.get_vertices("INVALID_ROI_NAME_12345")

def test_get_vertices_empty_string():
    with pytest.raises(ValueError):
        roi_atlas.get_vertices("")

def test_vertex_indices_bounds():
    rois = roi_atlas.get_all_supported_rois()
    for r in rois:
        vertices = roi_atlas.get_vertices(r)
        assert np.all(vertices >= 0)
        assert np.all(vertices < 20484)

def test_no_duplicate_vertices_per_roi():
    rois = roi_atlas.get_all_supported_rois()
    for r in rois:
        vertices = roi_atlas.get_vertices(r)
        assert len(vertices) == len(np.unique(vertices))

def test_large_random_input_roi():
    with pytest.raises(ValueError):
        roi_atlas.get_vertices("a" * 10000)

# Tier 3 - Cross-Feature Combinations

def test_roi_overlap_properties():
    # Verify anatomical distinction of different ROIs
    v1 = set(roi_atlas.get_vertices("V1").tolist())
    ffa = set(roi_atlas.get_vertices("FFA").tolist())
    intersection = v1.intersection(ffa)
    # Ensure they are distinct and overlap is minor
    assert len(intersection) < 0.05 * min(len(v1), len(ffa))
