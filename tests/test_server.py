import pytest
import concurrent.futures

try:
    import server
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

pytestmark = pytest.mark.skipif(not HAS_SERVER, reason="server.py is not present")

@pytest.fixture(autouse=True)
def check_client(test_client):
    if test_client is None:
        pytest.skip("FastAPI / TestClient is not available or server.py failed to import")

# Tier 1 - Feature Coverage

def test_get_rois_endpoint(test_client):
    response = test_client.get("/rois")
    assert response.status_code == 200
    data = response.json()
    rois = data if isinstance(data, list) else data.get("rois", data.get("regions", []))
    assert any("v1" in r.lower() for r in rois)

def test_get_preview_endpoint(test_client):
    response = test_client.get("/preview")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_post_generate_endpoint(test_client, monkeypatch):
    try:
        import generate
        monkeypatch.setattr(generate, "generate_guided_image", lambda *args, **kwargs: None)
    except ImportError:
        pass
        
    payload = {
        "prompt": "a photo of a cat",
        "targets": {"FFA": 1.2},
        "guidance_strength": 1.0,
        "steps": 5
    }
    response = test_client.post("/generate", json=payload)
    assert response.status_code in [200, 202]

def test_websocket_generate_endpoint(test_client):
    with test_client.websocket_connect("/ws/generate") as websocket:
        websocket.send_json({
            "prompt": "a photo of a cat",
            "targets": {"FFA": 1.2},
            "guidance_strength": 1.0,
            "steps": 3
        })
        messages = []
        for _ in range(5):
            try:
                msg = websocket.receive_json()
                messages.append(msg)
                if "final" in msg or "image" in msg or "url" in msg or ("status" in msg and msg["status"] == "complete"):
                    break
            except Exception:
                break
        assert len(messages) > 0

def test_server_concurrent_requests(test_client):
    def send_req():
        return test_client.get("/rois")
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_req) for _ in range(10)]
        results = [f.result() for f in futures]
        
    for r in results:
        assert r.status_code == 200

# Tier 2 - Boundary & Corner Cases

def test_post_generate_invalid_json(test_client):
    response = test_client.post("/generate", content="invalid json text")
    assert response.status_code == 422

def test_post_generate_unsupported_roi(test_client):
    payload = {
        "prompt": "test",
        "targets": {"UNSUPPORTED_ROI": 1.5},
        "guidance_strength": 1.0
    }
    response = test_client.post("/generate", json=payload)
    assert response.status_code in [400, 422]

def test_websocket_abrupt_disconnect(test_client):
    try:
        with test_client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json({
                "prompt": "test",
                "targets": {"FFA": 1.2},
                "guidance_strength": 1.0,
                "steps": 10
            })
            websocket.receive_json()
    except Exception:
        pass
        
    # Subsequent calls should still work
    response = test_client.get("/rois")
    assert response.status_code == 200

def test_post_generate_empty_payload(test_client):
    response = test_client.post("/generate", json={})
    assert response.status_code in [400, 422]

def test_generate_invalid_types(test_client):
    payload = {
        "prompt": "test",
        "targets": {"FFA": "high"},
        "guidance_strength": "invalid_strength"
    }
    response = test_client.post("/generate", json=payload)
    assert response.status_code == 422

# Tier 3 - Cross-Feature Combinations

def test_ws_preview_state_integration(test_client):
    with test_client.websocket_connect("/ws/generate") as websocket:
        websocket.send_json({
            "prompt": "test face",
            "targets": {"FFA": 1.4, "amygdala": 0.1},
            "guidance_strength": 5.0,
            "steps": 5
        })
        
        # Verify socket initiates and we can query preview state in parallel
        try:
            websocket.receive_json()
        except Exception:
            pass
            
        response = test_client.get("/preview")
        assert response.status_code == 200
        preview_data = response.json()
        if "targets" in preview_data:
            assert preview_data["targets"].get("FFA") == 1.4
