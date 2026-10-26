import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_checkpoint, monkeypatch, tmp_path):
    monkeypatch.setenv("FER_CHECKPOINT", str(tiny_checkpoint))
    monkeypatch.setenv("FER_METRICS", str(tmp_path / "missing.json"))
    from app import server

    server.get_recognizer.cache_clear()
    with TestClient(server.app) as c:
        yield c
    server.get_recognizer.cache_clear()


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_index_serves_the_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AI Emotion Recognition" in r.text


def test_info_reports_model(client):
    info = client.get("/api/info").json()
    assert info["model"]["arch"] == "small_cnn"
    assert info["detector"] in {"yunet", "haar"}
    assert len(info["emotions"]) == 7


def test_analyze_blank_frame_returns_no_faces(client):
    r = client.post("/api/analyze", files={"image": ("f.jpg", jpeg(np.zeros((240, 320, 3), np.uint8)), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["faces"] == []
    assert body["image"] == {"width": 320, "height": 240}


def test_analyze_rejects_non_images(client):
    r = client.post("/api/analyze", files={"image": ("f.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
