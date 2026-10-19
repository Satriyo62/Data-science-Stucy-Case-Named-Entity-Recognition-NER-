import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# 1. Uji masukan normal
def test_extract_entities_normal():
    payload = {
        "title": "Microsoft Expands Data Center in London",
        "body": "Satya Nadella announced Microsoft is investing heavily in London."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "meta" in data
    assert data["meta"]["entity_count"] > 0
    # Verifikasi skema setiap entitas
    for ent in data["entities"]:
        assert "text" in ent
        assert "label" in ent
        assert "start" in ent
        assert "end" in ent
        assert ent["start"] < ent["end"]

# 2. Uji masukan kosong (wajib mengembalikan HTTP 400)
def test_extract_entities_empty_input():
    # Title kosong
    res1 = client.post("/extract-entities", json={"title": "", "body": "Valid body text"})
    assert res1.status_code == 400

    # Body kosong
    res2 = client.post("/extract-entities", json={"title": "Valid title", "body": "   "})
    assert res2.status_code == 400

# 3. Uji tanda baca dan teks non-ASCII
def test_extract_entities_punctuation_and_non_ascii():
    payload = {
        "title": "René & Co. visit Tokyo's Shibuya-ku district!",
        "body": "Café owners in Zürich and Tokyo signed a €50,000 deal on 2026-05-12."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["entities"], list)

# 4. Uji entitas yang muncul berulang
def test_extract_entities_repeated():
    payload = {
        "title": "Travel Guide to France",
        "body": "France is a popular country in Europe. Millions of travelers visit France Every year."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    apple_entities = [e for e in data["entities"] if e["text"] == "France"]
    # Memastikan model mendeteksi kemunculan entitas lebih dari 1 kali
    assert len(apple_entities) >= 2

# 5. Uji teks yang melampaui batas panjang maksimum (wajib HTTP 400)
def test_extract_entities_exceeds_max_length():
    payload = {
        "title": "Excessive Text",
        "body": "A" * 20000  # Melebihi MAX_INPUT_LENGTH = 15000
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "exceeds maximum" in data["detail"]