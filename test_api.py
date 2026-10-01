from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_spam():
    r = client.post("/predict", json={"text": "WINNER!! Claim your free prize now, call 09061701461"})
    assert r.json()["label"] == "spam"


def test_ham():
    r = client.post("/predict", json={"text": "Are we still meeting for lunch tomorrow?"})
    assert r.json()["label"] == "ham"


def test_empty_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422
