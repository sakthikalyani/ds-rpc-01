from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/test", auth=("Tony", "password123"))
    assert response.status_code == 200

def test_login():
    response = client.get("/login", auth=("Tony", "password123"))
    assert response.status_code == 200

def test_unauthorized():
    response = client.get("/test")
    assert response.status_code == 401  # no credentials = should fail