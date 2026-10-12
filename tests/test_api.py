
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "age": 30,
        "gender": "Male",
        "tenure": 12,
        "usage_frequency": 15,
        "support_calls": 3,
        "payment_delay": 5,
        "subscription_type": "Standard",
        "contract_length": "Annual",
        "total_spend": 600.0,
        "last_interaction": 10,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(result["top_reasons"], list)


def test_prediction_rejects_zero_tenure():
    payload = {
        "age": 30,
        "gender": "Male",
        "tenure": 0,
        "usage_frequency": 15,
        "support_calls": 3,
        "payment_delay": 5,
        "subscription_type": "Standard",
        "contract_length": "Annual",
        "total_spend": 600.0,
        "last_interaction": 10,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
