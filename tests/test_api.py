from fastapi.testclient import TestClient

import main
from clusters import ClusterStatus, ClustersUnavailableError
from main import create_app
from models import Prediction


VALID_REQUEST = {
    "pupilId": 123,
    "psychTests": {
        "Temperament": {
            "completionTimeSeconds": 17,
            "psychParams": [{"name": "extrav_introver_score", "param": 13}],
            "testTypeName": "Temperament",
            "createdAt": None,
        }
    },
}


def fake_prediction(pupil, _settings):
    return Prediction(
        pupil_id=pupil.pupil_id,
        cluster=1,
        predicted_profession="Profession A",
        nearest_specialist_id=42,
        distance=0.5,
        confidence_category="Очень маленькое",
    )


def test_health_does_not_trigger_cluster_update(settings, monkeypatch):
    monkeypatch.setattr(main, "cluster_status", lambda *_args: ClusterStatus("fresh", None))
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "referenceData": "fresh",
        "lastSuccessfulRefresh": None,
    }


def test_predict_returns_typed_response(settings, monkeypatch):
    monkeypatch.setattr(main, "predict_pupil", fake_prediction)
    client = TestClient(create_app(settings))

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["pupilId"] == 123
    assert response.json()["predictedProfession"] == "Profession A"
    assert set(response.json()) == {
        "pupilId",
        "cluster",
        "predictedProfession",
        "nearestSpecialistId",
        "distance",
        "confidenceCategory",
    }


def test_predict_rejects_invalid_body(settings):
    client = TestClient(create_app(settings))

    response = client.post("/predict", json={"pupilId": -1, "psychTests": {}})

    assert response.status_code == 422


def test_predict_returns_503_when_clusters_are_unavailable(settings, monkeypatch):
    def unavailable(*_args):
        raise ClustersUnavailableError("Cluster data is temporarily unavailable")

    monkeypatch.setattr(main, "predict_pupil", unavailable)
    client = TestClient(create_app(settings))

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "CLUSTERS_UNAVAILABLE"


def test_real_service_returns_safe_503_when_first_refresh_cannot_start(settings):
    client = TestClient(create_app(settings))

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "CLUSTERS_UNAVAILABLE",
            "message": "Cluster data is temporarily unavailable",
        }
    }
