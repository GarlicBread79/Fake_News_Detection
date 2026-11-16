from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import main


client = TestClient(main.app)


class StubDetector:
    def predict(self, text: str) -> dict[str, str]:
        return {
            "Logistic Regression": "Real News",
            "BERT": "Fake News",
        }


def test_health_reports_degraded_when_artifacts_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(main, "LOGISTIC_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(main, "VECTORIZER_PATH", tmp_path / "vectorizer.pkl")
    monkeypatch.setattr(main, "BERT_MODEL_PATH", tmp_path / "bert")

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "prediction_ready": False}


def test_health_reports_ready_when_all_artifacts_exist(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    bert_path = tmp_path / "bert"
    bert_path.mkdir()
    model_path.write_bytes(b"model")
    vectorizer_path.write_bytes(b"vectorizer")
    (bert_path / "config.json").write_text("{}")
    (bert_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(main, "LOGISTIC_MODEL_PATH", model_path)
    monkeypatch.setattr(main, "VECTORIZER_PATH", vectorizer_path)
    monkeypatch.setattr(main, "BERT_MODEL_PATH", bert_path)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "prediction_ready": True}


def test_predict_returns_both_model_predictions(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_detector", lambda: StubDetector())

    response = client.post(
        "/api/v1/predict", json={"text": "An article to analyze"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": {
            "Logistic Regression": "Real News",
            "BERT": "Fake News",
        }
    }


def test_predict_rejects_blank_article() -> None:
    response = client.post("/api/v1/predict", json={"text": "   "})

    assert response.status_code == 422


def test_predict_rejects_oversized_article() -> None:
    response = client.post("/api/v1/predict", json={"text": "x" * 20_001})

    assert response.status_code == 422
