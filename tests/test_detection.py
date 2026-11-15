import pytest
from fastapi.testclient import TestClient

from app.intent import detect_intent, extract_single_url
from app.main import app
from app.predictors import (
    fuse_all,
    heuristic_text_score,
    heuristic_url_score,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Check https://example.com/path?q=1.", "https://example.com/path?q=1"),
        ("Visit example.org/login", "https://example.org/login"),
        ("There is no link here", None),
    ],
)
def test_extract_single_url(text, expected):
    assert extract_single_url(text) == expected


@pytest.mark.parametrize(
    ("query", "has_file", "expected"),
    [
        ("", True, "image_check"),
        ("Is this website safe? https://example.com", False, "url_check"),
        ("Please check this message for me", True, "combined"),
        ("Urgent: verify your account", False, "text_message"),
    ],
)
def test_detect_intent(query, has_file, expected):
    assert detect_intent(query, has_file=has_file) == expected


def test_heuristic_text_score_is_bounded_and_rewards_suspicious_content():
    benign = heuristic_text_score("The meeting starts at 10 tomorrow.")
    suspicious = heuristic_text_score("Urgent: verify your account now.")

    assert 0.0 <= benign <= 1.0
    assert 0.0 <= suspicious <= 1.0
    assert suspicious > benign


def test_heuristic_url_score_returns_normalized_probabilities():
    scores = heuristic_url_score("https://paypal-secure-login.xyz")

    assert set(scores) == {"benign", "phishing", "malware"}
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=2e-4)
    assert scores["phishing"] > scores["benign"]


def test_fuse_all_combines_text_and_url_probabilities():
    result = fuse_all(text_p=0.8, url_p_bin=0.2, img_p=None)

    assert result["fused_prob"] == pytest.approx(0.68)
    assert result["label"] == "phishing"
    assert result["probs"] == {"benign": 0.32, "phishing": 0.68}


def test_text_check_returns_prediction_and_explanation(monkeypatch):
    monkeypatch.setattr(
        "app.main.predict_text_probs",
        lambda _text: {"benign": 0.1, "phishing": 0.9},
    )
    monkeypatch.setattr(
        "app.main.make_llm_explanation",
        lambda _summary: "Phishing indicators were detected.",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/text_check",
            json={"text": "Verify your account"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "label": "phishing",
        "probs": {"benign": 0.1, "phishing": 0.9},
        "explanation": "Phishing indicators were detected.",
        "intent": "text_message",
    }
