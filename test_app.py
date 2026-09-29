"""Unit tests for single-file application."""

import pytest
from unittest.mock import MagicMock, patch
from app import (
    translate_text,
    parse_llm_json,
    offline_fallback_translate,
    build_system_prompt,
)


def test_build_system_prompt():
    prompt = build_system_prompt("English", "Slang → Professional", "Formal")
    assert "English" in prompt
    assert "Formal" in prompt


def test_parse_llm_json_valid():
    raw = '{"translation": "Hello friend", "detected_slang": [{"term": "fam", "meaning": "friend"}]}'
    parsed = parse_llm_json(raw)
    assert parsed["translation"] == "Hello friend"


def test_offline_fallback():
    res = offline_fallback_translate("He ghosted me", "English", "Slang → Professional")
    assert res["offline"] is True
    assert "stopped responding" in res["translation"]


@patch("app.Groq")
def test_translate_text_success(mock_groq):
    mock_client = MagicMock()
    mock_groq.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"translation": "He is the best.", "detected_slang": [{"term": "GOAT", "meaning": "Greatest of all time"}]}'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}):
        res = translate_text("He is the GOAT.", "English", "Slang → Professional", "Professional")
        assert res["translation"] == "He is the best."
        assert res["offline"] is False