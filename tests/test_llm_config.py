from __future__ import annotations

from backend.transformer.extractors.llm_extractor import configured_keys
from backend.transformer.gemini_hybrid import configured_gemini_keys


def test_configured_keys_supports_single_comma_and_numbered_env(monkeypatch):
    for name in ["OPENROUTER_API_KEY", "OPENROUTER_KEYS", *[f"OPENROUTER_KEY_{index}" for index in range(1, 6)]]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key_a, key_b")
    monkeypatch.setenv("OPENROUTER_KEYS", "key_b key_c")
    monkeypatch.setenv("OPENROUTER_KEY_1", "key_d")
    monkeypatch.setenv("OPENROUTER_KEY_2", "key_a")
    monkeypatch.setenv("OPENROUTER_KEY_5", "key_e")

    assert configured_keys() == ["key_a", "key_b", "key_c", "key_d", "key_e"]


def test_gemini_keys_support_lowercase_numbered_env(monkeypatch):
    for name in [
        "GEMINI_KEYS",
        *[f"gem{index}" for index in range(1, 6)],
        *[f"GEM{index}" for index in range(1, 6)],
        *[f"GEMINI_KEY_{index}" for index in range(1, 6)],
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("gem1", "gem_a")
    monkeypatch.setenv("gem2", "gem_b")
    monkeypatch.setenv("GEMINI_KEY_1", "gem_c")
    monkeypatch.setenv("GEMINI_KEYS", "gem_d, gem_a")

    assert configured_gemini_keys() == ["gem_a", "gem_b", "gem_c", "gem_d"]
