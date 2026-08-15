import pytest

from llm_spam_organizer.config import ConfigError, parse_config

MINIMAL = {"imap": {"host": "imap.example.com", "username": "me", "password": "s3cret"}}


def test_defaults_are_filled_in():
    config = parse_config(MINIMAL)
    assert config.imap.port == 993
    assert config.imap.spam_folder == "Junk"
    assert config.ollama.threshold == 0.5
    assert config.evaluation.archive_folders == []


def test_password_command_uses_the_first_output_line():
    config = parse_config(
        {
            "imap": {
                "host": "imap.example.com",
                "username": "me",
                "password_command": ["printf", "hunter2\nignored"],
            }
        }
    )
    assert config.imap.password == "hunter2"


def test_password_and_command_are_exclusive():
    with pytest.raises(ConfigError, match="not both"):
        parse_config({"imap": MINIMAL["imap"] | {"password_command": "true"}})


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="Unknown keys in \\[ollama\\]"):
        parse_config(MINIMAL | {"ollama": {"modell": "qwen"}})


def test_missing_credentials_are_rejected():
    with pytest.raises(ConfigError, match="Incomplete \\[imap\\]"):
        parse_config({"imap": {"host": "imap.example.com"}})


def test_threshold_range_is_checked():
    with pytest.raises(ConfigError, match="threshold"):
        parse_config(MINIMAL | {"ollama": {"threshold": 1.5}})
