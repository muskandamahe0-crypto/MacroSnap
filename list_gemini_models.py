"""List Gemini models available to the configured key without printing it."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import toml
from google import genai


PROJECT_DIR = Path(__file__).resolve().parent
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"


def configured_api_key() -> str:
    """Load the same secret sources as the app without displaying their values."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if SECRETS_PATH.exists():
        try:
            secrets: dict[str, Any] = toml.loads(
                SECRETS_PATH.read_text(encoding="utf-8")
            )
            return str(secrets.get("GEMINI_API_KEY", "")).strip()
        except (OSError, ValueError) as exc:
            raise RuntimeError("Could not read the Streamlit secrets file.") from exc
    return ""


def model_id(model: Any) -> str:
    name = getattr(model, "name", "") or ""
    return name.removeprefix("models/")


def main() -> int:
    api_key = configured_api_key()
    if not api_key:
        print(
            "No GEMINI_API_KEY was found in Streamlit secrets or the environment."
        )
        return 1

    try:
        client = genai.Client(api_key=api_key)
        compatible = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if "generateContent" in actions:
                compatible.append(model_id(model))
    except Exception as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        exception_type = type(exc).__name__.lower()
        if status in (401, 403) or "permission" in exception_type or "unauthorized" in exception_type:
            print(
                "Gemini rejected the configured API key or its project permissions. "
                "Check the key and Gemini API access; the key was not displayed."
            )
        elif "connection" in exception_type or "timeout" in exception_type:
            print(
                "Could not reach the Gemini API. Check the network and try again; "
                "the key was not displayed."
            )
        else:
            print(
                "Could not list Gemini models "
                f"({type(exc).__name__}). Check key, API access, and network; "
                "the key was not displayed."
            )
        return 1
    finally:
        if "client" in locals():
            client.close()

    if not compatible:
        print("The configured key has no models supporting generateContent.")
        return 2

    print("Models available to this key with generateContent support:")
    for name in compatible:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
