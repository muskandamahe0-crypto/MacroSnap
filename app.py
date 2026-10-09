"""MacroSnap: a session-based AI nutrition chat application."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

from prompts import SYSTEM_INSTRUCTION


APP_TITLE = "MacroSnap 🥗"
DEFAULT_MODEL = "gemini-flash-latest"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_CONTEXT_MESSAGES = 20
MEAL_MARKER = re.compile(
    r"(?:^|\n)MEAL_DATA_JSON:\s*(\{.*?\}|null)\s*$", re.DOTALL
)


class ModelConfigurationError(RuntimeError):
    """The configured model is not usable with the current API key."""


def initialize_session() -> None:
    """Create session values once, preserving them across Streamlit reruns."""
    defaults: dict[str, Any] = {
        "user_name": "",
        "messages": [],
        "meals": [],
        "summary_text": "",
        "confirm_clear_meals": False,
        "chat_form_version": 0,
        "notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_key() -> str:
    """Read the key from Streamlit secrets first, then the environment."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        secret_key = ""
    return str(secret_key or os.environ.get("GEMINI_API_KEY", "")).strip()


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    """Reuse a Gemini client for this API key during the app session."""
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    model_name = os.environ.get("GEMINI_MODEL", "").strip()
    if model_name:
        return model_name
    try:
        model_name = str(st.secrets.get("GEMINI_MODEL", "")).strip()
    except (FileNotFoundError, KeyError):
        model_name = ""
    return model_name or DEFAULT_MODEL


@st.cache_data(ttl=300, show_spinner=False)
def list_generate_content_models(api_key: str) -> tuple[str, ...]:
    """Cache the models this key can actually call for content generation."""
    client = genai.Client(api_key=api_key)
    try:
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if "generateContent" in actions:
                name = (getattr(model, "name", "") or "").removeprefix("models/")
                if name:
                    names.append(name)
        return tuple(sorted(set(names)))
    finally:
        client.close()


def verify_model_available(api_key: str, model_name: str) -> None:
    models = list_generate_content_models(api_key)
    if not models:
        raise ModelConfigurationError(
            "No Gemini models supporting generateContent are available to this "
            "API key. Check the key and project access, then run "
            "`python list_gemini_models.py` to diagnose."
        )
    if model_name not in models:
        raise ModelConfigurationError(
            f"Configured Gemini model '{model_name}' is not in this key's "
            "generateContent model list. Run `python list_gemini_models.py` "
            "and set GEMINI_MODEL to a returned model."
        )


def read_uploaded_image(uploaded_file: Any) -> tuple[bytes, str]:
    """Validate image content and return its bytes and supported MIME type."""
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        raise ValueError("The selected image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Please choose an image smaller than 10 MB.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("That file is not a valid JPG, JPEG, or PNG image.") from exc

    mime_type = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image_format or "")
    if mime_type is None:
        raise ValueError("Please upload a JPG, JPEG, or PNG image.")
    return image_bytes, mime_type


def build_contents(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Convert saved chat turns into the Gemini SDK's typed content format."""
    contents: list[types.Content] = []
    for message in messages[-MAX_CONTEXT_MESSAGES:]:
        parts: list[types.Part] = []
        text = message.get("text", "")
        if text:
            parts.append(types.Part.from_text(text=text))
        image_bytes = message.get("image_bytes")
        if image_bytes:
            parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=message["image_mime_type"],
                )
            )
        if parts:
            contents.append(types.Content(role=message["role"], parts=parts))
    return contents


def parse_meal_marker(response_text: str) -> tuple[str, dict[str, Any] | None, bool]:
    """Remove and decode Gemini's private structured meal-data marker."""
    match = MEAL_MARKER.search(response_text)
    if match is None:
        return response_text.strip(), None, False

    visible_text = response_text[: match.start()].strip()
    payload_text = match.group(1)
    if payload_text == "null":
        return visible_text, None, True
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return visible_text, None, False
    if not isinstance(payload, dict):
        return visible_text, None, False
    return visible_text, payload, True


def validate_meal_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only complete, plausible numeric estimates for the meal log."""
    description = payload.get("food_description")
    serving_size = payload.get("serving_size")
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(serving_size, str) or not serving_size.strip():
        return None

    result: dict[str, Any] = {
        "food_description": description.strip(),
        "serving_size": serving_size.strip(),
    }
    limits = {
        "calories": 100_000,
        "protein_g": 10_000,
        "carbohydrates_g": 10_000,
        "fat_g": 10_000,
    }
    for field, maximum in limits.items():
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not 0 <= numeric_value <= maximum:
            return None
        result[field] = numeric_value
    return result


def explain_api_error(error: Exception) -> str:
    """Return useful, secret-safe guidance without showing raw SDK errors."""
    if isinstance(error, ModelConfigurationError):
        return str(error)

    error_name = type(error).__name__.lower()
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    status_text = str(status).lower()

    if status == 429 or "resourceexhausted" in error_name or "quota" in status_text:
        return (
            "Gemini's request limit or quota was reached. Wait a little and retry, "
            "or check the quota for your Google AI project."
        )
    if status in (401, 403) or "permission" in error_name or "unauthorized" in error_name:
        return (
            "Gemini rejected the API key or project permissions. Check "
            "GEMINI_API_KEY and Gemini API access, then run "
            "`python list_gemini_models.py` to verify the key."
        )
    if status == 400:
        return (
            "Gemini rejected the request. Check that GEMINI_API_KEY is valid "
            "and the configured model is available to its project."
        )
    if status == 404 or "notfound" in error_name:
        return (
            f"Gemini returned 404 for model '{get_model_name()}'. Check that "
            "GEMINI_MODEL matches a model listed for this key by "
            "`python list_gemini_models.py`, then restart Streamlit."
        )
    if isinstance(error, ValueError) and "empty response" in str(error).lower():
        return "Gemini returned an empty response. Please try your message again."
    if (
        "timeout" in error_name
        or "connection" in error_name
        or "network" in error_name
        or "unavailable" in error_name
        or status == 503
    ):
        return "A network or Gemini service issue interrupted the request. Check your connection and retry."
    return "Gemini could not complete this request. Check your model and API-key access, then retry."


def request_gemini_response(
    client: genai.Client, history: list[dict[str, Any]]
) -> str:
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_contents(history),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
        ),
    )
    text = response.text
    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text.strip()


def add_user_message(
    text: str, image_bytes: bytes | None, mime_type: str | None
) -> bool:
    """Send a new turn, keeping failed turns available for the user's retry."""
    user_message: dict[str, Any] = {
        "role": "user",
        "text": text.strip(),
        "image_bytes": image_bytes,
        "image_mime_type": mime_type,
    }
    if not user_message["text"] and not image_bytes:
        st.warning("Enter a message or attach a meal photo first.")
        return False

    st.session_state.messages.append(user_message)
    api_key = get_api_key()
    if not api_key:
        st.session_state.messages.pop()
        st.session_state.notice = (
            "error",
            "Gemini API key not found. Add GEMINI_API_KEY to "
            "`.streamlit/secrets.toml` or your environment, then try again.",
        )
        return False

    try:
        model_name = get_model_name()
        verify_model_available(api_key, model_name)
        client = get_gemini_client(api_key)
        with st.spinner("Your nutrition buddy is thinking..."):
            response_text = request_gemini_response(
                client, st.session_state.messages
            )
    except Exception as exc:
        st.session_state.messages.pop()
        st.session_state.notice = (
            "error",
            f"{explain_api_error(exc)} Your message was not sent; the form "
            "contents are preserved so you can retry.",
        )
        return False

    visible_text, payload, marker_valid = parse_meal_marker(response_text)
    assistant_message: dict[str, Any] = {
        "role": "model",
        "text": visible_text or "I couldn't read a response. Please try again.",
    }
    st.session_state.messages.append(assistant_message)

    if payload is not None:
        meal = validate_meal_data(payload)
        if meal is not None:
            meal["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.meals.append(meal)
            st.session_state.summary_text = ""
            st.session_state.notice = ("success", "Meal estimate added to your log.")
        else:
            st.session_state.messages[-1]["text"] += (
                "\n\nI couldn't reliably save all the nutrition numbers. "
                "Please confirm the portion or correct the meal details and try again."
            )
    elif not marker_valid:
        st.session_state.messages[-1]["text"] += (
            "\n\nI couldn't verify structured nutrition estimates, so I haven't "
            "added a meal. If this was a meal, please confirm or correct its details."
        )
    return True


def calculate_totals(meals: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("calories", "protein_g", "carbohydrates_g", "fat_g")
    return {key: sum(float(meal[key]) for meal in meals) for key in keys}


def format_number(value: float) -> str:
    return f"{value:,.1f}".rstrip("0").rstrip(".")


def create_summary(meals: list[dict[str, Any]]) -> str:
    totals = calculate_totals(meals)
    lines = [
        "MacroSnap Nutrition Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Logged meals:",
    ]
    for index, meal in enumerate(meals, start=1):
        lines.append(
            f"{index}. {meal['food_description']} "
            f"({meal['serving_size']}) — {format_number(meal['calories'])} kcal"
        )
        lines.append(
            "   "
            f"Protein {format_number(meal['protein_g'])} g | "
            f"Carbohydrates {format_number(meal['carbohydrates_g'])} g | "
            f"Fat {format_number(meal['fat_g'])} g"
        )
    lines.extend(
        [
            "",
            "Estimated totals:",
            f"Calories: {format_number(totals['calories'])} kcal",
            f"Protein: {format_number(totals['protein_g'])} g",
            f"Carbohydrates: {format_number(totals['carbohydrates_g'])} g",
            f"Fat: {format_number(totals['fat_g'])} g",
            "",
            "Estimates only — not medical advice.",
        ]
    )
    return "\n".join(lines)


def render_sidebar() -> None:
    meals = st.session_state.meals
    totals = calculate_totals(meals)
    with st.sidebar:
        st.markdown("## Your dashboard")
        st.write(f"Hi, **{st.session_state.user_name}**!")
        st.caption("Your meal log is kept for this session.")
        st.metric("Logged meals", len(meals))
        col1, col2 = st.columns(2)
        col1.metric("Calories", f"{format_number(totals['calories'])} kcal")
        col2.metric("Protein", f"{format_number(totals['protein_g'])} g")
        col1.metric("Carbohydrates", f"{format_number(totals['carbohydrates_g'])} g")
        col2.metric("Fat", f"{format_number(totals['fat_g'])} g")

        st.markdown("### Meal history")
        if meals:
            for index, meal in enumerate(meals, start=1):
                st.caption(
                    f"{index}. {meal['food_description']} — "
                    f"{format_number(meal['calories'])} kcal"
                )
        else:
            st.caption("Your confirmed meal estimates will appear here.")

        if not st.session_state.confirm_clear_meals:
            if st.button("Clear meal history", disabled=not meals, use_container_width=True):
                st.session_state.confirm_clear_meals = True
                st.rerun()
        else:
            st.warning("Clear all logged meals?")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Clear", type="primary", use_container_width=True):
                st.session_state.meals.clear()
                st.session_state.summary_text = ""
                st.session_state.confirm_clear_meals = False
                st.session_state.notice = ("success", "Meal history cleared.")
                st.rerun()
            if cancel_col.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear_meals = False
                st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = "assistant" if message["role"] == "model" else "user"
        with st.chat_message(role):
            if message.get("text"):
                st.markdown(message["text"])
            if message.get("image_bytes"):
                st.image(message["image_bytes"], caption="Uploaded meal photo", use_container_width=True)


def render_summary() -> None:
    st.markdown("## Daily nutrition summary")
    if st.button("Generate My Nutrition Summary", type="primary"):
        if st.session_state.meals:
            st.session_state.summary_text = create_summary(st.session_state.meals)
        else:
            st.session_state.summary_text = ""
            st.info("Log a meal first, then generate your summary.")

    summary = st.session_state.summary_text
    if not summary:
        return

    st.text_area(
        "Your summary (available to copy manually if clipboard access is blocked)",
        value=summary,
        height=250,
        key=f"summary_copy_area_{hashlib.sha256(summary.encode('utf-8')).hexdigest()[:12]}",
    )
    copy_col, download_col = st.columns(2)
    escaped_summary = json.dumps(summary).replace("<", "\\u003c")
    with copy_col:
        st.html(
            f"""
            <button id="macro-copy-button" style="width:100%;padding:0.5rem;
              border:1px solid #21864b;border-radius:0.5rem;background:#21864b;
              color:white;font-size:1rem;cursor:pointer">Copy Summary</button>
            <div style="font:14px sans-serif;color:#216e39;padding-top:0.35rem"
              id="copy-status" aria-live="polite"></div>
            <textarea id="macro-summary" style="position:fixed;left:-9999px"></textarea>
            <script>
              const text = {escaped_summary};
              const button = document.getElementById("macro-copy-button");
              const status = document.getElementById("copy-status");
              button.addEventListener("click", async () => {{
                try {{
                  if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(text);
                  }} else {{
                    const field = document.getElementById("macro-summary");
                    field.value = text;
                    field.select();
                    if (!document.execCommand("copy")) throw new Error("copy unavailable");
                  }}
                  status.textContent = "Summary copied to clipboard.";
                }} catch (_) {{
                  status.textContent = "Clipboard unavailable — copy the visible summary above.";
                  status.style.color = "#9a3412";
                }}
              }});
            </script>
            """,
            unsafe_allow_javascript=True,
        )
    with download_col:
        st.download_button(
            "Download TXT Report",
            data=summary,
            file_name="macrosnap-nutrition-summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_onboarding() -> None:
    st.title(APP_TITLE)
    st.subheader("Your personal AI nutrition buddy")
    st.write("Get friendly, practical food estimates and keep track of your meals.")
    with st.form("onboarding_form"):
        name = st.text_input("What should I call you?", max_chars=60)
        started = st.form_submit_button("Get Started", type="primary")
    if started:
        cleaned_name = name.strip()
        if not cleaned_name:
            st.warning("Please enter your name to get started.")
        else:
            st.session_state.user_name = cleaned_name
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="MacroSnap",
        page_icon="🥗",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root { --green: #21864b; --pale-green: #edf7ef; }
        .stApp { background-color: #fbfdfb; }
        h1, h2, h3 { color: #176338; }
        [data-testid="stSidebar"] { background: #f1f8f2; }
        [data-testid="stMetric"] {
          background: white; border: 1px solid #dcecdf; padding: 0.65rem;
          border-radius: 0.75rem;
        }
        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button[kind="primary"] {
          background-color: #21864b; border-color: #21864b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    initialize_session()
    if not st.session_state.user_name:
        render_onboarding()
        return

    render_sidebar()
    st.title(APP_TITLE)
    st.caption(f"Your personal AI nutrition buddy · Welcome, {st.session_state.user_name}!")
    if st.session_state.notice:
        notice_type, notice_text = st.session_state.notice
        st.session_state.notice = None
        if notice_type == "success":
            st.success(notice_text)
        else:
            st.error(notice_text)
    api_key = get_api_key()
    if not api_key:
        st.warning(
            "Gemini API key not configured. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` or set it in your environment to start chatting."
        )
    render_chat_history()

    form_version = st.session_state.chat_form_version
    with st.form(f"chat_form_{form_version}"):
        user_text = st.text_area(
            "Message your nutrition buddy",
            placeholder="Try: I ate 2 boiled eggs and 2 slices of bread.",
            height=90,
            key=f"chat_text_{form_version}",
        )
        uploaded_file = st.file_uploader(
            "Add a meal photo (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key=f"meal_photo_{form_version}",
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        image_bytes: bytes | None = None
        mime_type: str | None = None
        if uploaded_file is not None:
            try:
                image_bytes, mime_type = read_uploaded_image(uploaded_file)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
        if not user_text.strip() and image_bytes is None:
            st.warning("Enter a message or attach a meal photo first.")
        else:
            succeeded = add_user_message(user_text, image_bytes, mime_type)
            if succeeded:
                st.session_state.chat_form_version += 1
            st.rerun()

    st.divider()
    render_summary()


if __name__ == "__main__":
    main()
