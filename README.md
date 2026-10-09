# MacroSnap 🥗

MacroSnap is a Streamlit nutrition chat app powered by Google's Gemini API. Ask
about a meal in text or upload a JPG, JPEG, or PNG photo. The app estimates
calories, protein, carbohydrates, and fat, keeps a session-based meal log, and
creates a summary you can copy or download. It does not use WhatsApp or Twilio.

Nutrition values are estimates, not medical advice. Photo estimates are
especially approximate.

## Prerequisites

- Python 3.10 or newer
- A Gemini API key with access to a supported text-and-image model
- Internet access while chatting with Gemini

The default model is `gemini-flash-latest`. MacroSnap checks the selected
model against the `generateContent` models available to the configured key.
Set `GEMINI_MODEL` in Streamlit secrets or the environment to select another
model returned by the diagnostic script.

## Install

Open PowerShell in this `macrosnap` folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow it for this terminal
only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configure your Gemini API key

Choose either method below. Do not commit your real key.

**Streamlit secrets:** Copy the example and edit the new file:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

Replace the placeholder with your key:

```toml
GEMINI_API_KEY = "your-real-key"
GEMINI_MODEL = "gemini-flash-latest"
```

**Environment variable:** Set the key for the current PowerShell window:

```powershell
$env:GEMINI_API_KEY = "your-real-key"
```

Optionally choose a model in the current PowerShell window:

```powershell
$env:GEMINI_MODEL = "gemini-flash-latest"
```

`GEMINI_MODEL` in the environment takes precedence over the value in
`secrets.toml`.

To list models this key can use for `generateContent` (without displaying the
key), run this from the `macrosnap` folder:

```powershell
python .\list_gemini_models.py
```

The app checks the configured model against this list before sending a chat
request. If the key is invalid, has no compatible models, or the configured
model is not listed, MacroSnap shows a targeted diagnostic message.

## Run

From the `macrosnap` folder, with the virtual environment activated:

```powershell
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Keep the
terminal open while using the app.

## Try MacroSnap

1. Enter your name and select **Get Started**.
2. Type `I ate 2 boiled eggs` and select **Send**. Review the estimated
   serving and macros; estimates can vary.
3. For a photo test, send a JPG, JPEG, or PNG meal image, with or without text.
   The photo preview appears in the chat. Only foods the model can reasonably
   identify should be estimated.
4. Ask a follow-up such as `How much protein was in that?` to check that the
   chat retains context. Follow-up questions are not logged as new meals.
5. Review the sidebar totals, select **Generate My Nutrition Summary**, then
   use **Copy Summary** or **Download TXT Report**. If browser clipboard access
   is blocked, copy from the visible summary text area.
6. **Clear meal history** asks for confirmation before removing the log.

## Troubleshooting

- **API key not configured:** Add `GEMINI_API_KEY` to
  `.streamlit/secrets.toml` or set the environment variable, then restart the
  Streamlit app.
- **Unauthorized / permission error:** Check that the key is valid and that its
  Google AI project has access to Gemini API models.
- **Model not found or unavailable:** Set `GEMINI_MODEL` to a supported model
  available to your key using the `generateContent` model list from
  `python .\list_gemini_models.py`, then restart Streamlit.
- **Quota or rate limit:** Wait before retrying and check the quota associated
  with your Google AI project.
- **Network or service error:** Check your internet connection and retry.
- **Image rejected:** Upload a valid JPG, JPEG, or PNG smaller than 10 MB.
- **Clipboard unavailable:** Use the summary text area to select and copy the
  report manually.
- **Missing Python packages:** Activate the virtual environment and run
  `python -m pip install -r requirements.txt` again.

The app does not fabricate a response if Gemini fails. Failed requests remain
out of the chat history and meal totals so you can retry.
