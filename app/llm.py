import time

from google import genai
from google.genai import types

from app.config import LLM_API_KEY

# Chosen after timing tests: much faster than gemini-3.8-flash on the free tier.
MODEL_NAME = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3

_client = None


class LLMError(Exception):
    """An LLM problem, with a message that is safe to show to the user."""


def _get_client():
    """Create the Gemini client once and reuse it."""
    global _client
    if _client is None:
        if not LLM_API_KEY or LLM_API_KEY.startswith("your_"):
            raise LLMError("LLM_API_KEY is missing. Add your Gemini key to the .env file.")
        _client = genai.Client(api_key=LLM_API_KEY)
    return _client


def _friendly_message(code):
    """Turn an HTTP status code into a message for the user."""
    if code is None:
        return "Could not reach the Gemini API. Check your internet connection."
    if code == 400:
        return "Gemini rejected the request (400). This is usually an invalid LLM_API_KEY."
    if code in (401, 403):
        return "Gemini refused access (" + str(code) + "). Check LLM_API_KEY in your .env file."
    if code == 404:
        return "Model '" + MODEL_NAME + "' was not found. Pick another name in app/llm.py."
    if code == 429:
        return (
            "Gemini rate limit or daily quota reached. Wait a minute and try again, "
            "or switch MODEL_NAME to a lighter model."
        )
    if code in (500, 503):
        return "Gemini is temporarily unavailable. Try again shortly."
    return "Gemini returned an error (" + str(code) + ")."


def generate(prompt, system_instruction=None):
    """Send a prompt to the model and return its text answer.

    system_instruction: optional standing rules for the model.
    Raises LLMError with a user-safe message on failure.
    """
    client = _get_client()

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    response = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            break
        except Exception as error:
            code = getattr(error, "code", None)
            temporary = code in (429, 500, 503)
            if temporary and attempt < MAX_ATTEMPTS:
                time.sleep(4 * attempt)
                continue
            raise LLMError(_friendly_message(code)) from error

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise LLMError("The model returned no text (the response may have been blocked). Try rephrasing.")
    return text.strip()


if __name__ == "__main__":
    print("Model:", MODEL_NAME)

    try:
        started = time.time()
        answer = generate("In one short sentence, what is a vector database?")
        print("Answer 1:", answer.encode("ascii", "replace").decode())
        print("Took", round(time.time() - started, 1), "seconds")

        answer = generate(
            "What is the capital of France?",
            system_instruction="Answer with exactly one word and nothing else.",
        )
        print("Answer 2:", answer.encode("ascii", "replace").decode())

    except LLMError as error:
        print("ERROR:", error)