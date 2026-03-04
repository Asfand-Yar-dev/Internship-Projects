"""
FastRTC Voice Agent (LM Studio – Mistral Compatible)
===================================================
Speech -> Text -> LM Studio (local Mistral) -> Text -> Speech
"""

import requests
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model

# ---------------- CONFIG ---------------- #

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

SYSTEM_PROMPT = """You are a voice-based assistant with strict turn-taking control.

When you start speaking a response, you must continue speaking until the response is fully finished.

While you are speaking:
- Ignore all incoming audio
- Ignore background noise, partial speech, or accidental sounds
- Do not pause, stop, or interrupt your response for any reason
- Background noise or faint speech must never cause you to stop speaking.
- You may only stop speaking after the response is completely finished.

After finishing your response:
- Switch to listening mode
- Listen for the user's voice
- Do nothing until the user fully stops speaking

A user turn is considered complete only when the user clearly pauses or stops speaking.

After the user turn completes:
- Process the full input
- Generate one complete response
- Speak the response fully

Never overlap listening and speaking.
Never respond mid-speech.
Always complete one full response before listening again.
Keep each response concise and complete in one turn."""

MAX_TOKENS = 150
TEMPERATURE = 0.7

# ---------------- STATE ---------------- #

chat_history = []
LM_MODEL = "local-model"

stt_model = get_stt_model()
tts_model = get_tts_model()

# ---------------- LM STUDIO ---------------- #

def get_lm_model_name() -> str:
    global LM_MODEL
    try:
        r = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            LM_MODEL = models[0]["id"]
    except Exception:
        pass
    return LM_MODEL


def call_lm_studio(user_text: str) -> str:
    global chat_history

    # FIRST TURN → embed system prompt
    if not chat_history:
        content = f"""[INST]
{SYSTEM_PROMPT}

User: {user_text}
[/INST]"""
        messages = [{"role": "user", "content": content}]
    else:
        messages = chat_history + [{"role": "user", "content": user_text}]

    try:
        r = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": LM_MODEL,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=60,
        )
        r.raise_for_status()

        reply = r.json()["choices"][0]["message"]["content"].strip()

        # save history
        if not chat_history:
            chat_history.append({"role": "user", "content": content})
        else:
            chat_history.append({"role": "user", "content": user_text})

        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        print(f"[LM Studio Error] {e}")
        return "Sorry, I couldn't respond."


# ---------------- FASTRTC ---------------- #

def voice_handler(audio):
    try:
        text = stt_model.stt(audio)
        if not text or not text.strip():
            return

        text = text.strip()
        print(f"You: {text}")

        reply = call_lm_studio(text)
        print(f"Bot: {reply}")

        for chunk in tts_model.stream_tts_sync(reply):
            yield chunk

    except Exception as e:
        print(f"[FastRTC Error] {e}")


# ---------------- MAIN ---------------- #

def main():
    try:
        r = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        r.raise_for_status()
        model_name = get_lm_model_name()
        print(f"LM Studio connected — model: {model_name}")
    except Exception:
        print("❌ LM Studio not running on port 1234")
        return

    stream = Stream(
        ReplyOnPause(voice_handler),
        modality="audio",
        mode="send-receive",
    )

    print("\nOpen browser at the local URL shown below:\n")
    stream.ui.launch()


if __name__ == "__main__":
    main()
