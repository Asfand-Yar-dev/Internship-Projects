import os
import sys
import requests
import pyttsx3
import speech_recognition as sr

def configure_silence_detection(recognizer: sr.Recognizer):
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5

# Audio Listening
def listen_until_silence(recognizer: sr.Recognizer, microphone: sr.Microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("Listening... (speak now)")
        audio = recognizer.listen(source)  # Returns when silence is detected
    print("Silence detected. Stopped listening.")
    return audio



# Speech-to-Text (STT)
def speech_to_text(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        return text.strip()
    except sr.UnknownValueError:
        print("Didn't catch that (speech unintelligible).")
        return ""
    except sr.RequestError as e:
        print(f"STT service error: {e}")
        return ""

# System prompt for high-quality responses
SYSTEM_PROMPT = """You are a highly capable, thoughtful, and helpful AI assistant. Follow these guidelines:
- Provide accurate, well-structured, and concise responses.
- When explaining complex topics, break them down into clear steps.
- If you're unsure about something, say so honestly rather than guessing.
- Adapt your tone to the context: professional for technical questions, friendly for casual conversation.
- Give direct answers first, then elaborate if needed.
- Use examples when they help clarify a concept.
- Keep responses focused and avoid unnecessary filler."""

# Conversation history manager
class ConversationHistory:
    def __init__(self, system_prompt: str, max_turns: int = 20):
        # Store system prompt to inject into first user message (some models don't support "system" role)
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.max_turns = max_turns  # max user+assistant pairs to retain

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        # Inject system prompt into the first user message to avoid "system" role issues
        if not self.history:
            return []
        messages = []
        for i, msg in enumerate(self.history):
            if i == 0 and msg["role"] == "user":
                # Prepend system prompt to first user message
                messages.append({
                    "role": "user",
                    "content": f"[System Instructions: {self.system_prompt}]\n\nUser: {msg['content']}"
                })
            else:
                messages.append(msg)
        return messages

    def _trim(self):
        """Keep only the last max_turns pairs to avoid exceeding context limits."""
        max_messages = self.max_turns * 2  # each turn = user + assistant
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear(self):
        self.history.clear()


# LLM API (LM Studio local server)
def call_llm(user_text: str, api_url: str, model_name: str, conversation: ConversationHistory) -> str:
    conversation.add_user_message(user_text)

    payload = {
        "model": model_name,
        "messages": conversation.get_messages(),
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    try:
        print("Thinking...")
        resp = requests.post(api_url, json=payload, timeout=120)
        if resp.status_code >= 400:
            print(f"LLM API HTTP {resp.status_code}: {resp.text}")
            resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            reply = (data["choices"][0]["message"]["content"] or "").strip()
            conversation.add_assistant_message(reply)
            return reply
        return "I couldn't generate a response."
    except requests.RequestException as e:
        print(f"LLM API error: {e}")
        # Remove the user message that failed so history stays clean
        if conversation.history and conversation.history[-1]["role"] == "user":
            conversation.history.pop()
        return "I ran into a problem contacting the local LLM."


# Text-to-Speech (TTS)
def speak_text(engine: pyttsx3.Engine, text: str):
    if not text:
        return
    print("Speaking...")
    engine.say(text)
    engine.runAndWait()
    print("Finished speaking.")


# Main Loop
def main():
    # Read LM Studio settings from environment or use defaults
    api_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
    model_name = os.environ.get("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")

    # Basic checks
    print(f"LM Studio endpoint: {api_url}")
    if "Your-Model-Name-Here" in model_name or not model_name.strip():
        print("Error: LMSTUDIO_MODEL is not set to a valid model name.")
        print("Open LM Studio, start the OpenAI Compatible Server, and copy the exact model name shown.")
        print('Then set it for this session: set LMSTUDIO_MODEL="Exact-Model-Name-From-LM-Studio"')
        sys.exit(1)

    # Initialize components
    recognizer = sr.Recognizer()
    configure_silence_detection(recognizer)

    try:
        microphone = sr.Microphone()  # Use default input device
    except OSError as e:
        print(f"Microphone error: {e}")
        print("Ensure a working microphone is connected and PyAudio is installed.")
        sys.exit(1)

    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 180)
    tts_engine.setProperty("volume", 1.0)

    # Initialize conversation history with system prompt
    conversation = ConversationHistory(SYSTEM_PROMPT, max_turns=20)

    print("Voice assistant ready. Speak after the prompt.")
    print("While processing or speaking, the mic is idle. After speaking, listening resumes.")
    print("Say 'clear history' or 'reset conversation' to start fresh.\n")

    # Natural back-and-forth loop
    try:
        while True:
            # 1) Listen until the user stops speaking
            audio = listen_until_silence(recognizer, microphone)

            # 2) Transcribe speech to text
            user_text = speech_to_text(recognizer, audio)
            if not user_text:
                print("No text recognized. Resuming listening...\n")
                continue

            print(f"You said: {user_text}")

            # Check for reset commands
            if user_text.lower() in ("clear history", "reset conversation", "start over", "new conversation"):
                conversation.clear()
                print("Conversation history cleared.\n")
                speak_text(tts_engine, "Conversation history cleared. Let's start fresh.")
                continue

            # 3) Call local LLM and wait for full response (with history)
            assistant_text = call_llm(user_text, api_url, model_name, conversation)
            print(f"Assistant: {assistant_text}")

            # 4) Speak the response without interruption
            speak_text(tts_engine, assistant_text)

            # 5) Loop resumes listening automatically
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")


if __name__ == "__main__":
    main()