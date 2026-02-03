# 🤖 AI Powered Voice Bot

An AI-powered voice assistant that uses local LLM (via LM Studio) for intelligent responses.

## 🔄 Pipeline Flow

```
🎤 Audio Input
      ↓
📝 Speech-to-Text (STT)
      ↓
🔍 Voice Activity Detection (VAD) - Silero
      ↓
🧠 Local LLM (LM Studio)
      ↓
🔊 Text-to-Speech (TTS)
      ↓
🔈 Audio Output
```

## 📦 Tech Stack

| Component          | Technology              |
| ------------------ | ----------------------- |
| Pipeline Framework | Pipecat                 |
| VAD                | Silero VAD              |
| STT                | Whisper / Deepgram      |
| LLM                | Local LLM via LM Studio |
| TTS                | Edge TTS (free)         |
| Real-time Audio    | FastRTC / WebRTC        |

## 🚀 Setup Instructions

### 1. Install LM Studio

1. Download LM Studio from: https://lmstudio.ai/
2. Open LM Studio and download a model (recommended: Llama 3, Mistral, or Phi-3)
3. Load the model and start the local server
4. Server will run on `http://localhost:1234`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pipecat-ai[silero] aiohttp python-dotenv sounddevice numpy
```

### 3. Run the Voice Bot

**Text Demo Mode (for testing):**

```bash
python voice_agent.py
```

**Full Voice Mode (with microphone):**

```bash
python voice_agent.py --full
```

## 📁 Project Structure

```
Project/
├── voice_agent.py      # Main voice bot code
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Configuration

Edit these values in `voice_agent.py`:

```python
agent = VoiceAgent(
    lm_studio_url="http://localhost:1234/v1",  # LM Studio URL
    model_name="local-model",                   # Model name
    system_prompt="Your custom prompt here"     # Bot personality
)
```

## 🎯 Features

- ✅ Local LLM integration (no cloud API needed)
- ✅ Voice Activity Detection (VAD)
- ✅ Speech-to-Text (STT)
- ✅ Text-to-Speech (TTS)
- ✅ Real-time conversation
- ✅ Conversation history maintained

## 📝 How It Works

1. **Audio Input**: Captures audio from microphone
2. **VAD (Silero)**: Detects when user starts/stops speaking
3. **STT**: Converts speech to text
4. **LLM**: Sends text to local LLM (LM Studio) and gets response
5. **TTS**: Converts response text to speech
6. **Audio Output**: Plays the response through speakers

## 🐛 Troubleshooting

**"Cannot connect to LM Studio"**

- Make sure LM Studio is running
- Check if a model is loaded
- Verify the server is started (default: localhost:1234)

**"No audio devices found"**

- Check microphone permissions
- Install audio drivers
- Try running with `--full` flag for voice mode

## 👨‍💻 Author

Created for internship project.
