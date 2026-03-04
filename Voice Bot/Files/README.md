# 🤖 AI Powered Voice Bot

An AI-powered voice assistant that uses local LLM (via LM Studio) for intelligent responses, built with Pipecat pipeline architecture.

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

| Component          | Technology                |
| ------------------ | ------------------------- |
| Pipeline Framework | Pipecat                   |
| VAD                | Silero VAD                |
| STT                | Google Speech Recognition |
| LLM                | Local LLM via LM Studio   |
| TTS                | pyttsx3 / Edge TTS        |
| Web Interface      | WebSocket + aiohttp       |
| Real-time Audio    | sounddevice               |

## 🚀 Quick Start

### 1. Install LM Studio

1. Download LM Studio from: https://lmstudio.ai/
2. Open LM Studio and download a model (recommended: Llama 3, Mistral, or Phi-3)
3. Load the model and start the local server
4. Server will run on `http://localhost:1234`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Voice Bot

```bash
# Classic mode (simple)
python voice_bot.py

# Pipecat mode (pipeline architecture)
python voice_bot.py --pipecat

# Pipecat with streaming responses
python voice_bot.py --streaming

# Advanced Pipecat with more features
python pipecat_voice_bot.py

# Web interface (browser-based)
python pipecat_web.py
```

## 📁 Project Structure

```
Voice Bot/
├── voice_bot.py           # Main voice bot (classic + pipecat modes)
├── pipecat_voice_bot.py   # Pipecat-focused implementation
├── pipecat_advanced.py    # Advanced features (VAD, function calling)
├── pipecat_web.py         # Web interface with WebSocket server
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🎯 Features by File

### voice_bot.py

- ✅ Classic mode (speech_recognition + pyttsx3)
- ✅ Pipecat mode (pipeline architecture)
- ✅ Streaming responses
- ✅ Command line arguments

### pipecat_voice_bot.py

- ✅ Full Pipecat pipeline
- ✅ Streaming LLM responses
- ✅ Interruption handling
- ✅ Conversation commands (clear, help)
- ✅ Conversation logging
- ✅ WebSocket support (optional)

### pipecat_advanced.py

- ✅ Silero VAD integration
- ✅ Function/tool calling
- ✅ Real-time audio input/output
- ✅ Audio buffer management
- ✅ Example functions (time, weather, calculator)

### pipecat_web.py

- ✅ WebSocket server
- ✅ REST API endpoint
- ✅ Beautiful web interface
- ✅ Real-time streaming to browser
- ✅ Multiple client support
- ✅ Session management

## 🔧 Configuration

Edit the configuration in any file:

```python
# voice_bot.py
bot = ClassicVoiceBot(base_url="http://localhost:1234/v1")

# pipecat_voice_bot.py
class BotConfig:
    LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
    MODEL_NAME = "local-model"
    SYSTEM_PROMPT = "Your custom prompt here"
    MAX_TOKENS = 150
```

## 📝 How It Works

1. **Audio Input**: Captures audio from microphone
2. **VAD (Silero)**: Detects when user starts/stops speaking
3. **STT**: Converts speech to text using Google Speech Recognition
4. **LLM**: Sends text to local LLM (LM Studio) and gets response
5. **TTS**: Converts response text to speech using pyttsx3
6. **Audio Output**: Plays the response through speakers

## 🌐 Web Interface

Start the web server:

```bash
python pipecat_web.py --port 8765
```

Then open http://localhost:8765 in your browser.

API Endpoints:

- `GET /` - Web interface
- `GET /ws` - WebSocket connection
- `POST /api/chat` - REST chat endpoint
- `GET /api/health` - Health check
- `GET /api/models` - List available models

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
