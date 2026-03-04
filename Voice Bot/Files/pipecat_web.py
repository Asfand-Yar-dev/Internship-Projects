import asyncio
import aiohttp
from aiohttp import web
import json
import uuid
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
import base64
import threading

# Server-side TTS
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    TTS_ENGINE.setProperty('rate', 180)
    TTS_AVAILABLE = True
    print("✅ Server-side TTS enabled (pyttsx3)")
except Exception as e:
    TTS_ENGINE = None
    TTS_AVAILABLE = False
    print(f"⚠️ Server-side TTS not available: {e}")

def speak_text(text: str):
    """Speak text using server-side TTS in a separate thread"""
    if TTS_AVAILABLE and TTS_ENGINE and text:
        def _speak():
            try:
                TTS_ENGINE.say(text)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    AudioRawFrame,
    StartFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class WebConfig:
    """Web server configuration"""
    host: str = "localhost"
    port: int = 8765
    lm_studio_url: str = "http://127.0.0.1:1234/v1"
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief."
    max_tokens: int = 150
    enable_cors: bool = True


# ============================================================================
# CLIENT SESSION
# ============================================================================
@dataclass
class ClientSession:
    """Represents a connected client"""
    id: str
    websocket: Optional[web.WebSocketResponse] = None
    conversation: list = field(default_factory=list)
    created_at: float = 0.0


# ============================================================================
# WEB LLM PROCESSOR
# ============================================================================
class WebLLMProcessor(FrameProcessor):
    """LLM Processor that sends responses to web clients"""
    
    def __init__(self, config: WebConfig, send_callback, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.send_callback = send_callback
        self.sessions: Dict[str, ClientSession] = {}
    
    def get_or_create_session(self, session_id: str) -> ClientSession:
        """Get or create a client session"""
        if session_id not in self.sessions:
            import time
            self.sessions[session_id] = ClientSession(
                id=session_id,
                created_at=time.time()
            )
        return self.sessions[session_id]
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            session_id = getattr(frame, 'user_id', 'default')
            session = self.get_or_create_session(session_id)
            
            user_text = frame.text
            print(f"[{session_id}] 🎤 User: {user_text}")
            
            # Get LLM response with streaming
            response = await self._get_streaming_response(session, user_text)
            
            # Send complete response
            await self.send_callback(session_id, {
                "type": "response_complete",
                "text": response
            })
            
            # Server-side TTS - speak the response
            print(f"[{session_id}] 🔊 Bot: {response}")
            speak_text(response)
            
            await self.push_frame(TextFrame(text=response))
        else:
            await self.push_frame(frame, direction)
    
    async def _get_streaming_response(self, session: ClientSession, user_text: str) -> str:
        """Get streaming response from LLM"""
        messages = self._build_messages(session, user_text)
        
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    f"{self.config.lm_studio_url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "stream": True
                    }
                ) as resp:
                    if resp.status == 200:
                        full_response = ""
                        
                        async for line in resp.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    chunk = data["choices"][0].get("delta", {}).get("content", "")
                                    if chunk:
                                        full_response += chunk
                                        # Send chunk to client
                                        await self.send_callback(session.id, {
                                            "type": "response_chunk",
                                            "text": chunk
                                        })
                                except:
                                    pass
                        
                        # Save to conversation
                        session.conversation.append({"role": "user", "content": user_text})
                        session.conversation.append({"role": "assistant", "content": full_response})
                        
                        return full_response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    def _build_messages(self, session: ClientSession, user_text: str) -> list:
        """Build messages for API - embed system prompt in first user message for compatibility"""
        if len(session.conversation) == 0:
            # First message: include system prompt in user message (Mistral compatibility)
            return [
                {"role": "user", "content": f"[Instructions: {self.config.system_prompt}]\n\nUser: {user_text}"}
            ]
        return session.conversation + [{"role": "user", "content": user_text}]


# ============================================================================
# WEB SERVER
# ============================================================================
class PipecatWebServer:
    """WebSocket and HTTP server for voice bot"""
    
    def __init__(self, config: WebConfig = None):
        self.config = config or WebConfig()
        self.app = web.Application()
        self.websockets: Dict[str, web.WebSocketResponse] = {}
        
        # Create LLM processor
        self.llm = WebLLMProcessor(self.config, self._send_to_client)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP and WebSocket routes"""
        self.app.router.add_get('/', self._handle_index)
        self.app.router.add_get('/ws', self._handle_websocket)
        self.app.router.add_post('/api/chat', self._handle_chat)
        self.app.router.add_get('/api/health', self._handle_health)
        self.app.router.add_get('/api/models', self._handle_models)
        
        # Add CORS middleware if enabled
        if self.config.enable_cors:
            try:
                import aiohttp_cors
                cors = aiohttp_cors.setup(self.app, defaults={
                    "*": aiohttp_cors.ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*"
                    )
                })
                for route in list(self.app.router.routes()):
                    cors.add(route)
            except ImportError:
                print("⚠️ aiohttp-cors not installed, CORS disabled")
    
    async def _send_to_client(self, session_id: str, message: dict):
        """Send message to WebSocket client"""
        if session_id in self.websockets:
            ws = self.websockets[session_id]
            if not ws.closed:
                await ws.send_json(message)
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the web interface"""
        html = self._get_html_interface()
        return web.Response(text=html, content_type='text/html')
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        self.websockets[session_id] = ws
        
        print(f"🌐 Client connected: {session_id}")
        
        # Send welcome message
        await ws.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to Pipecat Voice Bot"
        })
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_message(session_id, data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error: {ws.exception()}")
        finally:
            del self.websockets[session_id]
            print(f"🌐 Client disconnected: {session_id}")
        
        return ws
    
    async def _handle_ws_message(self, session_id: str, data: dict):
        """Handle incoming WebSocket message"""
        msg_type = data.get("type", "")
        
        if msg_type == "text":
            # Text message from client
            text = data.get("text", "")
            if text:
                # Send to LLM processor
                frame = TranscriptionFrame(
                    text=text,
                    user_id=session_id,
                    timestamp=""
                )
                await self.llm.process_frame(frame, FrameDirection.DOWNSTREAM)
        
        elif msg_type == "audio":
            # Audio data from client (base64 encoded)
            audio_b64 = data.get("audio", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                # TODO: Process audio through STT
                await self._send_to_client(session_id, {
                    "type": "info",
                    "message": "Audio received (STT not implemented in web mode)"
                })
        
        elif msg_type == "clear":
            # Clear conversation history
            session = self.llm.get_or_create_session(session_id)
            session.conversation = []
            await self._send_to_client(session_id, {
                "type": "info",
                "message": "Conversation cleared"
            })
    
    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle HTTP chat requests"""
        try:
            data = await request.json()
            text = data.get("text", "")
            session_id = data.get("session_id", "http_default")
            
            if not text:
                return web.json_response({"error": "No text provided"}, status=400)
            
            session = self.llm.get_or_create_session(session_id)
            response = await self.llm._get_streaming_response(session, text)
            
            return web.json_response({
                "response": response,
                "session_id": session_id
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        # Check LM Studio connection
        llm_status = "unknown"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.lm_studio_url}/models",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    llm_status = "connected" if resp.status == 200 else "error"
        except:
            llm_status = "disconnected"
        
        return web.json_response({
            "status": "ok",
            "llm_status": llm_status,
            "websocket_clients": len(self.websockets)
        })
    
    async def _handle_models(self, request: web.Request) -> web.Response:
        """Get available models from LM Studio"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.lm_studio_url}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return web.json_response(data)
                    return web.json_response({"error": "Failed to get models"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    def _get_html_interface(self) -> str:
        """Return the HTML interface"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Pipecat Voice Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 600px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2em; }
        .status {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        .status-item { display: flex; align-items: center; gap: 8px; }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
        }
        .status-dot.connected { background: #44ff44; }
        .chat-container {
            height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: linear-gradient(45deg, #7b2ff7, #f107a3);
            margin-left: auto;
            text-align: right;
        }
        .message.bot {
            background: rgba(255,255,255,0.15);
        }
        .voice-controls {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .mic-button {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(45deg, #7b2ff7, #f107a3);
            color: white;
            font-size: 2em;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(123, 47, 247, 0.4);
        }
        .mic-button:hover { transform: scale(1.1); }
        .mic-button:disabled { opacity: 0.5; cursor: not-allowed; }
        .mic-button.listening {
            animation: pulse 1.5s infinite;
            background: linear-gradient(45deg, #ff4444, #ff6666);
        }
        .mic-button.speaking {
            background: linear-gradient(45deg, #44ff44, #66ff66);
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
            50% { box-shadow: 0 0 0 20px rgba(255, 68, 68, 0); }
        }
        .voice-status { font-size: 0.9em; color: rgba(255,255,255,0.7); }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1em;
        }
        input[type="text"]::placeholder { color: rgba(255,255,255,0.5); }
        input[type="text"]:focus { outline: none; background: rgba(255,255,255,0.15); }
        button.send-btn {
            padding: 15px 25px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(45deg, #7b2ff7, #00d4ff);
            color: #fff;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button.send-btn:hover { transform: scale(1.05); }
        button.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .settings {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .settings label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
            color: rgba(255,255,255,0.7);
            cursor: pointer;
        }
        .divider {
            text-align: center;
            margin: 20px 0;
            color: rgba(255,255,255,0.4);
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Bot</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </div>

        <div class="chat-container" id="chatContainer"></div>

        <div class="voice-controls">
            <button class="mic-button" id="micBtn" disabled>🎤</button>
            <div class="voice-status" id="voiceStatus">Click mic to speak</div>
        </div>

        <div class="settings">
            <label><input type="checkbox" id="autoSpeak" checked> 🔊 Auto-speak</label>
            <label><input type="checkbox" id="continuousMode"> 🔄 Continuous</label>
        </div>

        <div class="divider">— or type —</div>

        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type a message..." disabled>
            <button class="send-btn" id="sendBtn" disabled>Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const voiceStatus = document.getElementById('voiceStatus');
        const autoSpeakCheckbox = document.getElementById('autoSpeak');
        const continuousModeCheckbox = document.getElementById('continuousMode');

        let ws = null;
        let isListening = false;
        let isSpeaking = false;
        let recognition = null;
        let synthesis = window.speechSynthesis;
        let fullResponse = '';

        // Speech Recognition setup
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        function initSpeechRecognition() {
            if (!SpeechRecognition) {
                voiceStatus.textContent = 'Speech not supported';
                return;
            }
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isListening = true;
                micBtn.classList.add('listening');
                micBtn.textContent = '🔴';
                voiceStatus.textContent = 'Listening...';
            };

            recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                if (event.results[event.results.length - 1].isFinal) {
                    sendMessage(transcript.trim());
                }
            };

            recognition.onend = () => {
                isListening = false;
                micBtn.classList.remove('listening');
                micBtn.textContent = '🎤';
                if (!isSpeaking) voiceStatus.textContent = 'Click mic to speak';
            };

            recognition.onerror = (e) => {
                console.log('Speech error:', e.error);
                isListening = false;
                micBtn.classList.remove('listening');
                micBtn.textContent = '🎤';
            };
        }

        function startListening() {
            if (recognition && !isListening && !isSpeaking) {
                try {
                    recognition.start();
                } catch(e) {
                    console.log('Recognition error:', e);
                }
            }
        }

        function speak(text) {
            if (!autoSpeakCheckbox.checked || !synthesis || !text) return;
            
            synthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            // Get a voice
            const voices = synthesis.getVoices();
            if (voices.length > 0) {
                const preferredVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
                utterance.voice = preferredVoice;
            }

            utterance.onstart = () => {
                isSpeaking = true;
                micBtn.classList.add('speaking');
                micBtn.textContent = '🔊';
                voiceStatus.textContent = 'Speaking...';
            };

            utterance.onend = () => {
                isSpeaking = false;
                micBtn.classList.remove('speaking');
                micBtn.textContent = '🎤';
                voiceStatus.textContent = 'Click mic to speak';
                if (continuousModeCheckbox.checked) {
                    setTimeout(startListening, 500);
                }
            };

            utterance.onerror = () => {
                isSpeaking = false;
                micBtn.classList.remove('speaking');
                micBtn.textContent = '🎤';
            };

            synthesis.speak(utterance);
        }

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected';
                micBtn.disabled = false;
                messageInput.disabled = false;
                sendBtn.disabled = false;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'response_chunk') {
                    fullResponse += data.text;
                    updateBotMessage(fullResponse);
                } else if (data.type === 'response_complete') {
                    speak(data.text);
                    fullResponse = '';
                }
            };

            ws.onclose = () => {
                statusDot.classList.remove('connected');
                statusText.textContent = 'Disconnected';
                micBtn.disabled = true;
                messageInput.disabled = true;
                sendBtn.disabled = true;
                setTimeout(connect, 3000);
            };
        }

        function addMessage(text, isUser) {
            const msg = document.createElement('div');
            msg.className = `message ${isUser ? 'user' : 'bot'}`;
            msg.textContent = text;
            chatContainer.appendChild(msg);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return msg;
        }

        let currentBotMessage = null;
        function updateBotMessage(text) {
            if (!currentBotMessage) {
                currentBotMessage = addMessage(text, false);
            } else {
                currentBotMessage.textContent = text;
            }
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage(text) {
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            addMessage(text, true);
            currentBotMessage = null;
            fullResponse = '';
            
            ws.send(JSON.stringify({ type: 'text', text: text }));
        }

        // Event listeners
        micBtn.addEventListener('click', () => {
            if (isSpeaking) {
                synthesis.cancel();
                return;
            }
            if (isListening) {
                recognition.stop();
            } else {
                startListening();
            }
        });

        sendBtn.addEventListener('click', () => {
            sendMessage(messageInput.value.trim());
            messageInput.value = '';
        });

        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage(messageInput.value.trim());
                messageInput.value = '';
            }
        });

        // Load voices
        if (synthesis) {
            synthesis.onvoiceschanged = () => synthesis.getVoices();
            synthesis.getVoices();
        }

        // Init
        initSpeechRecognition();
        connect();
    </script>
</body>
</html>'''

    async def start(self):
        """Start the web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        print(f"\n🌐 Pipecat Web Server started!")
        print(f"   📍 http://{self.config.host}:{self.config.port}")
        print(f"   📍 WebSocket: ws://{self.config.host}:{self.config.port}/ws")
        print(f"   📍 API: http://{self.config.host}:{self.config.port}/api/chat")
        print(f"\n   Press Ctrl+C to stop")
        
        # Keep running
        await asyncio.Event().wait()


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    import signal
    import sys
    
    parser = argparse.ArgumentParser(description="Pipecat Web Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--lm-url", default="http://127.0.0.1:1234/v1", help="LM Studio URL")
    args = parser.parse_args()
    
    def signal_handler(sig, frame):
        print("\n\n👋 Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    config = WebConfig(
        host=args.host,
        port=args.port,
        lm_studio_url=args.lm_url
    )
    
    server = PipecatWebServer(config)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
