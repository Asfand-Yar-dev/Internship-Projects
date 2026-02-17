import asyncio
import os
import json
import uuid
from aiohttp import web
import aiohttp
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:1234/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")
API_KEY = os.getenv("API_KEY", "not-needed")

# Store connected clients and their conversations
clients = {}
conversations = {}


async def get_llm_response(session_id: str, user_text: str) -> str:
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({"role": "user", "content": user_text})
    
    if len(conversations[session_id]) == 1:
        messages = [
            {"role": "user", "content": f"[You are a helpful voice assistant. Keep responses brief (1-2 sentences).]\n\n{user_text}"}
        ]
    else:
        # Subsequent messages: use conversation history as-is
        messages = conversations[session_id].copy()
    
    try:
        async with aiohttp.ClientSession() as http:
            headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
            async with http.post(
                f"{BASE_URL}/chat/completions",
                json={"model": MODEL_NAME, "messages": messages, "max_tokens": 150},
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data["choices"][0]["message"]["content"]
                    conversations[session_id].append({"role": "assistant", "content": response})
                    return response
                else:
                    return f"Error: LLM returned status {resp.status}"
    except Exception as e:
        return f"Error connecting to LLM: {e}"


async def handle_websocket(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session_id = str(uuid.uuid4())[:8]
    clients[session_id] = ws
    print(f"✅ Client connected: {session_id}")
    
    await ws.send_json({"type": "connected", "session_id": session_id})
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "text":
                    user_text = data.get("text", "")
                    print(f"🎤 [{session_id}] User: {user_text}")
                    
                    response = await get_llm_response(session_id, user_text)
                    print(f"🤖 [{session_id}] Bot: {response}")
                    
                    await ws.send_json({"type": "response", "text": response})
                elif data.get("type") == "clear":
                    # Clear conversation history for this session
                    if session_id in conversations:
                        conversations[session_id] = []
                    print(f"🗑️ [{session_id}] History cleared")
                    await ws.send_json({"type": "history_cleared"})
    finally:
        del clients[session_id]
        if session_id in conversations:
            del conversations[session_id]
        print(f"❌ Client disconnected: {session_id}")
    
    return ws


async def handle_index(request):
    """Serve the HTML interface"""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Voice Bot</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui; background: #1a1a2e; color: #fff; margin: 0; padding: 20px; min-height: 100vh; }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { text-align: center; }
        .status { text-align: center; padding: 10px; margin-bottom: 20px; }
        .status.connected { color: #4ade80; }
        .status.disconnected { color: #f87171; }
        .chat { background: #16213e; border-radius: 12px; padding: 20px; min-height: 300px; max-height: 400px; overflow-y: auto; margin-bottom: 20px; }
        .message { padding: 10px 15px; border-radius: 10px; margin: 8px 0; max-width: 80%; }
        .user { background: #4f46e5; margin-left: auto; }
        .bot { background: #374151; }
        .controls { text-align: center; }
        .mic-btn { width: 80px; height: 80px; border-radius: 50%; border: none; background: #4f46e5; color: white; font-size: 32px; cursor: pointer; transition: all 0.2s; }
        .mic-btn:hover { transform: scale(1.05); }
        .mic-btn:disabled { background: #666; cursor: not-allowed; }
        .mic-btn.listening { background: #ef4444; animation: pulse 1s infinite; }
        .mic-btn.speaking { background: #22c55e; }
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        .voice-status { margin-top: 15px; color: #9ca3af; }
        .input-row { display: flex; gap: 10px; margin-top: 20px; }
        .input-row input { flex: 1; padding: 12px; border: none; border-radius: 8px; background: #16213e; color: white; }
        .input-row button { padding: 12px 20px; border: none; border-radius: 8px; background: #4f46e5; color: white; cursor: pointer; }
        .settings { text-align: center; margin-top: 15px; color: #9ca3af; }
        .settings label { margin: 0 10px; cursor: pointer; }
        .history-info { text-align: center; margin-top: 10px; font-size: 12px; color: #6b7280; }
        .clear-btn { background: #dc2626; padding: 8px 16px; border: none; border-radius: 6px; color: white; cursor: pointer; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Bot</h1>
        <div class="status disconnected" id="status">Connecting...</div>
        <div class="chat" id="chat"></div>
        <div class="controls">
            <button class="mic-btn" id="micBtn" disabled>🎤</button>
            <div class="voice-status" id="voiceStatus">Click mic to start</div>
        </div>
        <div class="settings">
            <label><input type="checkbox" id="autoSpeak" checked> Auto-speak</label>
            <label><input type="checkbox" id="continuousMode" checked> Continuous mode</label>
            <label>Voice: <select id="voiceSelect"></select></label>
        </div>
        <div class="history-info">
            <span id="historyCount">Messages: 0</span>
            <button class="clear-btn" id="clearBtn">Clear History</button>
        </div>
        <div class="input-row">
            <input type="text" id="textInput" placeholder="Or type here..." disabled>
            <button id="sendBtn" disabled>Send</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const micBtn = document.getElementById('micBtn');
        const status = document.getElementById('status');
        const voiceStatus = document.getElementById('voiceStatus');
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendBtn');
        const autoSpeak = document.getElementById('autoSpeak');
        const continuousMode = document.getElementById('continuousMode');
        const historyCount = document.getElementById('historyCount');
        const clearBtn = document.getElementById('clearBtn');
        const voiceSelect = document.getElementById('voiceSelect');
        let ws, recognition, synthesis = window.speechSynthesis;
        let isListening = false, isSpeaking = false;
        let messageCount = 0;
        function loadVoice(){
            const voices = synthesis.getVoices();
            const currentValue = voiceSelect.value;
             voiceSelect.innerHTML = voices.map((v, i) => 
                `<option value="${i}">${v.name}</option>`
            ).join('');
             if (currentValue) voiceSelect.value = currentValue;  // Restore selection
}
synthesis.onvoiceschanged = loadVoice;
loadVoice();
        // Speech Recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {
                isListening = true;
                micBtn.classList.add('listening');
                micBtn.textContent = '🔴';
                voiceStatus.textContent = 'Listening...';
            };
            
            recognition.onresult = (e) => {
                const text = e.results[0][0].transcript;
                if (text.trim()) sendMessage(text);
            };
            
            recognition.onend = () => {
                isListening = false;
                micBtn.classList.remove('listening');
                micBtn.textContent = '🎤';
                if (!isSpeaking) {
                    voiceStatus.textContent = continuousMode.checked ? 'Waiting for response...' : 'Click mic to speak';
                }
            };
            
            recognition.onerror = (e) => {
                isListening = false;
                micBtn.classList.remove('listening');
                micBtn.textContent = '🎤';
                // Auto-restart on no-speech error if continuous mode
                if (e.error === 'no-speech' && continuousMode.checked && !isSpeaking) {
                    setTimeout(startListening, 500);
                }
            };
        }
        
        function startListening() {
            if (recognition && !isListening && !isSpeaking) {
                try {
                    recognition.start();
                } catch(e) {
                    console.log('Recognition restart error:', e);
                }
            }
        }
        
        function speak(text) {
            if (!autoSpeak.checked || !synthesis) {
                // If not speaking, start listening immediately in continuous mode
                if (continuousMode.checked) {
                    setTimeout(startListening, 300);
                }
                return;
            }
            synthesis.cancel();
             const utterance = new SpeechSynthesisUtterance(text);
             utterance.rate = 1.0;
             utterance.pitch = 1.2;
             const voices = synthesis.getVoices();
             utterance.voice = voices[voiceSelect.value];
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
                // Auto-start listening in continuous mode
                if (continuousMode.checked) {
                    voiceStatus.textContent = 'Starting to listen...';
                    setTimeout(startListening, 500);
                } else {
                    voiceStatus.textContent = 'Click mic to speak';
                }
            };
            utterance.onerror = () => {
                isSpeaking = false;
                micBtn.classList.remove('speaking');
                micBtn.textContent = '🎤';
                if (continuousMode.checked) {
                    setTimeout(startListening, 500);
                }
            };
            synthesis.speak(utterance);
        }
        
        function updateHistoryCount() {
            messageCount++;
            historyCount.textContent = 'Messages: ' + messageCount;
        }
        
        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            updateHistoryCount();
        }
        
        function sendMessage(text) {
            if (!text || !ws) return;
            addMessage(text, true);
            ws.send(JSON.stringify({type: 'text', text}));
        }
        
        function clearHistory() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'clear'}));
                chat.innerHTML = '';
                messageCount = 0;
                historyCount.textContent = 'Messages: 0';
                voiceStatus.textContent = continuousMode.checked ? 'History cleared - Click mic to start' : 'Click mic to speak';
            }
        }
        
        function connect() {
            ws = new WebSocket('ws://' + location.host + '/ws');
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.className = 'status connected';
                micBtn.disabled = false;
                textInput.disabled = false;
                sendBtn.disabled = false;
                voiceStatus.textContent = continuousMode.checked ? 'Click mic to start continuous mode' : 'Click mic to speak';
            };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'response') {
                    addMessage(data.text, false);
                    speak(data.text);
                } else if (data.type === 'history_cleared') {
                    chat.innerHTML = '';
                    messageCount = 0;
                    historyCount.textContent = 'Messages: 0';
                }
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected - Reconnecting...';
                status.className = 'status disconnected';
                micBtn.disabled = true;
                textInput.disabled = true;
                sendBtn.disabled = true;
                setTimeout(connect, 3000);
            };
        }
        
        micBtn.onclick = () => {
            if (isSpeaking) { synthesis.cancel(); return; }
            if (isListening) recognition.stop();
            else if (recognition) startListening();
        };
        
        clearBtn.onclick = clearHistory;
        sendBtn.onclick = () => { sendMessage(textInput.value); textInput.value = ''; };
        textInput.onkeypress = (e) => { if (e.key === 'Enter') { sendMessage(textInput.value); textInput.value = ''; } };
        
        connect();
    </script>
</body>
</html>'''
    return web.Response(text=html, content_type='text/html')


async def main():
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/ws', handle_websocket)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8765)
    await site.start()
    print("\n" + "="*50)
    print("🎙️  Voice Bot Server Running!")
    print("="*50)
    print(f"🌐 Open in browser: http://localhost:8765")
    print(f"🔗 LLM API: {BASE_URL}")
    print("="*50)
    print("Press Ctrl+C to stop\n")
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down server...")