"""
Pipecat Advanced Components
============================
Advanced Pipecat processors and transports for voice bot:
- Silero VAD integration
- Audio queue management  
- Real-time audio transport
- Function calling support

Requirements:
- pip install pipecat-ai[silero] aiohttp sounddevice numpy
"""

import asyncio
import aiohttp
import json
import time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    AudioRawFrame,
    StartFrame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    SystemFrame,
    ControlFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class VoiceBotConfig:
    """Configuration for advanced voice bot"""
    # LLM Settings
    lm_studio_url: str = "http://127.0.0.1:1234/v1"
    model_name: str = "local-model"
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief."
    max_tokens: int = 200
    temperature: float = 0.7
    
    # Audio Settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # VAD Settings
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.25
    vad_min_silence_duration: float = 0.5
    
    # TTS Settings
    tts_rate: int = 160
    
    # Behavior
    allow_interruptions: bool = True
    enable_function_calling: bool = False
    
    # Tools/Functions (for function calling)
    tools: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# AUDIO BUFFER
# ============================================================================
class AudioBuffer:
    """Thread-safe audio buffer for real-time processing"""
    
    def __init__(self, max_seconds: float = 30.0, sample_rate: int = 16000):
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer: List[bytes] = []
        self._lock = asyncio.Lock()
    
    async def add(self, audio_data: bytes):
        """Add audio data to buffer"""
        async with self._lock:
            self.buffer.append(audio_data)
            # Trim if too long
            total_samples = sum(len(b) // 2 for b in self.buffer)
            while total_samples > self.max_samples and len(self.buffer) > 1:
                self.buffer.pop(0)
                total_samples = sum(len(b) // 2 for b in self.buffer)
    
    async def get_all(self) -> bytes:
        """Get all buffered audio"""
        async with self._lock:
            result = b''.join(self.buffer)
            self.buffer.clear()
            return result
    
    async def clear(self):
        """Clear buffer"""
        async with self._lock:
            self.buffer.clear()


# ============================================================================
# SILERO VAD PROCESSOR
# ============================================================================
class SileroVADProcessor(FrameProcessor):
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._audio_buffer = AudioBuffer(sample_rate=sample_rate)
        
        # Load Silero VAD
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            print("✅ Silero VAD loaded")
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD: {e}")
            print("   VAD will be disabled")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame) and self._model:
            # Process audio for VAD
            is_speech = await self._detect_speech(frame.audio)
            
            current_time = time.time()
            
            if is_speech and not self._is_speaking:
                # Speech started
                self._speech_start_time = current_time
                self._silence_start_time = None
                
                # Wait for minimum speech duration
                if (current_time - self._speech_start_time) >= self.min_speech_duration:
                    self._is_speaking = True
                    await self.push_frame(UserStartedSpeakingFrame())
                    print("🎤 Speech detected")
            
            elif not is_speech and self._is_speaking:
                # Possible speech end
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                elif (current_time - self._silence_start_time) >= self.min_silence_duration:
                    # Speech ended
                    self._is_speaking = False
                    self._speech_start_time = None
                    self._silence_start_time = None
                    await self.push_frame(UserStoppedSpeakingFrame())
                    print("🔇 Speech ended")
            
            elif is_speech and self._is_speaking:
                # Continue speaking
                self._silence_start_time = None
        
        await self.push_frame(frame, direction)
    
    async def _detect_speech(self, audio_bytes: bytes) -> bool:
        """Detect if audio contains speech"""
        if self._model is None:
            return True  # Assume speech if no VAD
        
        try:
            import torch
            import numpy as np
            
            # Convert bytes to tensor
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)
            
            # Get speech probability
            speech_prob = self._model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > self.threshold
        except Exception:
            return True


# ============================================================================
# FUNCTION CALLING LLM PROCESSOR
# ============================================================================
class FunctionCallingLLMProcessor(FrameProcessor):
    """LLM Processor with function/tool calling support"""
    
    def __init__(
        self,
        config: VoiceBotConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.conversation: List[Dict[str, str]] = []
        self._functions: Dict[str, Callable[..., Awaitable[str]]] = {}
    
    def register_function(
        self, 
        name: str, 
        handler: Callable[..., Awaitable[str]],
        description: str = "",
        parameters: Dict[str, Any] = None
    ):
        """Register a function that the LLM can call"""
        self._functions[name] = handler
        
        # Add to tools list
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}}
            }
        }
        self.config.tools.append(tool_def)
        print(f"📌 Registered function: {name}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            print(f"\n🎤 User: {user_text}")
            
            response = await self._get_response_with_functions(user_text)
            print(f"🤖 Bot: {response}")
            
            await self.push_frame(TextFrame(text=response))
        else:
            await self.push_frame(frame, direction)
    
    async def _get_response_with_functions(self, user_text: str) -> str:
        """Get response, potentially calling functions"""
        messages = self._build_messages(user_text)
        
        try:
            async with aiohttp.ClientSession() as session:
                request_body = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                }
                
                # Add tools if function calling enabled
                if self.config.enable_function_calling and self.config.tools:
                    request_body["tools"] = self.config.tools
                    request_body["tool_choice"] = "auto"
                
                async with session.post(
                    f"{self.config.lm_studio_url}/chat/completions",
                    json=request_body
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choice = data["choices"][0]
                        message = choice["message"]
                        
                        # Check for function calls
                        if "tool_calls" in message and message["tool_calls"]:
                            return await self._handle_tool_calls(
                                user_text, messages, message["tool_calls"]
                            )
                        
                        response = message.get("content", "")
                        self._save_exchange(user_text, response)
                        return response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    async def _handle_tool_calls(
        self, 
        user_text: str,
        messages: List[Dict],
        tool_calls: List[Dict]
    ) -> str:
        """Handle function/tool calls from LLM"""
        results = []
        
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"].get("arguments", "{}"))
            
            print(f"🔧 Calling function: {func_name}({func_args})")
            
            if func_name in self._functions:
                try:
                    result = await self._functions[func_name](**func_args)
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": str(result)
                    })
                    print(f"✅ Function result: {result}")
                except Exception as e:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": f"Error: {e}"
                    })
            else:
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool", 
                    "content": f"Unknown function: {func_name}"
                })
        
        # Send function results back to LLM
        messages.append({"role": "assistant", "tool_calls": tool_calls})
        messages.extend(results)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.lm_studio_url}/chat/completions",
                    json={
                        "model": self.config.model_name,
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data["choices"][0]["message"]["content"]
                        self._save_exchange(user_text, response)
                        return response
        except:
            pass
        
        return "I completed the task."
    
    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """Build messages list - no system role (LM Studio Mistral only supports user/assistant)."""
        if len(self.conversation) == 0:
            return [
                {"role": "user", "content": f"[Instructions: {self.config.system_prompt}]\n\nUser: {user_text}"}
            ]
        return self.conversation + [{"role": "user", "content": user_text}]
    
    def _save_exchange(self, user_text: str, response: str):
        """Save to conversation history (user/assistant only)."""
        if len(self.conversation) == 0:
            self.conversation.append({"role": "user", "content": f"[Instructions: {self.config.system_prompt}]\n\nUser: {user_text}"})
        else:
            self.conversation.append({"role": "user", "content": user_text})
        self.conversation.append({"role": "assistant", "content": response})


# ============================================================================
# AUDIO INPUT PROCESSOR (using sounddevice)
# ============================================================================
class AudioInputProcessor(FrameProcessor):
    """Real-time audio input using sounddevice"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._stream = None
        self._running = False
    
    async def start(self):
        """Start audio capture"""
        try:
            import sounddevice as sd
            import numpy as np
            
            self._running = True
            
            def audio_callback(indata, frames, time_info, status):
                if self._running:
                    audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                    asyncio.create_task(self._process_audio(audio_bytes))
            
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                callback=audio_callback
            )
            self._stream.start()
            print("🎤 Audio input started")
            
        except Exception as e:
            print(f"❌ Audio input error: {e}")
    
    async def stop(self):
        """Stop audio capture"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("🔇 Audio input stopped")
    
    async def _process_audio(self, audio_bytes: bytes):
        """Process captured audio"""
        frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=self.sample_rate,
            num_channels=self.channels
        )
        await self.push_frame(frame)


# ============================================================================
# AUDIO OUTPUT PROCESSOR (using sounddevice)
# ============================================================================
class AudioOutputProcessor(FrameProcessor):
    """Real-time audio output using sounddevice"""
    
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._queue = asyncio.Queue()
    
    async def start(self):
        """Start audio playback"""
        try:
            import sounddevice as sd
            import numpy as np
            
            def audio_callback(outdata, frames, time_info, status):
                try:
                    data = self._queue.get_nowait()
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                    if len(audio_np) < frames:
                        audio_np = np.pad(audio_np, (0, frames - len(audio_np)))
                    outdata[:] = audio_np[:frames].reshape(-1, 1)
                except:
                    outdata.fill(0)
            
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback
            )
            self._stream.start()
            print("🔊 Audio output started")
            
        except Exception as e:
            print(f"❌ Audio output error: {e}")
    
    async def stop(self):
        """Stop audio playback"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("🔇 Audio output stopped")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame):
            await self._queue.put(frame.audio)
        
        await self.push_frame(frame, direction)


# ============================================================================
# EXAMPLE FUNCTIONS FOR FUNCTION CALLING
# ============================================================================
async def get_current_time() -> str:
    """Get the current time"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


async def get_weather(city: str = "London") -> str:
    """Get weather for a city (mock)"""
    # Mock weather data
    return f"The weather in {city} is sunny, 22°C"


async def calculate(expression: str) -> str:
    """Calculate a math expression"""
    try:
        # Safe eval for basic math
        allowed = set('0123456789+-*/().% ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return str(result)
        return "Invalid expression"
    except:
        return "Calculation error"


# ============================================================================
# ADVANCED BOT WITH FUNCTION CALLING
# ============================================================================
class AdvancedPipecatBot:
    """Advanced voice bot with function calling and VAD"""
    
    def __init__(self, config: VoiceBotConfig = None):
        self.config = config or VoiceBotConfig()
        
        # Create processors
        self.llm = FunctionCallingLLMProcessor(self.config)
        
        # Register example functions
        if self.config.enable_function_calling:
            self.llm.register_function(
                "get_current_time",
                get_current_time,
                "Get the current time",
                {"type": "object", "properties": {}}
            )
            self.llm.register_function(
                "get_weather",
                get_weather,
                "Get weather for a city",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    }
                }
            )
            self.llm.register_function(
                "calculate",
                calculate,
                "Calculate a math expression",
                {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            )
    
    async def run_demo(self):
        """Run a text-based demo (for testing without microphone)"""
        print("\n" + "="*50)
        print("🤖 ADVANCED PIPECAT BOT (Text Demo)")
        print("="*50)
        print("Type your messages, 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Bye!")
                    break
                
                # Process through LLM
                frame = TranscriptionFrame(
                    text=user_input,
                    user_id="user",
                    timestamp=""
                )
                await self.llm.process_frame(frame, FrameDirection.DOWNSTREAM)
                
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
            except EOFError:
                break
    
    async def run_voice(self):
        """Run with voice input (microphone)"""
        print("\n" + "="*50)
        print("🤖 ADVANCED PIPECAT BOT (Voice Mode)")
        print("="*50)
        
        # Setup STT
        try:
            import speech_recognition as sr
            import pyttsx3
        except ImportError:
            print("❌ Missing: pip install SpeechRecognition pyttsx3")
            return
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 1.0
        mic = sr.Microphone()
        
        # Setup TTS
        tts = pyttsx3.init()
        tts.setProperty('rate', 160)
        
        print("🎤 Calibrating microphone...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready!")
        
        print("\n" + "="*50)
        print("🎙️  VOICE MODE ACTIVE")
        print("   Speak to the bot!")
        print("   Say 'quit' or 'exit' to stop")
        print("="*50)
        
        # Greeting
        tts.say("Hello! I am ready. Speak now.")
        tts.runAndWait()
        
        while True:
            try:
                # Listen
                print("\n🎤 Listening...")
                with mic as source:
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
                
                print("🔄 Processing...")
                try:
                    user_text = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    print("❓ Could not understand")
                    continue
                except sr.RequestError as e:
                    print(f"❌ STT error: {e}")
                    continue
                
                print(f"👤 You: {user_text}")
                
                # Check exit
                if any(w in user_text.lower() for w in ['quit', 'exit', 'bye', 'stop']):
                    tts.say("Goodbye!")
                    tts.runAndWait()
                    print("👋 Bye!")
                    break
                
                # Get response
                print("🧠 Thinking...")
                response = await self.llm._get_response_with_functions(user_text)
                print(f"🤖 Bot: {response}")
                
                # Speak response
                print("🔊 Speaking...")
                tts.say(response)
                tts.runAndWait()
                
            except sr.WaitTimeoutError:
                continue
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Pipecat Voice Bot")
    parser.add_argument("--functions", "-f", action="store_true", help="Enable function calling")
    args = parser.parse_args()
    
    config = VoiceBotConfig(
        enable_function_calling=args.functions
    )
    
    bot = AdvancedPipecatBot(config)
    asyncio.run(bot.run_demo())
