"""
AI Voice Bot with Pipecat - Advanced Version
==============================================
Full Pipecat pipeline with:
- Voice Activity Detection (VAD) using Silero
- Streaming LLM responses
- Interruption handling
- Audio transport support

Requirements:
- pip install pipecat-ai[silero,daily] aiohttp sounddevice numpy
- LM Studio running on localhost:1234
"""

import asyncio
import aiohttp
import sys
import signal
from typing import Optional, AsyncGenerator, List, Dict, Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    StartFrame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    InterimTranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.sentence import SentenceAggregator


# ============================================================================
# CONFIGURATION
# ============================================================================
class BotConfig:
    """Configuration for the voice bot"""
    LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
    MODEL_NAME = "local-model"
    SYSTEM_PROMPT = """You are a helpful, friendly voice assistant. 
    Keep responses brief (1-2 sentences max) and conversational.
    Speak naturally as if having a casual conversation."""
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TTS_RATE = 160
    MIC_ENERGY_THRESHOLD = 300
    MIC_PAUSE_THRESHOLD = 1.0
    LISTEN_TIMEOUT = 8
    PHRASE_TIME_LIMIT = 15


# ============================================================================
# LOCAL LLM PROCESSOR (LM Studio) - Enhanced with Streaming
# ============================================================================
class LocalLLMProcessor(FrameProcessor):
    """Pipecat processor for Local LLM (LM Studio) with streaming support"""
    
    def __init__(
        self, 
        base_url: str = BotConfig.LM_STUDIO_URL,
        streaming: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.streaming = streaming
        self.conversation: List[Dict[str, str]] = []
        self.system_prompt = BotConfig.SYSTEM_PROMPT
        self._is_processing = False
        self._should_interrupt = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle interruptions
        if isinstance(frame, UserStartedSpeakingFrame):
            if self._is_processing:
                self._should_interrupt = True
                print("⚡ User interrupted!")
            await self.push_frame(frame, direction)
            return
        
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            print(f"\n🎤 User: {user_text}")
            
            self._is_processing = True
            self._should_interrupt = False
            
            # Signal LLM response start
            await self.push_frame(LLMFullResponseStartFrame())
            
            if self.streaming:
                response = await self._get_streaming_response(user_text)
            else:
                response = await self._get_response(user_text)
                print(f"🤖 Bot: {response}")
                await self.push_frame(TextFrame(text=response))
            
            # Signal LLM response end
            await self.push_frame(LLMFullResponseEndFrame())
            
            self._is_processing = False
        else:
            await self.push_frame(frame, direction)
    
    async def _get_response(self, user_text: str) -> str:
        """Call LM Studio API (non-streaming)"""
        messages = self._build_messages(user_text)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": BotConfig.MODEL_NAME,
                        "messages": messages,
                        "max_tokens": BotConfig.MAX_TOKENS,
                        "temperature": BotConfig.TEMPERATURE
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data["choices"][0]["message"]["content"]
                        self._save_to_conversation(user_text, response)
                        return response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    async def _get_streaming_response(self, user_text: str) -> str:
        """Call LM Studio API with streaming"""
        messages = self._build_messages(user_text)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": BotConfig.MODEL_NAME,
                        "messages": messages,
                        "max_tokens": BotConfig.MAX_TOKENS,
                        "temperature": BotConfig.TEMPERATURE,
                        "stream": True
                    }
                ) as resp:
                    if resp.status == 200:
                        full_response = ""
                        current_sentence = ""
                        
                        print("🤖 Bot: ", end="", flush=True)
                        
                        async for line in resp.content:
                            # Check for interruption
                            if self._should_interrupt:
                                print(" [interrupted]")
                                break
                            
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    import json
                                    data = json.loads(line[6:])
                                    chunk = data["choices"][0].get("delta", {}).get("content", "")
                                    if chunk:
                                        full_response += chunk
                                        current_sentence += chunk
                                        print(chunk, end="", flush=True)
                                        
                                        # Push sentences for TTS
                                        if any(p in chunk for p in ['.', '!', '?', '\n']):
                                            await self.push_frame(TextFrame(text=current_sentence.strip()))
                                            current_sentence = ""
                                except:
                                    pass
                        
                        print()  # New line
                        
                        # Push any remaining text
                        if current_sentence.strip():
                            await self.push_frame(TextFrame(text=current_sentence.strip()))
                        
                        if not self._should_interrupt:
                            self._save_to_conversation(user_text, full_response)
                        
                        return full_response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """Build message list for API"""
        if len(self.conversation) == 0:
            msg = f"[Instructions: {self.system_prompt}]\n\nUser: {user_text}"
        else:
            msg = user_text
        return self.conversation + [{"role": "user", "content": msg}]
    
    def _save_to_conversation(self, user_text: str, response: str):
        """Save exchange to conversation history"""
        if len(self.conversation) == 0:
            msg = f"[Instructions: {self.system_prompt}]\n\nUser: {user_text}"
        else:
            msg = user_text
        self.conversation.append({"role": "user", "content": msg})
        self.conversation.append({"role": "assistant", "content": response})
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation = []
        print("🗑️ Conversation history cleared")


# ============================================================================
# TTS PROCESSOR (Text to Speech) - Enhanced with Queue
# ============================================================================
class TTSProcessor(FrameProcessor):
    """TTS processor using pyttsx3 with sentence queuing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', BotConfig.TTS_RATE)
        self._is_speaking = False
        self._speech_queue: List[str] = []
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if text:
                await self._speak(text)
        
        await self.push_frame(frame, direction)
    
    async def _speak(self, text: str):
        """Speak text"""
        self._is_speaking = True
        await self.push_frame(BotStartedSpeakingFrame())
        
        print(f"🔊 Speaking: {text[:50]}..." if len(text) > 50 else f"🔊 Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
        self._is_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame())
    
    def stop_speaking(self):
        """Stop current speech"""
        if self._is_speaking:
            self.engine.stop()
            self._is_speaking = False


# ============================================================================
# STT PROCESSOR (Speech to Text) - Enhanced
# ============================================================================
class STTProcessor(FrameProcessor):
    """Speech to Text processor using SpeechRecognition with VAD-like behavior"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = BotConfig.MIC_ENERGY_THRESHOLD
        self.recognizer.pause_threshold = BotConfig.MIC_PAUSE_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()
        
        print("🎤 Calibrating mic...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready!")
    
    def listen(self) -> Optional[str]:
        """Listen and return text"""
        import speech_recognition as sr
        try:
            with self.mic as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=BotConfig.LISTEN_TIMEOUT, 
                    phrase_time_limit=BotConfig.PHRASE_TIME_LIMIT
                )
            print("🔄 Processing...")
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ STT service error: {e}")
            return None
        except Exception as e:
            print(f"❌ STT error: {e}")
            return None


# ============================================================================
# INTERRUPTION HANDLER PROCESSOR
# ============================================================================
class InterruptionHandler(FrameProcessor):
    """Handle user interruptions during bot speech"""
    
    def __init__(self, tts_processor: TTSProcessor, **kwargs):
        super().__init__(**kwargs)
        self.tts = tts_processor
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking - stop TTS if active
            if self.tts._is_speaking:
                print("⚡ Interrupting TTS...")
                self.tts.stop_speaking()
        
        await self.push_frame(frame, direction)


# ============================================================================
# CONVERSATION LOGGER PROCESSOR
# ============================================================================
class ConversationLogger(FrameProcessor):
    """Log conversation to file"""
    
    def __init__(self, log_file: str = "conversation_log.txt", **kwargs):
        super().__init__(**kwargs)
        self.log_file = log_file
        self.messages: List[Dict[str, str]] = []
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            self._log("USER", frame.text)
        elif isinstance(frame, TextFrame):
            self._log("BOT", frame.text)
        
        await self.push_frame(frame, direction)
    
    def _log(self, role: str, text: str):
        """Log message"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": timestamp, "role": role, "text": text}
        self.messages.append(entry)
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {role}: {text}\n")
        except:
            pass


# ============================================================================
# PIPECAT VOICE BOT - Enhanced
# ============================================================================
class PipecatVoiceBot:
    """Enhanced Voice Bot using Pipecat Pipeline"""
    
    def __init__(self, streaming: bool = True, log_conversation: bool = False):
        self.streaming = streaming
        self.log_conversation = log_conversation
        
        # Create processors
        self.llm = LocalLLMProcessor(streaming=streaming)
        self.tts = TTSProcessor()
        self.stt = STTProcessor()
        self.interruption_handler = InterruptionHandler(self.tts)
        
        # Build pipeline components
        pipeline_components = [
            self.llm,
            self.tts,
        ]
        
        # Add logger if enabled
        if log_conversation:
            self.logger = ConversationLogger()
            pipeline_components.insert(0, self.logger)
        
        # Build Pipecat Pipeline
        self.pipeline = Pipeline(pipeline_components)
        
        # Commands
        self.commands = {
            "clear": self._cmd_clear_history,
            "reset": self._cmd_clear_history,
            "help": self._cmd_help,
        }
    
    async def test_llm(self) -> bool:
        """Test LM Studio connection"""
        print("\n🔍 Testing LM Studio...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{BotConfig.LM_STUDIO_URL}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            print(f"✅ Connected! Model: {models[0]['id']}")
                            return True
        except:
            pass
        
        print("❌ LM Studio not connected!")
        print("   ➡️  Open LM Studio")
        print("   ➡️  Load a model")
        print("   ➡️  Start server (localhost:1234)")
        return False
    
    def _cmd_clear_history(self):
        """Clear conversation history"""
        self.llm.clear_history()
        self.tts.engine.say("Conversation cleared.")
        self.tts.engine.runAndWait()
    
    def _cmd_help(self):
        """Show help"""
        help_text = """
        Available commands:
        - Say 'clear' or 'reset' to clear conversation history
        - Say 'help' for this message
        - Say 'quit', 'exit', 'bye', or 'stop' to exit
        """
        print(help_text)
        self.tts.engine.say("Say clear to reset, or quit to exit.")
        self.tts.engine.runAndWait()
    
    def _check_command(self, text: str) -> bool:
        """Check if input is a command"""
        text_lower = text.lower().strip()
        for cmd, handler in self.commands.items():
            if cmd in text_lower:
                handler()
                return True
        return False
    
    async def run(self):
        """Run the voice bot"""
        mode_str = "Streaming" if self.streaming else "Standard"
        
        print("\n" + "="*50)
        print(f"🤖 PIPECAT VOICE BOT ({mode_str} Mode)")
        print("="*50)
        
        await self.test_llm()
        
        print("\n" + "="*50)
        print("🎙️  VOICE MODE ACTIVE")
        print("   Speak and get responses")
        print("   Say 'quit' or 'exit' to stop")
        print("   Say 'clear' to reset conversation")
        print("   Say 'help' for commands")
        print("   Press Ctrl+C to force quit")
        print("="*50)
        
        # Initialize pipeline
        task = PipelineTask(
            self.pipeline,
            params=PipelineParams(allow_interruptions=True)
        )
        
        # Send start frame
        await self.pipeline.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        
        # Greeting
        self.tts.engine.say("Hello! I am ready. Speak now.")
        self.tts.engine.runAndWait()
        
        # Main loop
        while True:
            try:
                # Listen
                user_text = self.stt.listen()
                
                if not user_text:
                    continue
                
                # Check for exit commands
                if any(w in user_text.lower() for w in ['quit', 'exit', 'bye', 'stop']):
                    self.tts.engine.say("Goodbye! Have a great day!")
                    self.tts.engine.runAndWait()
                    break
                
                # Check for other commands
                if self._check_command(user_text):
                    continue
                
                # Signal user started speaking (for interruption handling)
                await self.pipeline.process_frame(
                    UserStartedSpeakingFrame(), 
                    FrameDirection.DOWNSTREAM
                )
                
                # Process through pipeline
                transcription = TranscriptionFrame(
                    text=user_text,
                    user_id="user",
                    timestamp=""
                )
                await self.pipeline.process_frame(transcription, FrameDirection.DOWNSTREAM)
                
                # Signal user stopped speaking
                await self.pipeline.process_frame(
                    UserStoppedSpeakingFrame(), 
                    FrameDirection.DOWNSTREAM
                )
                
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        # Send end frame
        await self.pipeline.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


# ============================================================================
# WEBSOCKET TRANSPORT (for web-based UI)
# ============================================================================
class WebSocketTransport:
    """WebSocket transport for connecting to web clients"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
    
    async def start(self, bot: PipecatVoiceBot):
        """Start WebSocket server"""
        try:
            import websockets
        except ImportError:
            print("❌ websockets not installed. Run: pip install websockets")
            return
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            print(f"🌐 Client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    import json
                    data = json.loads(message)
                    
                    if data.get("type") == "transcription":
                        # Process user input
                        transcription = TranscriptionFrame(
                            text=data["text"],
                            user_id="web_user",
                            timestamp=""
                        )
                        await bot.pipeline.process_frame(
                            transcription, 
                            FrameDirection.DOWNSTREAM
                        )
            except:
                pass
            finally:
                self.clients.discard(websocket)
                print(f"🌐 Client disconnected")
        
        print(f"🌐 Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(handler, self.host, self.port):
            await asyncio.Future()  # Run forever
            self.pipeline,
            params=PipelineParams(allow_interruptions=True)
        
        # Send start frame
        await self.pipeline.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        
        # Greet
        self.tts.engine.say("Hello! I am ready. Speak now.")
        self.tts.engine.runAndWait()
        
        # Main loop
        while True:
            try:
                # Listen
                user_text = self.stt.listen()
                
                if not user_text:
                    continue
                
                # Exit check
                if any(w in user_text.lower() for w in ['quit', 'exit', 'bye', 'stop']):
                    self.tts.engine.say("Bye bye!")
                    self.tts.engine.runAndWait()
                    break
                
                # Process through pipeline
                transcription = TranscriptionFrame(
                    text=user_text,
                    user_id="user",
                    timestamp=""
                )
                await self.pipeline.process_frame(transcription, FrameDirection.DOWNSTREAM)
                
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
        
        # Send end frame
        await self.pipeline.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


# ============================================================================
# MAIN - Enhanced with CLI arguments
# ============================================================================
def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Pipecat Voice Bot - AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipecat_voice_bot.py                    # Default streaming mode
  python pipecat_voice_bot.py --no-stream        # Non-streaming mode
  python pipecat_voice_bot.py --log              # Enable conversation logging
  python pipecat_voice_bot.py --websocket        # Start WebSocket server
        """
    )
    
    parser.add_argument(
        "--no-stream", 
        action="store_true",
        help="Disable streaming responses"
    )
    parser.add_argument(
        "--log", 
        action="store_true",
        help="Enable conversation logging to file"
    )
    parser.add_argument(
        "--websocket", "-ws",
        action="store_true",
        help="Start WebSocket server for web clients"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n\n👋 Bye bye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_arguments()
    
    streaming = not args.no_stream
    mode_str = "Streaming" if streaming else "Standard"
    
    print(f"\n🚀 Starting Pipecat Voice Bot ({mode_str} Mode)...")
    
    bot = PipecatVoiceBot(
        streaming=streaming,
        log_conversation=args.log
    )
    
    if args.websocket:
        # Run with WebSocket server
        async def run_with_websocket():
            ws = WebSocketTransport(port=args.ws_port)
            await asyncio.gather(
                bot.run(),
                ws.start(bot)
            )
        asyncio.run(run_with_websocket())
    else:
        # Run normally
        asyncio.run(bot.run())
