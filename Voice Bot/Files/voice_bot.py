"""
AI Voice Bot
=============
A voice assistant with optional Pipecat pipeline support.

Modes:
  - Classic: Uses basic speech_recognition + pyttsx3
  - Pipecat: Uses Pipecat pipeline architecture with VAD

Usage:
  python voice_bot.py              # Classic mode
  python voice_bot.py --pipecat    # Pipecat mode
  python voice_bot.py --streaming  # Pipecat with streaming responses
"""

import asyncio
import aiohttp
import argparse
import sys
from typing import Optional, AsyncGenerator
from abc import ABC, abstractmethod


# ============================================================================
# BASE VOICE BOT (Abstract)
# ============================================================================
class BaseVoiceBot(ABC):
    """Abstract base class for voice bots"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1"):
        self.base_url = base_url
        self.conversation = []
        self.system_prompt = "You are a helpful voice assistant. Keep responses brief and conversational."
    
    @abstractmethod
    async def run(self):
        """Main run loop"""
        pass
    
    async def test_llm_connection(self) -> bool:
        """Test if LM Studio is running"""
        print("\n🔍 Testing LM Studio connection...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            print(f"✅ Connected! Model: {models[0]['id']}")
                            return True
        except Exception as e:
            pass
        
        print("❌ LM Studio not connected!")
        print("   ➡️  Open LM Studio")
        print("   ➡️  Load a model")
        print("   ➡️  Start server on localhost:1234")
        return False
    
    async def get_response(self, user_text: str) -> str:
        """Get response from LLM"""
        if len(self.conversation) == 0:
            msg = f"[Instructions: {self.system_prompt}]\n\nUser: {user_text}"
        else:
            msg = user_text
        
        try:
            async with aiohttp.ClientSession() as session:
                messages = self.conversation + [{"role": "user", "content": msg}]
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data["choices"][0]["message"]["content"]
                        self.conversation.append({"role": "user", "content": msg})
                        self.conversation.append({"role": "assistant", "content": response})
                        return response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    async def get_streaming_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """Get streaming response from LLM"""
        if len(self.conversation) == 0:
            msg = f"[Instructions: {self.system_prompt}]\n\nUser: {user_text}"
        else:
            msg = user_text
        
        try:
            async with aiohttp.ClientSession() as session:
                messages = self.conversation + [{"role": "user", "content": msg}]
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": messages,
                        "max_tokens": 150,
                        "temperature": 0.7,
                        "stream": True
                    }
                ) as resp:
                    if resp.status == 200:
                        full_response = ""
                        async for line in resp.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    import json
                                    data = json.loads(line[6:])
                                    chunk = data["choices"][0].get("delta", {}).get("content", "")
                                    if chunk:
                                        full_response += chunk
                                        yield chunk
                                except:
                                    pass
                        
                        # Save to conversation
                        self.conversation.append({"role": "user", "content": msg})
                        self.conversation.append({"role": "assistant", "content": full_response})
                    else:
                        yield "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            yield "Sorry, connection error."


# ============================================================================
# CLASSIC VOICE BOT (Original Implementation)
# ============================================================================
class ClassicVoiceBot(BaseVoiceBot):
    """Classic voice bot using speech_recognition + pyttsx3"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1"):
        super().__init__(base_url)
        
        import speech_recognition as sr
        import pyttsx3
        
        # Speaking TTS
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        
        # Listening STT
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.0
        self.mic = sr.Microphone()
        
        print("🎤 Calibrating mic...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready!")
    
    def speak(self, text: str):
        """Text to speech"""
        print(f"🤖 Bot: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen(self) -> str:
        """Listen and return transcribed text"""
        import speech_recognition as sr
        try:
            with self.mic as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
            print("🔄 Processing...")
            return self.recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            print(f"Listen error: {e}")
            return ""
    
    async def run(self):
        """Main conversation loop"""
        print("\n" + "="*50)
        print("🤖 CLASSIC VOICE BOT")
        print("="*50)
        
        await self.test_llm_connection()
        
        print("\n" + "="*50)
        print("🎙️  VOICE MODE ACTIVE")
        print("   Say 'quit' or 'exit' to stop")
        print("   Press Ctrl+C to force quit")
        print("="*50)
        
        self.speak("Hello! How can I help you today?")
        
        while True:
            try:
                user_text = self.listen()
                
                if not user_text:
                    continue
                
                print(f"👤 You: {user_text}")
                
                # Exit check
                if any(w in user_text.lower() for w in ['quit', 'exit', 'bye', 'stop']):
                    self.speak("Goodbye! Have a great day!")
                    break
                
                print("🧠 Thinking...")
                response = await self.get_response(user_text)
                self.speak(response)
                
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break


# ============================================================================
# PIPECAT VOICE BOT (Pipeline Architecture)
# ============================================================================
class PipecatVoiceBot(BaseVoiceBot):
    """Voice bot using Pipecat pipeline architecture"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", streaming: bool = False):
        super().__init__(base_url)
        self.streaming = streaming
        self._setup_components()
    
    def _setup_components(self):
        """Initialize Pipecat components"""
        try:
            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.pipeline.task import PipelineTask, PipelineParams
            from pipecat.frames.frames import (
                Frame, TextFrame, TranscriptionFrame, 
                StartFrame, EndFrame, LLMFullResponseStartFrame,
                LLMFullResponseEndFrame
            )
            from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
            
            self.Pipeline = Pipeline
            self.PipelineTask = PipelineTask
            self.PipelineParams = PipelineParams
            self.TranscriptionFrame = TranscriptionFrame
            self.TextFrame = TextFrame
            self.StartFrame = StartFrame
            self.EndFrame = EndFrame
            self.FrameProcessor = FrameProcessor
            self.FrameDirection = FrameDirection
            self.Frame = Frame
            
        except ImportError as e:
            print(f"❌ Pipecat not installed: {e}")
            print("   Install with: pip install pipecat-ai[silero]")
            sys.exit(1)
        
        # Initialize TTS
        import pyttsx3
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160)
        
        # Initialize STT
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.0
        self.mic = sr.Microphone()
        
        print("🎤 Calibrating mic...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready!")
        
        # Create processors
        self._create_processors()
    
    def _create_processors(self):
        """Create Pipecat frame processors"""
        bot = self  # Reference for closures
        
        class LLMProcessor(self.FrameProcessor):
            """Process transcriptions and get LLM responses"""
            
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                
                if isinstance(frame, bot.TranscriptionFrame):
                    user_text = frame.text
                    print(f"\n👤 User: {user_text}")
                    
                    if bot.streaming:
                        # Streaming response
                        print("🧠 ", end="", flush=True)
                        full_response = ""
                        async for chunk in bot.get_streaming_response(user_text):
                            print(chunk, end="", flush=True)
                            full_response += chunk
                        print()  # New line
                        await self.push_frame(bot.TextFrame(text=full_response))
                    else:
                        # Regular response
                        print("🧠 Thinking...")
                        response = await bot.get_response(user_text)
                        print(f"🤖 Bot: {response}")
                        await self.push_frame(bot.TextFrame(text=response))
                else:
                    await self.push_frame(frame, direction)
        
        class TTSProcessor(self.FrameProcessor):
            """Convert text to speech"""
            
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                
                if isinstance(frame, bot.TextFrame):
                    print("🔊 Speaking...")
                    bot.tts_engine.say(frame.text)
                    bot.tts_engine.runAndWait()
                
                await self.push_frame(frame, direction)
        
        self.llm_processor = LLMProcessor()
        self.tts_processor = TTSProcessor()
        
        # Build pipeline
        self.pipeline = self.Pipeline([
            self.llm_processor,
            self.tts_processor,
        ])
    
    def listen(self) -> str:
        """Listen and return transcribed text"""
        import speech_recognition as sr
        try:
            with self.mic as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
            print("🔄 Processing...")
            return self.recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            print(f"Listen error: {e}")
            return ""
    
    async def run(self):
        """Main conversation loop using Pipecat pipeline"""
        print("\n" + "="*50)
        print("🤖 PIPECAT VOICE BOT" + (" (Streaming)" if self.streaming else ""))
        print("="*50)
        
        await self.test_llm_connection()
        
        print("\n" + "="*50)
        print("🎙️  VOICE MODE ACTIVE")
        print("   Say 'quit' or 'exit' to stop")
        print("   Press Ctrl+C to force quit")
        print("="*50)
        
        # Initialize pipeline
        task = self.PipelineTask(
            self.pipeline,
            params=self.PipelineParams(allow_interruptions=True)
        )
        
        # Send start frame
        await self.pipeline.process_frame(self.StartFrame(), self.FrameDirection.DOWNSTREAM)
        
        # Greeting
        self.tts_engine.say("Hello! I am ready. Speak now.")
        self.tts_engine.runAndWait()
        
        # Main loop
        while True:
            try:
                user_text = self.listen()
                
                if not user_text:
                    continue
                
                # Exit check
                if any(w in user_text.lower() for w in ['quit', 'exit', 'bye', 'stop']):
                    self.tts_engine.say("Goodbye!")
                    self.tts_engine.runAndWait()
                    break
                
                # Process through pipeline
                transcription = self.TranscriptionFrame(
                    text=user_text,
                    user_id="user",
                    timestamp=""
                )
                await self.pipeline.process_frame(transcription, self.FrameDirection.DOWNSTREAM)
                
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
        
        # Send end frame
        await self.pipeline.process_frame(self.EndFrame(), self.FrameDirection.DOWNSTREAM)


# ============================================================================
# FACTORY & MAIN
# ============================================================================
def create_voice_bot(mode: str = "classic", streaming: bool = False) -> BaseVoiceBot:
    """Factory function to create the appropriate voice bot"""
    if mode == "pipecat":
        return PipecatVoiceBot(streaming=streaming)
    else:
        return ClassicVoiceBot()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Voice Bot with Pipecat support")
    parser.add_argument(
        "--pipecat", "-p",
        action="store_true",
        help="Use Pipecat pipeline mode"
    )
    parser.add_argument(
        "--streaming", "-s",
        action="store_true",
        help="Enable streaming responses (implies --pipecat)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:1234/v1",
        help="LM Studio API URL (default: http://127.0.0.1:1234/v1)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        print("\n\n👋 Bye bye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Determine mode
    if args.streaming:
        mode = "pipecat"
        streaming = True
        print("\n🚀 Starting Voice Bot (Pipecat + Streaming)...")
    elif args.pipecat:
        mode = "pipecat"
        streaming = False
        print("\n🚀 Starting Voice Bot (Pipecat Mode)...")
    else:
        mode = "classic"
        streaming = False
        print("\n🚀 Starting Voice Bot (Classic Mode)...")
    
    # Create and run bot
    if mode == "pipecat":
        bot = PipecatVoiceBot(base_url=args.url, streaming=streaming)
    else:
        bot = ClassicVoiceBot(base_url=args.url)
    
    asyncio.run(bot.run())