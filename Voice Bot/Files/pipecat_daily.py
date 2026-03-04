"""
Pipecat Daily Transport Integration
====================================
Real-time audio/video transport using Daily.co WebRTC.

Features:
- Real-time WebRTC audio/video
- Cloud-based room management
- Low latency communication
- Multi-participant support

Requirements:
- pip install pipecat-ai[daily]
- Daily.co API key (free tier available)

Usage:
    export DAILY_API_KEY="your-api-key"
    python pipecat_daily.py
"""

import asyncio
import aiohttp
import os
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try to import Pipecat Daily components
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.frames.frames import (
        Frame,
        TextFrame,
        TranscriptionFrame,
        AudioRawFrame,
        StartFrame,
        EndFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    )
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False
    print("⚠️ Pipecat not installed. Run: pip install pipecat-ai[daily]")

# Try to import Daily transport
try:
    from pipecat.transports.services.daily import DailyParams, DailyTransport
    DAILY_AVAILABLE = True
except ImportError:
    DAILY_AVAILABLE = False
    print("⚠️ Daily transport not available. Run: pip install pipecat-ai[daily]")


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class DailyConfig:
    """Daily.co configuration"""
    api_key: str = ""
    room_url: str = ""
    room_name: str = "pipecat-voice-bot"
    lm_studio_url: str = "http://127.0.0.1:1234/v1"
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief and conversational."
    max_tokens: int = 150
    
    def __post_init__(self):
        # Try to load from environment
        if not self.api_key:
            self.api_key = os.getenv("DAILY_API_KEY", "")


# ============================================================================
# DAILY ROOM MANAGER
# ============================================================================
class DailyRoomManager:
    """Manage Daily.co rooms via REST API"""
    
    BASE_URL = "https://api.daily.co/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def create_room(self, name: str = None, expires_in: int = 3600) -> Dict[str, Any]:
        """Create a new Daily room"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "properties": {
                    "exp": int(asyncio.get_event_loop().time()) + expires_in,
                    "enable_chat": True,
                    "enable_knocking": False,
                    "start_video_off": True,
                    "start_audio_off": False,
                }
            }
            if name:
                payload["name"] = name
            
            async with session.post(
                f"{self.BASE_URL}/rooms",
                headers=self.headers,
                json=payload
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    raise Exception(f"Failed to create room: {error}")
    
    async def get_room(self, name: str) -> Optional[Dict[str, Any]]:
        """Get room details"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/rooms/{name}",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    
    async def delete_room(self, name: str) -> bool:
        """Delete a room"""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.BASE_URL}/rooms/{name}",
                headers=self.headers
            ) as resp:
                return resp.status == 200
    
    async def create_meeting_token(
        self, 
        room_name: str, 
        is_owner: bool = True,
        user_name: str = "Bot"
    ) -> str:
        """Create a meeting token for joining a room"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "properties": {
                    "room_name": room_name,
                    "is_owner": is_owner,
                    "user_name": user_name,
                    "enable_recording": False,
                }
            }
            
            async with session.post(
                f"{self.BASE_URL}/meeting-tokens",
                headers=self.headers,
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["token"]
                else:
                    error = await resp.text()
                    raise Exception(f"Failed to create token: {error}")


# ============================================================================
# LLM PROCESSOR FOR DAILY
# ============================================================================
class DailyLLMProcessor(FrameProcessor):
    """LLM Processor optimized for Daily transport"""
    
    def __init__(self, config: DailyConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.conversation = []
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            print(f"🎤 User: {user_text}")
            
            await self.push_frame(LLMFullResponseStartFrame())
            
            response = await self._get_streaming_response(user_text)
            
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
    
    async def _get_streaming_response(self, user_text: str) -> str:
        """Get streaming response from LLM"""
        messages = self._build_messages(user_text)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
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
                        current_sentence = ""
                        
                        print("🤖 Bot: ", end="", flush=True)
                        
                        async for line in resp.content:
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
                                        
                                        # Send complete sentences for TTS
                                        if any(p in chunk for p in ['.', '!', '?']):
                                            await self.push_frame(TextFrame(text=current_sentence.strip()))
                                            current_sentence = ""
                                except:
                                    pass
                        
                        print()  # New line
                        
                        # Send any remaining text
                        if current_sentence.strip():
                            await self.push_frame(TextFrame(text=current_sentence.strip()))
                        
                        # Save to conversation
                        self.conversation.append({"role": "user", "content": user_text})
                        self.conversation.append({"role": "assistant", "content": full_response})
                        
                        return full_response
                    else:
                        error_msg = "Sorry, LLM error occurred."
                        await self.push_frame(TextFrame(text=error_msg))
                        return error_msg
        except Exception as e:
            print(f"Error: {e}")
            error_msg = "Sorry, connection error."
            await self.push_frame(TextFrame(text=error_msg))
            return error_msg
    
    def _build_messages(self, user_text: str):
        """Build messages for API - no system role (LM Studio Mistral only supports user/assistant)."""
        if len(self.conversation) == 0:
            return [
                {"role": "user", "content": f"[Instructions: {self.config.system_prompt}]\n\nUser: {user_text}"}
            ]
        return self.conversation + [{"role": "user", "content": user_text}]


# ============================================================================
# DAILY VOICE BOT
# ============================================================================
class DailyVoiceBot:
    """Voice bot using Daily.co WebRTC transport"""
    
    def __init__(self, config: DailyConfig = None):
        self.config = config or DailyConfig()
        
        if not PIPECAT_AVAILABLE or not DAILY_AVAILABLE:
            print("❌ Required packages not available")
            print("   Install with: pip install pipecat-ai[daily]")
            sys.exit(1)
        
        if not self.config.api_key:
            print("❌ DAILY_API_KEY not set")
            print("   Set environment variable: export DAILY_API_KEY='your-key'")
            print("   Or pass in config: DailyConfig(api_key='your-key')")
            sys.exit(1)
        
        self.room_manager = DailyRoomManager(self.config.api_key)
        self.llm = DailyLLMProcessor(self.config)
    
    async def test_lm_studio(self) -> bool:
        """Test LM Studio connection"""
        print("🔍 Testing LM Studio connection...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.lm_studio_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            print(f"✅ LM Studio connected: {models[0]['id']}")
                            return True
        except:
            pass
        
        print("❌ LM Studio not connected!")
        print("   Start LM Studio and load a model")
        return False
    
    async def run(self):
        """Run the Daily voice bot"""
        print("\n" + "="*50)
        print("🤖 DAILY VOICE BOT")
        print("="*50)
        
        # Test LM Studio
        if not await self.test_lm_studio():
            return
        
        # Create or get room
        print(f"\n🏠 Setting up room: {self.config.room_name}")
        try:
            room = await self.room_manager.get_room(self.config.room_name)
            if not room:
                room = await self.room_manager.create_room(self.config.room_name)
            
            room_url = room["url"]
            print(f"✅ Room ready: {room_url}")
            
            # Create meeting token
            token = await self.room_manager.create_meeting_token(
                self.config.room_name,
                user_name="Voice Bot"
            )
            
            print(f"\n" + "="*50)
            print(f"🔗 JOIN URL: {room_url}")
            print(f"   Open this URL in your browser to talk to the bot!")
            print("="*50)
            
            # Setup Daily transport
            transport = DailyTransport(
                room_url,
                token,
                "Voice Bot",
                DailyParams(
                    audio_out_enabled=True,
                    audio_in_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=None,  # Use default
                    transcription_enabled=True,
                )
            )
            
            # Build pipeline
            pipeline = Pipeline([
                transport.input(),
                self.llm,
                transport.output(),
            ])
            
            # Create and run task
            task = PipelineTask(
                pipeline,
                params=PipelineParams(allow_interruptions=True)
            )
            
            runner = PipelineRunner()
            await runner.run(task)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MOCK DAILY BOT (for testing without Daily API) - Uses Microphone
# ============================================================================
class MockDailyVoiceBot:
    """Mock Daily bot - uses local microphone instead of WebRTC"""
    
    def __init__(self, config: DailyConfig = None):
        self.config = config or DailyConfig()
        self.llm = DailyLLMProcessor(self.config) if PIPECAT_AVAILABLE else None
    
    async def run(self):
        """Run with local microphone (voice mode)"""
        print("\n" + "="*50)
        print("🤖 VOICE BOT (Local Mode)")
        print("   Using microphone - no Daily API key needed")
        print("="*50)
        
        if not self.llm:
            print("❌ Pipecat not available")
            return
        
        # Test LM Studio
        print("🔍 Testing LM Studio...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.lm_studio_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        print("✅ LM Studio connected")
                    else:
                        print("❌ LM Studio not responding")
                        return
        except:
            print("❌ LM Studio not running")
            return
        
        # Setup STT and TTS
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
                
                # Get response via LLM
                response = await self.llm._get_streaming_response(user_text)
                
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
def main():
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Daily.co Voice Bot")
    parser.add_argument("--api-key", help="Daily.co API key")
    parser.add_argument("--room", default="pipecat-voice-bot", help="Room name")
    parser.add_argument("--lm-url", default="http://127.0.0.1:1234/v1", help="LM Studio URL")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (text only)")
    args = parser.parse_args()
    
    def signal_handler(sig, frame):
        print("\n\n👋 Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    config = DailyConfig(
        api_key=args.api_key or os.getenv("DAILY_API_KEY", ""),
        room_name=args.room,
        lm_studio_url=args.lm_url
    )
    
    if args.mock or not config.api_key:
        if not args.mock:
            print("⚠️ No Daily API key found, running in mock mode")
        bot = MockDailyVoiceBot(config)
    else:
        bot = DailyVoiceBot(config)
    
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
