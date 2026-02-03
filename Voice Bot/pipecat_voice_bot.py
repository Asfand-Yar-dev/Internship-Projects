"""
AI Voice Bot with Pipecat
==========================
Using Pipecat pipeline for voice bot

Requirements:
- pip install pipecat-ai[silero] aiohttp
- LM Studio running on localhost:1234
"""

import asyncio
import aiohttp

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
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# ============================================================================
# LOCAL LLM PROCESSOR (LM Studio)
# ============================================================================
class LocalLLMProcessor(FrameProcessor):
    """Pipecat processor for Local LLM (LM Studio)"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.conversation = []
        self.system_prompt = "You are a helpful voice assistant. Keep responses brief (1-2 sentences)."
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            # User input received
            user_text = frame.text
            print(f"\n🎤 User: {user_text}")
            
            # Get LLM response
            response = await self._get_response(user_text)
            print(f"🤖 Bot: {response}")
            
            # Send to TTS
            await self.push_frame(TextFrame(text=response))
        else:
            await self.push_frame(frame, direction)
    
    async def _get_response(self, user_text: str) -> str:
        """Call LM Studio API"""
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
                        # Add to conversation
                        self.conversation.append({"role": "user", "content": msg})
                        self.conversation.append({"role": "assistant", "content": response})
                        return response
                    else:
                        return "Sorry, LLM error occurred."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."


# ============================================================================
# TTS PROCESSOR (Text to Speech)
# ============================================================================
class TTSProcessor(FrameProcessor):
    """Simple TTS processor using pyttsx3"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            # Speak the text
            print(f"🔊 Speaking...")
            self.engine.say(frame.text)
            self.engine.runAndWait()
        
        await self.push_frame(frame, direction)


# ============================================================================
# STT PROCESSOR (Speech to Text)
# ============================================================================
class STTProcessor(FrameProcessor):
    """Speech to Text processor using SpeechRecognition"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.0
        self.mic = sr.Microphone()
        
        print("🎤 Calibrating mic...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready!")
    
    def listen(self) -> str:
        """Listen and return text"""
        import speech_recognition as sr
        try:
            with self.mic as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
            print("🔄 Processing...")
            return self.recognizer.recognize_google(audio)
        except:
            return ""


# ============================================================================
# PIPECAT VOICE BOT
# ============================================================================
class PipecatVoiceBot:
    """Voice Bot using Pipecat Pipeline"""
    
    def __init__(self):
        self.llm = LocalLLMProcessor()
        self.tts = TTSProcessor()
        self.stt = STTProcessor()
        
        # Build Pipecat Pipeline
        self.pipeline = Pipeline([
            self.llm,  # Process user input -> LLM response
            self.tts,  # Speak the response
        ])
    
    async def test_llm(self) -> bool:
        """Test LM Studio connection"""
        print("\n🔍 Testing LM Studio...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:1234/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            print(f"✅ Connected! Model: {models[0]['id']}")
                            return True
        except:
            pass
        print("❌ LM Studio not connected!")
        print("   1. Open LM Studio")
        print("   2. Load a model")
        print("   3. Start server (localhost:1234)")
        return False
    
    async def run(self):
        """Run the voice bot"""
        print("\n" + "="*50)
        print("🤖 PIPECAT VOICE BOT")
        print("="*50)
        
        await self.test_llm()
        
        print("\n" + "="*50)
        print("🎙️  VOICE MODE ACTIVE")
        print("   Bolo aur jawab suno")
        print("   'quit' bol ke band karo")
        print("   Ctrl+C se bhi band kar sakte ho")
        print("="*50)
        
        # Initialize pipeline
        task = PipelineTask(
            self.pipeline,
            params=PipelineParams(allow_interruptions=True)
        )
        
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
# MAIN
# ============================================================================
if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n\n👋 Bye bye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n🚀 Starting Pipecat Voice Bot...")
    bot = PipecatVoiceBot()
    asyncio.run(bot.run())
