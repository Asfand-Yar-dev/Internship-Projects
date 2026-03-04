"""
Pipecat TTS Providers
======================
Multiple TTS providers for the voice bot:
- pyttsx3 (offline, basic)
- Edge TTS (free, high quality)
- ElevenLabs (premium quality)

Usage:
    from pipecat_tts import create_tts_processor, TTSProvider
    
    tts = create_tts_processor(TTSProvider.EDGE)
"""

import asyncio
from typing import Optional, AsyncGenerator
from enum import Enum
from dataclasses import dataclass
import io

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# ============================================================================
# TTS PROVIDER ENUM
# ============================================================================
class TTSProvider(Enum):
    PYTTSX3 = "pyttsx3"
    EDGE = "edge"
    ELEVENLABS = "elevenlabs"


# ============================================================================
# TTS CONFIGURATION
# ============================================================================
@dataclass
class TTSConfig:
    """TTS configuration"""
    provider: TTSProvider = TTSProvider.PYTTSX3
    
    # pyttsx3 settings
    pyttsx3_rate: int = 160
    
    # Edge TTS settings
    edge_voice: str = "en-US-AriaNeural"
    edge_rate: str = "+0%"
    edge_pitch: str = "+0Hz"
    
    # ElevenLabs settings
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    elevenlabs_model: str = "eleven_monolingual_v1"


# ============================================================================
# BASE TTS PROCESSOR
# ============================================================================
class BaseTTSProcessor(FrameProcessor):
    """Base class for TTS processors"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_speaking = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if text:
                await self._speak(text)
        
        await self.push_frame(frame, direction)
    
    async def _speak(self, text: str):
        """Override in subclass"""
        raise NotImplementedError


# ============================================================================
# PYTTSX3 TTS PROCESSOR
# ============================================================================
class Pyttsx3TTSProcessor(BaseTTSProcessor):
    """TTS using pyttsx3 (offline)"""
    
    def __init__(self, config: TTSConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or TTSConfig()
        
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.config.pyttsx3_rate)
        
        print("🔊 TTS: pyttsx3 (offline)")
    
    async def _speak(self, text: str):
        """Speak using pyttsx3"""
        self._is_speaking = True
        await self.push_frame(BotStartedSpeakingFrame())
        
        print(f"🔊 Speaking: {text[:50]}..." if len(text) > 50 else f"🔊 Speaking: {text}")
        
        # pyttsx3 is synchronous, run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._speak_sync, text)
        
        self._is_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame())
    
    def _speak_sync(self, text: str):
        """Synchronous speak"""
        self.engine.say(text)
        self.engine.runAndWait()


# ============================================================================
# EDGE TTS PROCESSOR
# ============================================================================
class EdgeTTSProcessor(BaseTTSProcessor):
    """TTS using Microsoft Edge TTS (free, high quality)"""
    
    def __init__(self, config: TTSConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or TTSConfig()
        
        # Check if edge-tts is installed
        try:
            import edge_tts
            self.edge_tts = edge_tts
            print(f"🔊 TTS: Edge TTS (voice: {self.config.edge_voice})")
        except ImportError:
            print("❌ edge-tts not installed. Run: pip install edge-tts")
            self.edge_tts = None
    
    async def _speak(self, text: str):
        """Speak using Edge TTS"""
        if not self.edge_tts:
            return
        
        self._is_speaking = True
        await self.push_frame(BotStartedSpeakingFrame())
        
        print(f"🔊 Speaking: {text[:50]}..." if len(text) > 50 else f"🔊 Speaking: {text}")
        
        try:
            # Generate audio
            communicate = self.edge_tts.Communicate(
                text,
                self.config.edge_voice,
                rate=self.config.edge_rate,
                pitch=self.config.edge_pitch
            )
            
            # Collect audio data
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            
            # Play audio
            audio_bytes = audio_data.getvalue()
            if audio_bytes:
                await self._play_audio(audio_bytes)
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
        
        self._is_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame())
    
    async def _play_audio(self, audio_bytes: bytes):
        """Play audio using sounddevice or pygame"""
        try:
            # Try sounddevice first
            import sounddevice as sd
            import numpy as np
            from pydub import AudioSegment
            
            # Convert MP3 to raw audio
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            samples = np.array(audio.get_array_of_samples())
            
            # Play
            sd.play(samples, audio.frame_rate)
            sd.wait()
        except ImportError:
            # Fallback to pygame
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(io.BytesIO(audio_bytes))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            except ImportError:
                print("⚠️ No audio player available (install sounddevice or pygame)")
    
    @classmethod
    async def list_voices(cls) -> list:
        """List available Edge TTS voices"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return voices
        except:
            return []


# ============================================================================
# EDGE TTS STREAMING PROCESSOR
# ============================================================================
class EdgeTTSStreamingProcessor(BaseTTSProcessor):
    """Streaming TTS using Edge TTS - speaks while generating"""
    
    def __init__(self, config: TTSConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or TTSConfig()
        
        try:
            import edge_tts
            self.edge_tts = edge_tts
            print(f"🔊 TTS: Edge TTS Streaming (voice: {self.config.edge_voice})")
        except ImportError:
            print("❌ edge-tts not installed. Run: pip install edge-tts")
            self.edge_tts = None
        
        self._audio_queue = asyncio.Queue()
        self._playback_task = None
    
    async def start(self):
        """Start the audio playback task"""
        self._playback_task = asyncio.create_task(self._audio_playback_loop())
    
    async def stop(self):
        """Stop the audio playback task"""
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
    
    async def _speak(self, text: str):
        """Stream TTS audio"""
        if not self.edge_tts:
            return
        
        self._is_speaking = True
        await self.push_frame(BotStartedSpeakingFrame())
        
        print(f"🔊 Streaming: {text[:50]}..." if len(text) > 50 else f"🔊 Streaming: {text}")
        
        try:
            communicate = self.edge_tts.Communicate(
                text,
                self.config.edge_voice,
                rate=self.config.edge_rate,
                pitch=self.config.edge_pitch
            )
            
            # Stream audio chunks to queue
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    await self._audio_queue.put(chunk["data"])
            
            # Signal end of this utterance
            await self._audio_queue.put(None)
            
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
        
        self._is_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame())
    
    async def _audio_playback_loop(self):
        """Continuously play audio from queue"""
        try:
            import sounddevice as sd
            import numpy as np
            from pydub import AudioSegment
            
            buffer = io.BytesIO()
            
            while True:
                chunk = await self._audio_queue.get()
                
                if chunk is None:
                    # End of utterance, play buffered audio
                    if buffer.tell() > 0:
                        buffer.seek(0)
                        audio = AudioSegment.from_mp3(buffer)
                        samples = np.array(audio.get_array_of_samples())
                        sd.play(samples, audio.frame_rate)
                        sd.wait()
                        buffer = io.BytesIO()
                else:
                    buffer.write(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ Playback error: {e}")


# ============================================================================
# ELEVENLABS TTS PROCESSOR
# ============================================================================
class ElevenLabsTTSProcessor(BaseTTSProcessor):
    """TTS using ElevenLabs (premium quality)"""
    
    def __init__(self, config: TTSConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or TTSConfig()
        
        if not self.config.elevenlabs_api_key:
            import os
            self.config.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        
        if not self.config.elevenlabs_api_key:
            print("⚠️ ElevenLabs API key not set")
            print("   Set ELEVENLABS_API_KEY environment variable")
        else:
            print(f"🔊 TTS: ElevenLabs (voice: {self.config.elevenlabs_voice_id})")
    
    async def _speak(self, text: str):
        """Speak using ElevenLabs"""
        if not self.config.elevenlabs_api_key:
            return
        
        self._is_speaking = True
        await self.push_frame(BotStartedSpeakingFrame())
        
        print(f"🔊 Speaking: {text[:50]}..." if len(text) > 50 else f"🔊 Speaking: {text}")
        
        try:
            import aiohttp
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice_id}"
            
            headers = {
                "xi-api-key": self.config.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "model_id": self.config.elevenlabs_model,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        await self._play_audio(audio_bytes)
                    else:
                        error = await resp.text()
                        print(f"❌ ElevenLabs error: {error}")
        except Exception as e:
            print(f"❌ ElevenLabs error: {e}")
        
        self._is_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame())
    
    async def _play_audio(self, audio_bytes: bytes):
        """Play audio"""
        try:
            import sounddevice as sd
            import numpy as np
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            samples = np.array(audio.get_array_of_samples())
            sd.play(samples, audio.frame_rate)
            sd.wait()
        except Exception as e:
            print(f"❌ Playback error: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def create_tts_processor(
    provider: TTSProvider = TTSProvider.PYTTSX3,
    config: TTSConfig = None,
    **kwargs
) -> BaseTTSProcessor:
    """Create a TTS processor based on provider"""
    
    config = config or TTSConfig(provider=provider)
    config.provider = provider
    
    if provider == TTSProvider.PYTTSX3:
        return Pyttsx3TTSProcessor(config, **kwargs)
    elif provider == TTSProvider.EDGE:
        return EdgeTTSProcessor(config, **kwargs)
    elif provider == TTSProvider.ELEVENLABS:
        return ElevenLabsTTSProcessor(config, **kwargs)
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


# ============================================================================
# VOICE LIST UTILITIES
# ============================================================================
async def list_edge_voices(language: str = "en") -> list:
    """List available Edge TTS voices for a language"""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        return [v for v in voices if v["Locale"].startswith(language)]
    except:
        return []


def list_pyttsx3_voices() -> list:
    """List available pyttsx3 voices"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return [{"id": v.id, "name": v.name} for v in voices]
    except:
        return []


# ============================================================================
# DEMO
# ============================================================================
async def demo():
    """Demo TTS providers"""
    print("\n" + "="*50)
    print("🔊 TTS PROVIDER DEMO")
    print("="*50)
    
    test_text = "Hello! This is a test of the text to speech system."
    
    print("\n1. pyttsx3 (offline):")
    tts1 = create_tts_processor(TTSProvider.PYTTSX3)
    await tts1._speak(test_text)
    
    print("\n2. Edge TTS (free, high quality):")
    tts2 = create_tts_processor(TTSProvider.EDGE)
    await tts2._speak(test_text)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    print("\nAvailable pyttsx3 voices:")
    for v in list_pyttsx3_voices():
        print(f"  - {v['name']}")
    
    print("\nRunning TTS demo...")
    asyncio.run(demo())
