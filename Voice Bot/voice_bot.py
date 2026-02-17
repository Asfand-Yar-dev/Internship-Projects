import asyncio
import aiohttp
import speech_recognition as sr
import pyttsx3


class VoiceBot:
    def __init__(self):
        # LLM
        self.base_url = "http://127.0.0.1:1234/v1"
        self.conversation = []
        
        # Speaking TTS
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        
        # Listening STT
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.0
        self.mic = sr.Microphone()
        
        print("Calibrating mic...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready!")
    
    def speak(self, text):
        print(f"Bot: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen(self):
        try:
            with self.mic as source:
                print("\nListening...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
            print("Processing...")
            return self.recognizer.recognize_google(audio)
        except:
            return ""
    
    async def get_response(self, user_text):
        if len(self.conversation) == 0:
            msg = f"[You are a helpful assistant. Keep responses brief.]\n\nUser: {user_text}"
        else:
            msg = user_text
        
        # Get response from LLM
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": self.conversation + [{"role": "user", "content": msg}],
                        "max_tokens": 150
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data["choices"][0]["message"]["content"]
                        # Add to conversation only on success
                        self.conversation.append({"role": "user", "content": msg})
                        self.conversation.append({"role": "assistant", "content": response})
                        return response
                    else:
                        return "Sorry, LLM error."
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, connection error."
    
    async def run(self):
        print("\n" + "="*40)
        print("VOICE BOT READY!")
        print("="*40)
        
        self.speak("Hello! How can I help you today?")
        
        while True:
            user_text = self.listen()
            
            if not user_text:
                continue
            
            print(f"You: {user_text}")
            
            print("Thinking...")
            response = await self.get_response(user_text)
            self.speak(response)


if __name__ == "__main__":
    print("\nStarting Voice Bot...")
    bot = VoiceBot()
    asyncio.run(bot.run())