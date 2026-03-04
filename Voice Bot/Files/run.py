"""
Pipecat Voice Bot - Unified Runner
===================================
Easy-to-use launcher for all voice bot modes.

Usage:
    python run.py                    # Interactive mode selector
    python run.py classic            # Classic voice bot
    python run.py pipecat            # Pipecat voice bot
    python run.py web                # Web interface
    python run.py advanced           # Advanced features demo
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 PIPECAT VOICE BOT                                    ║
║   ═══════════════════                                     ║
║                                                           ║
║   AI Voice Assistant with Local LLM                       ║
║   Powered by Pipecat + LM Studio                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_modes():
    """Print available modes"""
    modes = """
Available Modes (All use MICROPHONE for voice input):
──────────────────────────────────────────────────────
  1. classic     - Basic voice bot (speech_recognition + pyttsx3)
  2. pipecat     - Pipecat pipeline with streaming LLM
  3. web         - Web interface (browser-based chat)
  4. advanced    - Voice bot with function calling (time, weather, calc)
  5. daily       - Voice bot (local mic, or WebRTC with API key)
  6. tts-demo    - TTS providers demo

Options:
────────
  --help         - Show this help
  --check        - Check dependencies
"""
    print(modes)


def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking dependencies...\n")
    
    deps = [
        ("pipecat", "pipecat-ai[silero]"),
        ("aiohttp", "aiohttp"),
        ("speech_recognition", "SpeechRecognition"),
        ("pyttsx3", "pyttsx3"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("websockets", "websockets"),
    ]
    
    optional_deps = [
        ("edge_tts", "edge-tts"),
        ("torch", "torch"),
        ("pygame", "pygame"),
        ("pydub", "pydub"),
    ]
    
    all_ok = True
    
    print("Required packages:")
    for module, package in deps:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - pip install {package}")
            all_ok = False
    
    print("\nOptional packages:")
    for module, package in optional_deps:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚪ {package} - pip install {package}")
    
    print("\nLM Studio:")
    try:
        import aiohttp
        async def check_lm():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://127.0.0.1:1234/v1/models",
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = data.get("data", [])
                            if models:
                                print(f"  ✅ Connected - Model: {models[0]['id']}")
                                return True
            except:
                pass
            print("  ❌ Not running - Start LM Studio and load a model")
            return False
        asyncio.run(check_lm())
    except:
        print("  ❌ Cannot check (aiohttp not installed)")
    
    if all_ok:
        print("\n✅ All required dependencies installed!")
    else:
        print("\n⚠️ Some dependencies missing. Install with:")
        print("   pip install -r requirements.txt")
    
    return all_ok


def run_classic():
    """Run classic voice bot"""
    print("\n🚀 Starting Classic Voice Bot...")
    from voice_bot import ClassicVoiceBot
    bot = ClassicVoiceBot()
    asyncio.run(bot.run())


def run_pipecat():
    """Run Pipecat voice bot"""
    print("\n🚀 Starting Pipecat Voice Bot...")
    from voice_bot import PipecatVoiceBot
    bot = PipecatVoiceBot(streaming=True)
    asyncio.run(bot.run())


def run_web():
    """Run web interface"""
    print("\n🚀 Starting Web Interface...")
    from pipecat_web import PipecatWebServer, WebConfig
    config = WebConfig(host="localhost", port=8765)
    server = PipecatWebServer(config)
    asyncio.run(server.start())


def run_advanced():
    """Run advanced voice bot with function calling"""
    print("\n🚀 Starting Advanced Voice Bot...")
    from pipecat_advanced import AdvancedPipecatBot, VoiceBotConfig
    config = VoiceBotConfig(enable_function_calling=True)
    bot = AdvancedPipecatBot(config)
    asyncio.run(bot.run_voice())  # Use voice mode, not text demo


def run_daily():
    """Run Daily.co bot"""
    print("\n🚀 Starting Daily Bot...")
    api_key = os.getenv("DAILY_API_KEY", "")
    if not api_key:
        print("⚠️ No DAILY_API_KEY found, running in mock mode")
        from pipecat_daily import MockDailyVoiceBot, DailyConfig
        bot = MockDailyVoiceBot(DailyConfig())
    else:
        from pipecat_daily import DailyVoiceBot, DailyConfig
        bot = DailyVoiceBot(DailyConfig(api_key=api_key))
    asyncio.run(bot.run())


def run_tts_demo():
    """Run TTS demo"""
    print("\n🚀 Starting TTS Demo...")
    from pipecat_tts import demo
    asyncio.run(demo())


def interactive_mode():
    """Interactive mode selector"""
    print_banner()
    print_modes()
    
    while True:
        try:
            choice = input("\nSelect mode (1-6) or 'q' to quit: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                print("👋 Bye!")
                break
            elif choice in ['1', 'classic']:
                run_classic()
                break
            elif choice in ['2', 'pipecat']:
                run_pipecat()
                break
            elif choice in ['3', 'web']:
                run_web()
                break
            elif choice in ['4', 'advanced']:
                run_advanced()
                break
            elif choice in ['5', 'daily']:
                run_daily()
                break
            elif choice in ['6', 'tts-demo', 'tts']:
                run_tts_demo()
                break
            else:
                print("Invalid choice. Please enter 1-6 or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except EOFError:
            break


def main():
    """Main entry point"""
    import signal
    
    def signal_handler(sig, frame):
        print("\n\n👋 Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            print_banner()
            print_modes()
        elif arg in ['--check', 'check']:
            check_dependencies()
        elif arg in ['classic', '1']:
            run_classic()
        elif arg in ['pipecat', '2']:
            run_pipecat()
        elif arg in ['web', '3']:
            run_web()
        elif arg in ['advanced', '4']:
            run_advanced()
        elif arg in ['daily', '5']:
            run_daily()
        elif arg in ['tts-demo', 'tts', '6']:
            run_tts_demo()
        else:
            print(f"Unknown mode: {arg}")
            print_modes()
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
