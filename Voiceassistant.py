import os
import re
import csv
import sys
import json
import time
import queue
import logging
import datetime
import threading
import tempfile
import wave
import struct
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("voice_assistant.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_FILE = "conversations.csv"
CSV_HEADERS = [
    "timestamp", "session_id", "turn_id", "speaker",
    "raw_text", "intent", "entities", "sentiment",
    "response_text", "processing_time_ms", "tts_engine", "stt_engine",
]

INTENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b", "greeting"),
    (r"\b(bye|goodbye|see\s+you|farewell|exit|quit|stop)\b", "farewell"),
    (r"\b(what\s+(time|date|day)|current\s+time|today)\b", "datetime_query"),
    (r"\b(weather|temperature|forecast|rain|sunny|cloudy)\b", "weather_query"),
    (r"\b(help|assist|support|guide|how\s+to|tutorial)\b", "help_request"),
    (r"\b(calculate|compute|math|add|subtract|multiply|divide|\d+\s*[\+\-\*/]\s*\d+)\b", "math_query"),
    (r"\b(remind|reminder|alarm|schedule|appointment|meeting)\b", "reminder"),
    (r"\b(search|find|look\s+up|google|information\s+about)\b", "search_query"),
    (r"\b(joke|funny|laugh|humor|entertainment)\b", "entertainment"),
    (r"\b(news|latest|update|current\s+events|headline)\b", "news_query"),
    (r"\b(thank|thanks|thank\s+you|appreciate)\b", "gratitude"),
    (r"\b(sorry|apologize|excuse\s+me|pardon)\b", "apology"),
    (r"\b(my\s+name\s+is|i\s+am|call\s+me|i'm)\b", "self_introduction"),
    (r"\b(what\s+is\s+your\s+name|who\s+are\s+you|introduce\s+yourself)\b", "assistant_query"),
]

ENTITY_PATTERNS: Dict[str, str] = {
    "email":    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "phone":    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "url":      r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*",
    "number":   r"\b\d+(?:\.\d+)?\b",
    "date":     r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
    "time":     r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?\b",
    "currency": r"\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*(?:dollars?|euros?|rupees?|INR|USD|EUR)\b",
    "math_expr":r"\b\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?\b",
    "name":     r"\b(?:my\s+name\s+is|i\s+am|call\s+me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
}

SENTIMENT_PATTERNS = {
    "positive": r"\b(great|excellent|awesome|good|happy|love|fantastic|wonderful|perfect|amazing|nice|brilliant)\b",
    "negative": r"\b(bad|terrible|awful|hate|horrible|worst|poor|disappointing|frustrating|annoying|useless)\b",
    "neutral":  r".*",
}

# ── CSV Logger ─────────────────────────────────────────────────────────────────
class ConversationLogger:
    def __init__(self, filepath: str = CSV_FILE):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not Path(self.filepath).exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
            logger.info(f"Created CSV log: {self.filepath}")

    def log(self, record: dict):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow({k: record.get(k, "") for k in CSV_HEADERS})

    def get_history(self, session_id: str) -> List[dict]:
        history = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("session_id") == session_id:
                    history.append(row)
        return history

    def get_stats(self) -> dict:
        total = user_turns = assistant_turns = 0
        intents: Dict[str, int] = {}
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row["speaker"] == "user":
                    user_turns += 1
                    intent = row.get("intent", "unknown")
                    intents[intent] = intents.get(intent, 0) + 1
                else:
                    assistant_turns += 1
        return {
            "total_records": total,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "intent_distribution": intents,
        }


# ── NLP Engine (RegEx + Pattern Matching) ─────────────────────────────────────
class NLPEngine:
    def detect_intent(self, text: str) -> str:
        lower = text.lower()
        for pattern, intent in INTENT_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                return intent
        return "general_query"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                flat = [m if isinstance(m, str) else m[0] for m in matches]
                entities[entity_type] = flat
        return entities

    def detect_sentiment(self, text: str) -> str:
        lower = text.lower()
        pos = len(re.findall(SENTIMENT_PATTERNS["positive"], lower, re.IGNORECASE))
        neg = len(re.findall(SENTIMENT_PATTERNS["negative"], lower, re.IGNORECASE))
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"

    def eval_math(self, text: str) -> Optional[str]:
        expr = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)", text)
        if not expr:
            return None
        a, op, b = float(expr.group(1)), expr.group(2), float(expr.group(3))
        try:
            result = {"+": a+b, "-": a-b, "*": a*b, "/": a/b if b != 0 else None, "^": a**b}[op]
            return f"{a} {op} {b} = {result}" if result is not None else "Division by zero!"
        except Exception:
            return None

    def analyze(self, text: str) -> dict:
        return {
            "intent":    self.detect_intent(text),
            "entities":  self.extract_entities(text),
            "sentiment": self.detect_sentiment(text),
            "math_result": self.eval_math(text),
        }


# ── GenAI Client (supports Gemini, OpenAI-compatible, or offline fallback) ────
class GenAIClient:
    def __init__(self):
        self.provider = "offline"
        self.client = None
        self._init_provider()

    def _init_provider(self):
        # Try Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash-latest")
                self.provider = "gemini"
                logger.info("✅ GenAI provider: Google Gemini")
                return
            except ImportError:
                logger.warning("google-generativeai not installed, trying next provider")

        # Try OpenAI-compatible (Groq, Together, OpenAI, etc.)
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key, base_url=openai_base)
                self.provider = "openai"
                self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                logger.info(f"✅ GenAI provider: OpenAI-compatible ({openai_base})")
                return
            except ImportError:
                logger.warning("openai package not installed, using offline mode")

        logger.info("ℹ️  GenAI provider: Offline (rule-based responses)")
        self.provider = "offline"

    def _offline_response(self, text: str, nlp_result: dict, history: List[dict]) -> str:
        intent = nlp_result["intent"]
        entities = nlp_result["entities"]
        math = nlp_result["math_result"]

        responses = {
            "greeting":        "Hello! I'm your Voice AI Assistant. How can I help you today?",
            "farewell":        "Goodbye! It was great talking with you. Have a wonderful day!",
            "datetime_query":  f"The current date and time is: {datetime.datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}.",
            "weather_query":   "I don't have live weather data in offline mode, but I can help with many other things! Try asking me math questions, for the time, or just have a conversation.",
            "help_request":    "I can help with: answering questions, math calculations, telling the time/date, general conversation, and much more. Just speak naturally!",
            "entertainment":   "Why don't scientists trust atoms? Because they make up everything! 😄 Want to hear another one?",
            "gratitude":       "You're welcome! I'm always here to help. Is there anything else I can assist you with?",
            "apology":         "No worries at all! How can I assist you?",
            "assistant_query": "I'm your Voice AI Assistant, powered by advanced NLP and GenAI. I can understand your voice, process your requests, and respond intelligently!",
            "self_introduction": f"Nice to meet you! I'll remember you during our conversation. How can I help you today?",
            "math_query":      f"Math result: {math}" if math else "Please provide a math expression like '5 + 3' or '10 * 4'.",
            "reminder":        "I've noted your reminder request! In a full deployment, I'd integrate with your calendar.",
            "search_query":    "I'd search the web for that, but I'm in offline mode. Try enabling a GenAI API key for full web-aware responses!",
            "news_query":      "I don't have live news access in offline mode. Enable a GenAI API key for current events!",
        }

        base = responses.get(intent, None)
        if base:
            return base

        # Fallback: echo with context
        sentiment = nlp_result["sentiment"]
        ent_str = ", ".join(f"{k}: {v}" for k, v in entities.items() if v) if entities else ""
        reply = f"I understood you said: '{text}'. "
        if ent_str:
            reply += f"I detected: {ent_str}. "
        reply += f"Your tone seems {sentiment}. "
        reply += "I'm in offline mode — set GEMINI_API_KEY or OPENAI_API_KEY for smarter responses!"
        return reply

    def generate(self, user_text: str, nlp_result: dict, history: List[dict], session_id: str) -> str:
        if self.provider == "offline":
            return self._offline_response(user_text, nlp_result, history)

        # Build system prompt
        system_prompt = (
            "You are a helpful, friendly, and concise Voice AI Assistant. "
            "You respond in clear, spoken-friendly language without markdown formatting. "
            "Keep responses under 3 sentences unless detail is requested. "
            f"Detected intent: {nlp_result['intent']}. "
            f"Entities found: {json.dumps(nlp_result['entities'])}. "
            f"User sentiment: {nlp_result['sentiment']}."
        )

        # Conversation history (last 6 turns)
        messages = [{"role": "user", "content": system_prompt},
                    {"role": "assistant", "content": "Understood! I'll respond as a voice-friendly AI assistant."}]
        for turn in history[-6:]:
            if turn["speaker"] == "user":
                messages.append({"role": "user", "content": turn["raw_text"]})
            else:
                messages.append({"role": "assistant", "content": turn["response_text"]})
        messages.append({"role": "user", "content": user_text})

        try:
            if self.provider == "gemini":
                # Flatten for Gemini
                flat = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                resp = self.client.generate_content(flat)
                return resp.text.strip()

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                )
                return resp.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GenAI error: {e}")
            return self._offline_response(user_text, nlp_result, history)

        return self._offline_response(user_text, nlp_result, history)


# ── Speech-to-Text ─────────────────────────────────────────────────────────────
class SpeechToText:
    def __init__(self):
        self.engine = "none"
        self._init_engine()

    def _init_engine(self):
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.sr = sr
            self.engine = "speech_recognition"
            logger.info("✅ STT engine: SpeechRecognition (Google Web API)")
        except ImportError:
            logger.warning("SpeechRecognition not installed — STT unavailable (text mode only)")

    def listen(self, timeout: int = 8, phrase_limit: int = 15) -> Optional[str]:
        if self.engine == "none":
            return None
        try:
            with self.sr.Microphone() as source:
                print("\n🎤  Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            print("🔄  Transcribing...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"STT result: '{text}'")
            return text
        except self.sr.WaitTimeoutError:
            logger.info("STT: No speech detected (timeout)")
            return None
        except self.sr.UnknownValueError:
            logger.info("STT: Could not understand audio")
            return None
        except self.sr.RequestError as e:
            logger.error(f"STT API error: {e}")
            return None
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        if self.engine == "none":
            return None
        try:
            with self.sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            return self.recognizer.recognize_google(audio)
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None


# ── Text-to-Speech ─────────────────────────────────────────────────────────────
class TextToSpeech:
    def __init__(self):
        self.engine_name = "none"
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        # Try pyttsx3 (offline, cross-platform)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.engine.setProperty("volume", 0.95)
            # Pick a pleasant voice if available
            voices = self.engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.name.lower() or "samantha" in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    break
            self.engine_name = "pyttsx3"
            logger.info("✅ TTS engine: pyttsx3 (offline)")
            return
        except Exception:
            pass

        # Try gTTS (online, Google)
        try:
            from gtts import gTTS
            import pygame
            self.gtts = gTTS
            self.pygame = pygame
            self.pygame.mixer.init()
            self.engine_name = "gtts"
            logger.info("✅ TTS engine: gTTS + pygame (online)")
            return
        except Exception:
            pass

        logger.warning("No TTS engine available — text output only")

    def speak(self, text: str) -> bool:
        if not text:
            return False
        print(f"\n🔊  Assistant: {text}\n")

        if self.engine_name == "pyttsx3":
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            except Exception as e:
                logger.error(f"pyttsx3 error: {e}")

        elif self.engine_name == "gtts":
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp = f.name
                tts = self.gtts(text=text, lang="en", slow=False)
                tts.save(tmp)
                self.pygame.mixer.music.load(tmp)
                self.pygame.mixer.music.play()
                while self.pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.unlink(tmp)
                return True
            except Exception as e:
                logger.error(f"gTTS error: {e}")

        return False

    def save_to_file(self, text: str, filepath: str) -> bool:
        if self.engine_name == "pyttsx3":
            try:
                self.engine.save_to_file(text, filepath)
                self.engine.runAndWait()
                return True
            except Exception as e:
                logger.error(f"Save audio error: {e}")
        elif self.engine_name == "gtts":
            try:
                from gtts import gTTS
                gTTS(text=text, lang="en").save(filepath)
                return True
            except Exception as e:
                logger.error(f"Save gTTS error: {e}")
        return False


# ── Main Voice Assistant ───────────────────────────────────────────────────────
class VoiceAssistant:
    def __init__(self):
        self.session_id = datetime.datetime.now().strftime("sess_%Y%m%d_%H%M%S")
        self.turn_id = 0
        self.logger = ConversationLogger()
        self.nlp = NLPEngine()
        self.ai = GenAIClient()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.running = False
        self.user_name: Optional[str] = None
        logger.info(f"Voice Assistant initialized | Session: {self.session_id}")

    def _log_turn(self, user_text: str, response: str, nlp_result: dict,
                  processing_ms: float):
        self.turn_id += 1
        ts = datetime.datetime.now().isoformat()

        # User turn
        self.logger.log({
            "timestamp":         ts,
            "session_id":        self.session_id,
            "turn_id":           self.turn_id,
            "speaker":           "user",
            "raw_text":          user_text,
            "intent":            nlp_result["intent"],
            "entities":          json.dumps(nlp_result["entities"]),
            "sentiment":         nlp_result["sentiment"],
            "response_text":     "",
            "processing_time_ms": "",
            "tts_engine":        "",
            "stt_engine":        self.stt.engine,
        })

        # Assistant turn
        self.logger.log({
            "timestamp":         datetime.datetime.now().isoformat(),
            "session_id":        self.session_id,
            "turn_id":           self.turn_id,
            "speaker":           "assistant",
            "raw_text":          "",
            "intent":            nlp_result["intent"],
            "entities":          "",
            "sentiment":         "",
            "response_text":     response,
            "processing_time_ms": round(processing_ms, 2),
            "tts_engine":        self.tts.engine_name,
            "stt_engine":        "",
        })

    def _get_history(self) -> List[dict]:
        return self.logger.get_history(self.session_id)

    def process(self, user_text: str) -> str:
        """Core pipeline: NLP → GenAI → Response"""
        t0 = time.time()

        # 1. NLP Analysis
        nlp = self.nlp.analyze(user_text)
        logger.info(f"NLP → intent={nlp['intent']} | sentiment={nlp['sentiment']} | entities={nlp['entities']}")

        # 2. Extract user name if introduced
        name_match = re.search(r"\b(?:my\s+name\s+is|i\s+am|call\s+me|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
                               user_text, re.IGNORECASE)
        if name_match:
            self.user_name = name_match.group(1).title()

        # 3. GenAI response
        history = self._get_history()
        response = self.ai.generate(user_text, nlp, history, self.session_id)

        # 4. Personalize
        if self.user_name and self.user_name.lower() not in response.lower():
            if nlp["intent"] in ("greeting", "farewell", "gratitude"):
                response = f"{self.user_name}, {response[0].lower()}{response[1:]}"

        processing_ms = (time.time() - t0) * 1000
        self._log_turn(user_text, response, nlp, processing_ms)
        logger.info(f"Response generated in {processing_ms:.0f}ms")
        return response

    def voice_turn(self) -> bool:
        """One full voice interaction turn. Returns False to stop."""
        text = self.stt.listen()
        if text is None:
            self.tts.speak("I didn't catch that. Please try again.")
            return True

        # Check for exit commands
        if re.search(r"\b(exit|quit|goodbye|bye|stop|shut\s+down)\b", text, re.IGNORECASE):
            stats = self.logger.get_stats()
            farewell = (f"Goodbye! We had {stats['user_turns']} exchanges this session. "
                        "All conversations have been saved. Have a great day!")
            self.tts.speak(farewell)
            return False

        response = self.process(text)
        self.tts.speak(response)
        return True

    def text_turn(self, user_input: str) -> bool:
        """One text-based interaction turn. Returns False to stop."""
        if re.search(r"\b(exit|quit|goodbye|bye|stop)\b", user_input, re.IGNORECASE):
            stats = self.logger.get_stats()
            farewell = (f"Goodbye! We had {stats['user_turns']} exchanges. "
                        "All conversations saved to conversations.csv. Have a great day!")
            self.tts.speak(farewell)
            return False

        response = self.process(user_input)
        self.tts.speak(response)
        return True

    def run_interactive(self):
        """Auto-detect best input mode and run the assistant loop."""
        self.running = True
        welcome = ("Welcome to your Voice AI Assistant! "
                   f"Running in {'voice' if self.stt.engine != 'none' else 'text'} mode. "
                   "Say or type 'exit' to quit. How can I help you today?")
        print("\n" + "═"*60)
        print("🤖  VOICE AGENTIC AI ASSISTANT")
        print("═"*60)
        print(f"  Session ID : {self.session_id}")
        print(f"  STT Engine : {self.stt.engine}")
        print(f"  TTS Engine : {self.tts.engine_name}")
        print(f"  AI Engine  : {self.ai.provider}")
        print(f"  CSV Log    : {CSV_FILE}")
        print("═"*60)
        self.tts.speak(welcome)

        use_voice = self.stt.engine != "none"

        while self.running:
            try:
                if use_voice:
                    self.running = self.voice_turn()
                else:
                    user_input = input("\n💬  You: ").strip()
                    if not user_input:
                        continue
                    self.running = self.text_turn(user_input)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.tts.speak("I encountered an error. Please try again.")

        # Final stats
        stats = self.logger.get_stats()
        print("\n" + "═"*60)
        print("📊  SESSION STATS")
        print("═"*60)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("═"*60)
        print(f"✅  Conversations saved to: {CSV_FILE}")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run_interactive()