

import os
import re
import csv
import json
import time
import datetime
import tempfile
import threading
import pandas as pd
import streamlit as st
from pathlib import Path
from collections import Counter, defaultdict

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3a;
    --accent: #7c6af7;
    --accent2: #f76a8c;
    --accent3: #6af7c8;
    --text: #e8e8f0;
    --muted: #7070a0;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7, #f76a8c, #6af7c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

.subtitle {
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin-top: 4px;
    margin-bottom: 2rem;
}

.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-val { font-size: 2.2rem; font-weight: 800; color: var(--accent); }
.kpi-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 4px; font-family: 'Space Mono', monospace; }

.chat-bubble-user {
    background: linear-gradient(135deg, #1e1e30, #2a1a3a);
    border: 1px solid var(--accent);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}
.chat-bubble-ai {
    background: linear-gradient(135deg, #1a1a26, #1a2a26);
    border: 1px solid var(--accent3);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
}

.intent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    background: rgba(124, 106, 247, 0.2);
    border: 1px solid var(--accent);
    color: var(--accent);
    margin: 2px;
}

.agent-action {
    background: rgba(106, 247, 200, 0.1);
    border-left: 3px solid var(--accent3);
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin: 4px 0;
    color: var(--accent3);
}

.status-online {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6af7c8;
    box-shadow: 0 0 8px #6af7c8;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.stTextInput input, .stTextArea textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stSelectbox select, div[data-baseweb="select"] {
    background: var(--card) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
CSV_FILE = "conversations.csv"
CSV_HEADERS = [
    "timestamp", "session_id", "turn_id", "speaker",
    "raw_text", "intent", "entities", "sentiment",
    "response_text", "processing_time_ms", "tts_engine", "stt_engine",
    "agent_action", "agent_result"
]

# ── NLP Patterns ──────────────────────────────────────────────────
INTENT_PATTERNS = [
    (r"\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b", "greeting"),
    (r"\b(bye|goodbye|see\s+you|farewell|exit|quit)\b", "farewell"),
    (r"\b(what\s+(time|date|day)|current\s+time|today)\b", "datetime_query"),
    (r"\b(weather|temperature|forecast)\b", "weather_query"),
    (r"\b(help|assist|how\s+to)\b", "help_request"),
    (r"\b(\d+\s*[\+\-\*\/]\s*\d+|calculate|compute|math)\b", "math_query"),
    (r"\b(remind|reminder|alarm|schedule)\b", "reminder"),
    (r"\b(save|store|note|remember\s+this)\b", "save_note"),
    (r"\b(show|list|display|get)\s+(task|note|reminder|todo)\b", "show_tasks"),
    (r"\b(summarize|summary|brief|overview)\b", "summarize"),
    (r"\b(search|find|look\s+up)\b", "search_query"),
    (r"\b(joke|funny|laugh|humor)\b", "entertainment"),
    (r"\b(news|latest|update|headline)\b", "news_query"),
    (r"\b(thank|thanks)\b", "gratitude"),
]

ENTITY_PATTERNS = {
    "email":    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "phone":    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "date":     r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
    "time":     r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?\b",
    "number":   r"\b\d+(?:\.\d+)?\b",
    "math_expr":r"\b\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?\b",
    "url":      r"https?://\S+",
}

# ── Session State Init ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.datetime.now().strftime("sess_%Y%m%d_%H%M%S")
if "turn_id" not in st.session_state:
    st.session_state.turn_id = 0
if "notes" not in st.session_state:
    st.session_state.notes = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "offline"

# ── NLP Functions ─────────────────────────────────────────────────
def detect_intent(text):
    for pattern, intent in INTENT_PATTERNS:
        if re.search(pattern, text.lower(), re.IGNORECASE):
            return intent
    return "general_query"

def extract_entities(text):
    entities = {}
    for etype, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities[etype] = matches
    return entities

def detect_sentiment(text):
    pos = len(re.findall(r"\b(great|good|happy|love|excellent|amazing|wonderful|fantastic)\b", text.lower()))
    neg = len(re.findall(r"\b(bad|terrible|hate|awful|horrible|worst|poor|frustrating)\b", text.lower()))
    return "positive" if pos > neg else "negative" if neg > pos else "neutral"

def eval_math(text):
    expr = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)", text)
    if expr:
        a, op, b = float(expr.group(1)), expr.group(2), float(expr.group(3))
        ops = {"+": a+b, "-": a-b, "*": a*b, "/": a/b if b!=0 else None}
        result = ops.get(op)
        return f"{a} {op} {b} = {result}" if result is not None else "Division by zero!"
    return None

# ── Agentic Actions ───────────────────────────────────────────────
def handle_agentic_action(text, intent, entities):
    """Core agentic behavior — conditional routing based on intent"""
    action = None
    result = None

    if intent == "save_note":
        note_text = re.sub(r"\b(save|store|note|remember\s+this|:)\b", "", text, flags=re.IGNORECASE).strip()
        st.session_state.notes.append({
            "text": note_text,
            "timestamp": datetime.datetime.now().isoformat()
        })
        action = "SAVE_NOTE"
        result = f"Note saved: '{note_text}'"

    elif intent == "show_tasks":
        if st.session_state.tasks:
            result = "Tasks: " + ", ".join([t["text"] for t in st.session_state.tasks])
        else:
            result = "No tasks found. Say 'save task: <task>' to add one."
        action = "SHOW_TASKS"

    elif intent == "reminder":
        task_text = re.sub(r"\b(remind|reminder|alarm|schedule|me|to|about)\b", "", text, flags=re.IGNORECASE).strip()
        st.session_state.tasks.append({
            "text": task_text,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "pending"
        })
        action = "CREATE_REMINDER"
        result = f"Reminder set: '{task_text}'"

    elif intent == "summarize":
        if st.session_state.notes:
            result = f"You have {len(st.session_state.notes)} notes. Latest: '{st.session_state.notes[-1]['text']}'"
        else:
            result = "No notes to summarize yet."
        action = "SUMMARIZE"

    elif intent == "math_query":
        math_result = eval_math(text)
        if math_result:
            action = "CALCULATE"
            result = math_result

    elif intent == "datetime_query":
        action = "GET_DATETIME"
        result = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return action, result

# ── GenAI Client ──────────────────────────────────────────────────
def init_ai():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            client = genai.GenerativeModel("gemini-2.0-flash")
            st.session_state.ai_provider = "gemini"
            return client, "gemini"
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, base_url=openai_base)
            st.session_state.ai_provider = "openai"
            return client, "openai"
        except Exception:
            pass

    st.session_state.ai_provider = "offline"
    return None, "offline"

def generate_response(text, intent, entities, sentiment, agent_result, history):
    client, provider = init_ai()

    # If agentic action already has a result, use it as context
    context = f"Agentic action result: {agent_result}. " if agent_result else ""

    if provider == "offline":
        offline_responses = {
            "greeting": "Hello! I'm your Voice AI Assistant. How can I help?",
            "farewell": "Goodbye! Have a great day!",
            "datetime_query": f"Current time: {datetime.datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}",
            "entertainment": "Why don't scientists trust atoms? Because they make up everything! 😄",
            "gratitude": "You're welcome! Happy to help!",
            "math_query": agent_result or "Please give me a math expression like '5 + 3'",
            "save_note": agent_result or "Note saved!",
            "show_tasks": agent_result or "No tasks found.",
            "reminder": agent_result or "Reminder created!",
            "summarize": agent_result or "Nothing to summarize yet.",
        }
        return offline_responses.get(intent, f"I understood: '{text}'. Sentiment: {sentiment}. Set an API key for smarter responses!")

    system = (
        f"You are a concise voice AI assistant. Respond in 1-3 sentences max, spoken-friendly. "
        f"Intent: {intent}. Entities: {json.dumps(entities)}. Sentiment: {sentiment}. {context}"
    )

    msgs = [{"role": "user", "content": system},
            {"role": "assistant", "content": "Understood, I'll respond concisely."}]
    for m in history[-4:]:
        msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})
    msgs.append({"role": "user", "content": text})

    try:
        if provider == "gemini":
            flat = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
            resp = client.generate_content(flat)
            return resp.text.strip()
        elif provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=200)
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {str(e)[:80]}. Using offline mode."

# ── CSV Logger ────────────────────────────────────────────────────
def ensure_csv():
    if not Path(CSV_FILE).exists():
        with open(CSV_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

def log_to_csv(record):
    ensure_csv()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(
            {k: record.get(k, "") for k in CSV_HEADERS}
        )

def load_csv():
    ensure_csv()
    try:
        return pd.read_csv(CSV_FILE)
    except Exception:
        return pd.DataFrame(columns=CSV_HEADERS)

# ── TTS ───────────────────────────────────────────────────────────
def speak_text(text):
    try:
        from gtts import gTTS
        import pygame
        pygame.mixer.init()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        gTTS(text=text, lang="en", slow=False).save(tmp)
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        os.unlink(tmp)
        return True
    except Exception:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            return False

# ── Process Message ───────────────────────────────────────────────
def process_message(user_text, speak=False):
    t0 = time.time()
    st.session_state.turn_id += 1

    # NLP
    intent = detect_intent(user_text)
    entities = extract_entities(user_text)
    sentiment = detect_sentiment(user_text)

    # Agentic action
    action, agent_result = handle_agentic_action(user_text, intent, entities)

    # Build history for context
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[-6:]]

    # GenAI response
    response = generate_response(user_text, intent, entities, sentiment, agent_result, history)
    processing_ms = (time.time() - t0) * 1000

    # TTS
    if speak:
        threading.Thread(target=speak_text, args=(response,), daemon=True).start()

    # Log to CSV
    ts = datetime.datetime.now().isoformat()
    log_to_csv({
        "timestamp": ts, "session_id": st.session_state.session_id,
        "turn_id": st.session_state.turn_id, "speaker": "user",
        "raw_text": user_text, "intent": intent,
        "entities": json.dumps(entities), "sentiment": sentiment,
        "processing_time_ms": "", "tts_engine": "", "stt_engine": "streamlit_ui",
        "agent_action": action or "", "agent_result": agent_result or "",
    })
    log_to_csv({
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        "turn_id": st.session_state.turn_id, "speaker": "assistant",
        "raw_text": "", "intent": intent, "entities": "", "sentiment": "",
        "response_text": response, "processing_time_ms": round(processing_ms, 1),
        "tts_engine": "gtts", "stt_engine": "",
        "agent_action": action or "", "agent_result": agent_result or "",
    })

    # Update chat history
    st.session_state.messages.append({
        "role": "user", "content": user_text,
        "intent": intent, "entities": entities, "sentiment": sentiment,
        "action": action, "agent_result": agent_result,
    })
    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "processing_ms": round(processing_ms, 1),
    })

    return response, intent, entities, sentiment, action, agent_result

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 System Status")
    _, provider = init_ai()
    provider_color = {"gemini": "#6af7c8", "openai": "#6af7c8", "offline": "#f7a86a"}[provider]
    st.markdown(f'<span style="color:{provider_color}">● AI: {provider.upper()}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#6af7c8">● Session: {st.session_state.session_id[-8:]}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#6af7c8">● Turns: {st.session_state.turn_id}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Settings")
    enable_tts = st.toggle("🔊 Speak responses", value=False)
    show_debug = st.toggle("🔍 Show NLP debug", value=True)

    st.divider()
    st.markdown("### 📋 Saved Notes")
    if st.session_state.notes:
        for i, note in enumerate(st.session_state.notes[-5:]):
            st.markdown(f"**{i+1}.** {note['text']}")
    else:
        st.caption("No notes yet. Say 'save note: ...'")

    st.divider()
    st.markdown("### ✅ Tasks & Reminders")
    if st.session_state.tasks:
        for task in st.session_state.tasks[-5:]:
            st.markdown(f"⏰ {task['text']}")
    else:
        st.caption("No tasks yet. Say 'remind me to ...'")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.turn_id = 0
        st.rerun()

# ── MAIN LAYOUT ───────────────────────────────────────────────────
st.markdown('<div class="main-title">Voice Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">// Python · RegEx NLP · GenAI · Voice I/O · CSV Analytics · Agentic Workflow</div>', unsafe_allow_html=True)

# KPIs
df = load_csv()
user_rows = df[df["speaker"] == "user"] if not df.empty else pd.DataFrame()
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(df)}</div><div class="kpi-lbl">TOTAL RECORDS</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(user_rows)}</div><div class="kpi-lbl">USER TURNS</div></div>', unsafe_allow_html=True)
with col3:
    sessions = df["session_id"].nunique() if not df.empty else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{sessions}</div><div class="kpi-lbl">SESSIONS</div></div>', unsafe_allow_html=True)
with col4:
    avg_ms = round(pd.to_numeric(df["processing_time_ms"], errors="coerce").mean(), 0) if not df.empty else 0
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_ms}ms</div><div class="kpi-lbl">AVG RESPONSE</div></div>', unsafe_allow_html=True)
with col5:
    notes_count = len(st.session_state.notes)
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{notes_count}</div><div class="kpi-lbl">SAVED NOTES</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 Analytics", "📁 CSV Data"])

# ── TAB 1: CHAT ────────────────────────────────────────────────────
with tab1:
    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'''
                <div class="chat-bubble-user">
                    <strong>👤 You</strong><br>{msg["content"]}
                    {"<br><span class='intent-badge'>"+msg.get("intent","")+"</span>" if show_debug else ""}
                    {"<span class='intent-badge' style='border-color:#f76a8c;color:#f76a8c'>"+msg.get("sentiment","")+"</span>" if show_debug else ""}
                </div>''', unsafe_allow_html=True)
                if show_debug and msg.get("action"):
                    st.markdown(f'<div class="agent-action">⚡ AGENT ACTION: {msg["action"]} → {msg.get("agent_result","")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-bubble-ai">
                    <strong>🤖 Assistant</strong><br>{msg["content"]}
                    {"<br><small style='color:var(--muted)'>"+str(msg.get('processing_ms',''))+"ms</small>" if show_debug else ""}
                </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Type your message",
                placeholder="Try: 'save note: buy groceries' | 'remind me to call John' | 'show tasks' | 'what time is it'",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            process_message(user_input.strip(), speak=enable_tts)
        st.rerun()

    # Quick action buttons
    st.markdown("**Quick Actions:**")
    qcols = st.columns(6)
    quick = ["Hello!", "What time is it?", "Tell me a joke", "Show my tasks", "Save note: test", "Summarize"]
    for i, q in enumerate(quick):
        with qcols[i]:
            if st.button(q, key=f"q{i}"):
                with st.spinner("Thinking..."):
                    process_message(q, speak=enable_tts)
                st.rerun()

# ── TAB 2: ANALYTICS ──────────────────────────────────────────────
with tab2:
    df = load_csv()
    if df.empty:
        st.info("No conversation data yet. Start chatting to see analytics!")
    else:
        user_df = df[df["speaker"] == "user"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📊 Intent Distribution")
            if not user_df.empty and "intent" in user_df.columns:
                intent_counts = user_df["intent"].value_counts()
                st.bar_chart(intent_counts)

        with col_b:
            st.markdown("#### 😊 Sentiment Distribution")
            if not user_df.empty and "sentiment" in user_df.columns:
                sent_counts = user_df["sentiment"].value_counts()
                st.bar_chart(sent_counts)

        st.markdown("#### ⚡ Agentic Actions Taken")
        action_df = df[df["agent_action"] != ""]
        if not action_df.empty:
            st.dataframe(
                action_df[["timestamp", "speaker", "raw_text", "agent_action", "agent_result"]].tail(10),
                use_container_width=True
            )
        else:
            st.info("No agentic actions yet. Try 'save note:', 'remind me to...', 'show tasks'")

        st.markdown("#### 🔍 Entity Extraction Log")
        entity_df = user_df[user_df["entities"] != "{}"].copy() if not user_df.empty else pd.DataFrame()
        if not entity_df.empty:
            st.dataframe(entity_df[["timestamp", "raw_text", "entities"]].tail(10), use_container_width=True)
        else:
            st.info("No entities detected yet. Try saying an email, phone number, or math expression.")

# ── TAB 3: CSV DATA ────────────────────────────────────────────────
with tab3:
    df = load_csv()
    st.markdown(f"#### 📁 conversations.csv — {len(df)} records")

    if not df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            speaker_filter = st.selectbox("Filter by speaker", ["All", "user", "assistant"])
        with col_f2:
            intent_options = ["All"] + sorted(df["intent"].dropna().unique().tolist())
            intent_filter = st.selectbox("Filter by intent", intent_options)
        with col_f3:
            session_options = ["All"] + sorted(df["session_id"].dropna().unique().tolist())
            session_filter = st.selectbox("Filter by session", session_options)

        filtered = df.copy()
        if speaker_filter != "All":
            filtered = filtered[filtered["speaker"] == speaker_filter]
        if intent_filter != "All":
            filtered = filtered[filtered["intent"] == intent_filter]
        if session_filter != "All":
            filtered = filtered[filtered["session_id"] == session_filter]

        st.dataframe(filtered.tail(50), use_container_width=True)

        csv_data = filtered.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv_data, "conversations_export.csv", "text/csv")
    else:
        st.info("No data yet. Start a conversation!")