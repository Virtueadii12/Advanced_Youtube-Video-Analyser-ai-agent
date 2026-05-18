# =========================================================
# AI YOUTUBE ANALYZER PRO
# FULL FINAL UI.PY
# =========================================================

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from textwrap import dedent
import whisper

from whisper_utils import transcribe_youtube_video

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.youtube import YouTubeTools

from rag_chat import (
    create_vector_db,
    get_relevant_context
)

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI YouTube Analyzer PRO",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

whisper_model = load_whisper()

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_content" not in st.session_state:
    st.session_state.analysis_content = ""

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* =========================================================
BACKGROUND
========================================================= */

.stApp {

    background:
    radial-gradient(circle at top left, rgba(255,0,128,0.15), transparent 25%),
    radial-gradient(circle at top right, rgba(0,102,255,0.15), transparent 25%),
    radial-gradient(circle at bottom left, rgba(255,153,0,0.12), transparent 25%),
    #030712;

    color: white;
}

/* =========================================================
HIDE DEFAULT
========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* =========================================================
SIDEBAR FIX
========================================================= */

[data-testid="stSidebar"] {

    display: block !important;

    background:
    linear-gradient(
        180deg,
        rgba(15,23,42,0.98),
        rgba(2,6,23,1)
    );

    border-right: 1px solid rgba(255,255,255,0.06);

    width: 320px !important;
}

/* SIDEBAR BUTTON */

[data-testid="collapsedControl"] {

    display: flex !important;
    visibility: visible !important;

    position: fixed;

    top: 14px;
    left: 14px;

    z-index: 999999;

    background: rgba(255,255,255,0.08);

    border-radius: 12px;

    padding: 6px;
}

/* =========================================================
MAIN
========================================================= */

.block-container {

    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* =========================================================
TITLE
========================================================= */

.main-title {

    font-size: 72px;
    font-weight: 800;
    text-align: center;

    background: linear-gradient(
        90deg,
        #ff4d4d,
        #ff7b00,
        #ff00cc,
        #7d5fff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow:
    0px 0px 25px rgba(255,0,200,0.35);

    margin-bottom: 10px;
}

.subtitle {

    text-align: center;
    font-size: 22px;
    color: #d1d5db;

    margin-bottom: 40px;
}

/* =========================================================
GLASS CARD
========================================================= */

.glass {

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 30px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.4),
    0 0 20px rgba(255,0,150,0.08);

    margin-bottom: 30px;
}

/* =========================================================
INPUT
========================================================= */

.stTextInput > div > div > input {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 18px;

    padding: 16px;

    color: white;

    font-size: 18px;
}

/* =========================================================
BUTTON
========================================================= */

.stButton > button {

    background: linear-gradient(
        90deg,
        #ff4d4d,
        #ff7b00,
        #ff00cc
    );

    color: white;

    border: none;

    border-radius: 18px;

    padding: 16px 24px;

    font-size: 18px;

    font-weight: 700;

    width: 100%;

    margin-top: 10px;

    box-shadow:
    0 10px 25px rgba(255,0,120,0.35);

    transition: 0.3s ease-in-out;
}

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
    0 15px 35px rgba(255,0,120,0.5);
}

/* =========================================================
SIDEBAR BOX
========================================================= */

.sidebar-box {

    background: rgba(255,255,255,0.04);

    border-radius: 20px;

    padding: 18px;

    margin-bottom: 20px;

    border: 1px solid rgba(255,255,255,0.06);

    color: white;
}

/* =========================================================
VOICE BOX
========================================================= */

.voice-box {

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 35px;

    margin-bottom: 30px;
}

/* =========================================================
CHAT
========================================================= */

.stChatMessage {

    background: rgba(255,255,255,0.04);

    border-radius: 18px;

    padding: 14px;

    margin-bottom: 12px;
}

/* =========================================================
FOOTER
========================================================= */

.footer {

    text-align: center;

    color: #9ca3af;

    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
🎥 AI YouTube Analyzer PRO
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Analyze YouTube Videos using Groq + Whisper + RAG + GenAI 🚀
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-box">
    <h2>🚀 FEATURES</h2>

    ✅ AI Video Analysis <br><br>
    ✅ Whisper Transcription <br><br>
    ✅ Flashcards <br><br>
    ✅ MCQs <br><br>
    ✅ AI Memory <br><br>
    ✅ Voice Chat <br><br>
    ✅ Premium UI <br><br>
    ✅ RAG Chatbot
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-box">
    <h2>📊 ANALYTICS</h2>

    Videos Analyzed: 1+ <br><br>
    Chat Messages: {len(st.session_state.messages)} <br><br>
    AI Status: Online ✅
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()

# =========================================================
# BUILD AGENT
# =========================================================

@st.cache_resource
def build_agent():

    return Agent(

        name="YouTube AI Analyzer",

        model=Groq(
            id="llama-3.3-70b-versatile"
        ),

        tools=[YouTubeTools()],

        instructions=dedent("""
        You are an elite AI YouTube Analyzer.

        Generate:
        - Summary
        - Timestamps
        - Flashcards
        - MCQs
        - Interview Questions
        - Revision Notes
        """),

        markdown=True,
    )

youtube_agent = build_agent()

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

video_url = st.text_input(
    "🔗 Paste YouTube Video URL",
    placeholder="https://youtube.com/watch?v=..."
)

# BUTTONS BELOW URL

col1, col2 = st.columns(2)

with col1:
    test_whisper = st.button(
        "🎤 Test Whisper",
        use_container_width=True
    )

with col2:
    analyze = st.button(
        "🚀 Analyze Video",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# VOICE ASSISTANT
# =========================================================

st.markdown("""
<div class="voice-box">

<h1>🎙 AI Voice Assistant</h1>

<p style="font-size:20px; color:#d1d5db;">
Talk directly with your AI Analyzer using Whisper AI
</p>

<br>

<div style="
background: rgba(255,255,255,0.03);
padding: 28px;
border-radius: 22px;
border: 1px solid rgba(255,255,255,0.06);
">

<h2>✨ Features</h2>

<br>

<div style="display:flex; justify-content:space-between;">

<div style="font-size:18px;">
✅ Real-time Voice Input <br><br>
✅ Smart AI Responses
</div>

<div style="font-size:18px;">
✅ Whisper AI Recognition <br><br>
✅ Context Memory
</div>

</div>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TEST WHISPER
# =========================================================

if test_whisper:

    if not video_url:

        st.warning("⚠️ Please enter YouTube URL")

    else:

        with st.spinner("🎙 Testing Whisper AI..."):

            try:

                transcript = transcribe_youtube_video(
                    video_url
                )

                st.success("✅ Whisper Working Successfully!")

                st.text_area(
                    "Transcript Preview",
                    transcript[:3000],
                    height=300
                )

            except Exception as e:

                st.error(f"❌ Whisper Error: {str(e)}")

# =========================================================
# ANALYZE VIDEO
# =========================================================

if analyze:

    if not video_url:

        st.warning("⚠️ Please enter YouTube URL")

    else:

        with st.spinner("🚀 AI analyzing video..."):

            try:

                analysis_prompt = f"""
                Analyze this YouTube video deeply.

                VIDEO URL:
                {video_url}

                Generate:
                - Summary
                - Timestamps
                - Key Learnings
                - Flashcards
                - MCQs
                - Interview Questions
                """

                response = youtube_agent.run(
                    analysis_prompt
                )

                st.session_state.analysis_content = response.content

                vector_db = create_vector_db(
                    response.content
                )

                st.session_state.vector_db = vector_db

                st.session_state.analysis_done = True

                st.success("✅ Analysis Completed!")

                st.balloons()

            except Exception as e:

                st.error(f"❌ Error: {str(e)}")

# =========================================================
# SHOW ANALYSIS
# =========================================================

if st.session_state.analysis_done:

    st.markdown("## 📚 Video Analysis")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown(
        st.session_state.analysis_content
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CHATBOT
# =========================================================

if st.session_state.analysis_done:

    st.markdown("## 💬 Chat With Video")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask anything about the video..."
    )

    if user_question:

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                try:

                    context = get_relevant_context(
                        st.session_state.vector_db,
                        user_question
                    )

                    chat_prompt = f"""
                    CONTEXT:
                    {context}

                    QUESTION:
                    {user_question}

                    Answer using context only.
                    """

                    answer = youtube_agent.run(
                        chat_prompt
                    )

                    st.markdown(answer.content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer.content
                    })

                except Exception as e:

                    st.error(f"❌ Chat Error: {str(e)}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="
text-align:center;
margin-top:80px;
padding-bottom:40px;
font-size:22px;
font-weight:700;
display:flex;
justify-content:center;
align-items:center;
gap:10px;
">

<span style="
background: linear-gradient(90deg,#ff4d4d,#ff00cc,#7d5fff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
Built with
</span>

<span style="font-size:24px;">😊</span>

<span style="
background: linear-gradient(90deg,#ff4d4d,#ff00cc,#7d5fff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
by Aditya Singh
</span>

<span style="font-size:24px;">🚀</span>

</div>
""", unsafe_allow_html=True)