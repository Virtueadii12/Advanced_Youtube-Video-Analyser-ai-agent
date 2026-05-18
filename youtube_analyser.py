from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.youtube import YouTubeTools

from rag_chat import (
    create_vector_db,
    get_relevant_context
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# BUILD AI AGENT
# =========================================================

def build_youtube_agent():

    return Agent(

        name="Advanced YouTube AI Analyzer",

        # LLM MODEL
        model=Groq(
            id="llama-3.3-70b-versatile"
        ),

        # TOOLS
        tools=[YouTubeTools()],

        # SYSTEM PROMPT
        instructions=dedent("""
        You are an Elite AI-Powered YouTube Video Analysis Assistant. 🎥🤖

        Your goal is to deeply analyze YouTube videos and generate:
        - Video summaries
        - Timestamps
        - Key learnings
        - Technical concepts
        - Flashcards
        - MCQs
        - Interview questions
        - Revision notes
        - Actionable insights

        RULES:
        - NEVER hallucinate timestamps
        - ONLY use transcript/context available
        - Keep formatting professional
        - Use markdown formatting
        - Be educational and beginner friendly
        - Focus on technical accuracy
        """),

        markdown=True,
        add_datetime_to_context=True,
    )

# =========================================================
# CREATE AGENT
# =========================================================

youtube_agent = build_youtube_agent()

# =========================================================
# MAIN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🎥 ADVANCED YOUTUBE AI ANALYZER + RAG CHATBOT")
    print("=" * 60)

    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------

    video_url = input("\n🔗 Enter YouTube Video URL: ")

    print("\n⚡ Analyzing Video...\n")

    # -----------------------------------------------------
    # ANALYZE VIDEO
    # -----------------------------------------------------

    analysis_prompt = f"""
    Analyze this YouTube video in complete detail:

    Video URL:
    {video_url}

    Generate:
    - Video overview
    - Accurate timestamps
    - Topic breakdown
    - Key learnings
    - Technical concepts
    - Flashcards
    - MCQs
    - Interview questions
    - Revision notes
    - Final summary

    Make the output:
    - Highly detailed
    - Educational
    - Well-structured
    - Beginner-friendly
    """

    # Generate analysis
    response = youtube_agent.run(analysis_prompt)

    # Print analysis
    print(response.content)

    # -----------------------------------------------------
    # CREATE VECTOR DATABASE
    # -----------------------------------------------------

    print("\n🧠 Creating RAG Knowledge Base...\n")

    transcript_text = response.content

    vector_db = create_vector_db(transcript_text)

    print("✅ RAG Chatbot Ready!")
    print("💬 Ask questions about the video.")
    print("❌ Type 'exit' to quit.\n")

    # -----------------------------------------------------
    # CHAT LOOP
    # -----------------------------------------------------

    while True:

        user_question = input("🧑 You: ")

        # Exit condition
        if user_question.lower() == "exit":

            print("\n👋 Exiting Chatbot...")
            break

        # ---------------------------------------------
        # RETRIEVE RELEVANT CONTEXT
        # ---------------------------------------------

        context = get_relevant_context(
            vector_db,
            user_question
        )

        # ---------------------------------------------
        # GENERATE ANSWER
        # ---------------------------------------------

        chat_prompt = f"""
        Answer the user's question using ONLY the provided context.

        =========================
        CONTEXT
        =========================

        {context}

        =========================
        QUESTION
        =========================

        {user_question}

        =========================
        INSTRUCTIONS
        =========================

        - Be accurate
        - Be concise
        - Use simple explanations
        - Do not hallucinate
        - If answer is unavailable,
          clearly mention it
        """

        answer = youtube_agent.run(chat_prompt)

        # ---------------------------------------------
        # PRINT ANSWER
        # ---------------------------------------------

        print(f"\n🤖 AI: {answer.content}\n")