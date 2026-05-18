from gtts import gTTS

text = """
Welcome to the AI YouTube Video Analyzer project.
This project uses Groq, Agno, and Generative AI.
"""

tts = gTTS(text=text, lang="en")

tts.save("output.mp3")

print("Voice Generated Successfully!")