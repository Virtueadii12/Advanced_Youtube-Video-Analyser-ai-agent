import whisper
import yt_dlp
import tempfile

# LOAD MODEL
model = whisper.load_model("tiny")

# ======================================================
# TRANSCRIBE YOUTUBE VIDEO
# ======================================================

def transcribe_youtube_video(video_url):

    temp_dir = tempfile.mkdtemp()

    audio_path = f"{temp_dir}/audio.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_path,
        "quiet": True,
    }

    # DOWNLOAD AUDIO
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # TRANSCRIBE
    result = model.transcribe(audio_path)

    return result["text"]