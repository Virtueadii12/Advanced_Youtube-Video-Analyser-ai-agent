from yt_dlp import YoutubeDL
import whisper
import os
import uuid

model = whisper.load_model("tiny")


def transcribe_youtube_video(video_url):

    filename = f"{uuid.uuid4()}.mp3"

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": filename,
        "quiet": True,
        "noplaylist": True,
        "extractaudio": True,
        "audioformat": "mp3",

        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },

        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    result = model.transcribe(filename)

    if os.path.exists(filename):
        os.remove(filename)

    return result["text"]