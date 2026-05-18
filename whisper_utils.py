from yt_dlp import YoutubeDL
import whisper
import os

model = whisper.load_model("tiny")


def transcribe_youtube_video(video_url):

    output_file = "audio.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_file,
        "quiet": True,
        "noplaylist": True,

        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    result = model.transcribe(output_file)

    if os.path.exists(output_file):
        os.remove(output_file)

    return result["text"]