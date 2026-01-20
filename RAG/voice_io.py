# voice_io.py
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# ASR — Speech to Text
# -------------------------------
def transcribe_audio(audio_path: str) -> str:
    logger.info(f"Starting transcription for {audio_path}")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            response_format="text",
        )

    return response.strip()

#voice = "alloy"
# -------------------------------
# TTS — Text to Speech
# -------------------------------
def speak_to_file(text: str, output_path: str, voice: str = "alloy"):
    """
    Generate speech audio using OpenAI TTS and save to WAV file.
    Compatible with current OpenAI Python SDK (binary streaming).
    """
    try:
        logger.info("Generating TTS audio")

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
        ) as response:
            response.stream_to_file(output_path)

        logger.info("TTS audio written successfully")

    except Exception as e:
        logger.error("TTS generation failed", exc_info=True)
        raise


def speak_stream(text: str, voice: str = "alloy"):
    logger.info("Streaming TTS audio (MP3)")

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        for chunk in response.iter_bytes():
            yield chunk
