
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
import time
import os
import tempfile
import json

from dotenv import load_dotenv
load_dotenv()
from livekit.api import AccessToken, VideoGrants


# 🔹 load agent module once; invoke compiled graph via agent.app
import openaibot as agent

# 🔹 voice I/O
from voice_io import transcribe_audio, speak_to_file, speak_stream


app = Flask(__name__)

# CORS support for frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


# -------------------------------
# 💬 TEXT CHAT
# -------------------------------
# -------------------------------
# 💬 TEXT CHAT
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}

        # ✅ Accept multiple frontend payload shapes
        question = (
            data.get("question")
            or data.get("message")
            or data.get("input")
            or ""
        ).strip()

        thread_id = data.get("thread_id") or f"thread_{int(time.time())}"
        stream = bool(data.get("stream", False))

        if not question:
            return jsonify({
                "answer": "Please ask a more specific question.",
                "sources": [],
                "thread_id": thread_id,
                "in_scope": False,
            }), 200

        # 🚨 Invoke the already-compiled LangGraph app once per request
        initial_state = {
            "question": question,
            "context_chunks": [],
            "generated_answer": None,
            "thread_id": thread_id,
            "destination": ""
        }
        result = agent.app.invoke(initial_state)
        ga = result.get("generated_answer")

        # If graph didn't produce an answer object
        if not ga:
            return jsonify({
                "answer": "I do not have enough information.",
                "sources": [],
                "thread_id": thread_id,
                "in_scope": False,
            }), 200

        # Normalize answer text safely
        answer_text = (getattr(ga, "answer", "") or "").strip()
        sources_used = getattr(ga, "sources_used", None) or []

        # Decide in_scope based on your current RAG behavior
        # (You can tighten this later with confidence scores.)
        lowered = answer_text.lower()
        in_scope = bool(answer_text) and lowered not in {
            "i do not have enough information.",
            "i do not have enough information",
        }

        # -----------------------------------
        # 🔁 Optional response streaming (UI only)
        # -----------------------------------
        if stream:
            def generate():
                # Stream the answer as display chunks (same as before)
                for chunk in answer_text.split(" "):
                    yield f"data: {json.dumps({'type': 'chunk', 'data': chunk + ' '})}\n\n"
                    time.sleep(0.01)

                # Include sources/thread_id as before
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_used})}\n\n"
                yield f"data: {json.dumps({'type': 'thread_id', 'data': thread_id})}\n\n"

                # ✅ Also include in_scope (harmless for your existing UI)
                yield f"data: {json.dumps({'type': 'in_scope', 'data': in_scope})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream"
            )

        # -----------------------------------
        # 🧾 Normal JSON response
        # -----------------------------------
        return jsonify({
            "answer": answer_text,
            "sources": sources_used,
            "thread_id": thread_id,
            "in_scope": in_scope,
        }), 200

    except Exception as e:
        print("🔥 /chat ERROR:", repr(e))
        return jsonify({
            "answer": "Internal server error.",
            "sources": [],
            "thread_id": None,
            "in_scope": False,
        }), 500



# -------------------------------
# 🎙️ SPEECH → TEXT (ASR)
# -------------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    import logging
    logger = logging.getLogger(__name__)

    audio_path = None

    try:
        logger.info("=== TRANSCRIBE REQUEST RECEIVED ===")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")

        # --------------------------------------------------
        # Case 1: multipart/form-data (audio in request.files)
        # --------------------------------------------------
        if "audio" in request.files:
            audio_file = request.files["audio"]
            content_type = audio_file.content_type or ""
            logger.info(f"Multipart upload detected: {content_type}")

            # Decide extension
            if "wav" in content_type:
                suffix = ".wav"
            elif "webm" in content_type:
                suffix = ".webm"
            elif "mpeg" in content_type or "mp3" in content_type:
                suffix = ".mp3"
            else:
                suffix = ".wav"  # safe default

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                audio_file.save(tmp.name)
                audio_path = tmp.name

        # --------------------------------------------------
        # Case 2: raw audio body (Blob sent directly)
        # --------------------------------------------------
        elif request.data and len(request.data) > 0:
            content_type = request.content_type or "audio/wav"
            logger.info(f"Raw audio body detected: {content_type}")

            if "wav" in content_type:
                suffix = ".wav"
            elif "webm" in content_type:
                suffix = ".webm"
            elif "mpeg" in content_type or "mp3" in content_type:
                suffix = ".mp3"
            else:
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(request.data)
                audio_path = tmp.name

        # --------------------------------------------------
        # No audio found
        # --------------------------------------------------
        else:
            logger.error("No audio data received in request")
            return jsonify({"error": "audio data required"}), 400

        logger.info(f"Saved audio file to: {audio_path}")

        # --------------------------------------------------
        # Transcription (OpenAI ASR via voice_io)
        # --------------------------------------------------
        text = transcribe_audio(audio_path)

        logger.info(f"Transcription successful ({len(text)} characters)")
        return jsonify({"text": text}), 200

    except Exception as e:
        logger.exception("🔥 Transcription failed")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug(f"Cleaned up temp file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {audio_path}: {e}")



# -------------------------------
# 🔊 TEXT → SPEECH (TTS)
# -------------------------------
@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    voice = data.get("voice", "alloy")

    if not text:
        return jsonify({"error": "text is required"}), 400

    # Create temp file (DO NOT delete immediately on Windows)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = tmp.name
    tmp.close()

    try:
        speak_to_file(text, output_path, voice=voice)

        # IMPORTANT: send_file first, cleanup later
        response = send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="speech.wav"
        )

        return response

    except Exception as e:
        print("🔥 /speak ERROR:", repr(e))
        return jsonify({"error": "TTS failed"}), 500


@app.route("/speak_stream", methods=["POST"])
def speak_stream_route():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    voice = data.get("voice", "alloy")

    if not text:
        return jsonify({"error": "text is required"}), 400

    return Response(
        stream_with_context(speak_stream(text, voice)),
        mimetype="audio/mpeg"

    )

@app.route("/livekit-token", methods=["GET"])
def livekit_token():
    identity = request.args.get("identity", f"browser-{int(time.time())}")
    room_name = request.args.get("room", "chat")

    token = (
        AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_identity(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    return {"token": token, "identity": identity, "room": room_name}


@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    thread_id = data.get("thread_id") or f"thread_{int(time.time())}"

    if not text:
        return jsonify({"error": "text required"}), 400

    def generate():
        initial_state = {
            "question": text,
            "context_chunks": [],
            "generated_answer": None,
            "thread_id": thread_id,
            "destination": ""
        }

        result = agent.app.invoke(initial_state)
        ga = result.get("generated_answer")

        if not ga:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Chunk for TTS friendliness
        words = ga.answer.split(" ")
        buffer = ""

        for w in words:
            buffer += w + " "
            if len(buffer) > 40:
                yield f"data: {json.dumps({'type': 'chunk', 'data': buffer.strip()})}\n\n"
                buffer = ""

        if buffer:
            yield f"data: {json.dumps({'type': 'chunk', 'data': buffer.strip()})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream"
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
