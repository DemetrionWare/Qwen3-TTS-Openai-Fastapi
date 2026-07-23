# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
)
from ..services.text_processing import normalize_text

ULAW_CHUNK_BYTES = 1600     # 200ms per yield — amortizes ASGI overhead without big bursts
ULAW_CHUNK_INTERVAL = 0.200 # seconds


async def pace_ulaw_stream(audio_generator):
    """Real-time pacing for ulaw_8000 streaming (200ms chunks at 200ms intervals).

    Delivers audio at exactly real-time rate so the agent's Telnyx pipeline
    never receives a burst larger than 200ms. Without pacing, the burst caused
    the first frames to be sent to Telnyx faster than real-time, compressing
    the beginning of each utterance.

    Model RTF ~0.53 means it generates 1.9x faster than real-time, so the
    concurrent collector always stays ahead — no underruns.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def _collect():
        async for chunk in audio_generator:
            if chunk:
                await queue.put(chunk)
        await queue.put(None)

    collector = asyncio.create_task(_collect())
    buffer = bytearray()
    next_send = None

    try:
        while True:
            chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            if chunk is None:
                break
            buffer.extend(chunk)

            if next_send is None and len(buffer) >= ULAW_CHUNK_BYTES:
                next_send = loop.time()

            if next_send is not None:
                while len(buffer) >= ULAW_CHUNK_BYTES:
                    delay = next_send - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    yield bytes(buffer[:ULAW_CHUNK_BYTES])
                    del buffer[:ULAW_CHUNK_BYTES]
                    next_send += ULAW_CHUNK_INTERVAL
    finally:
        collector.cancel()

    if next_send is None:
        next_send = loop.time()
    while len(buffer) >= ULAW_CHUNK_BYTES:
        delay = next_send - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        yield bytes(buffer[:ULAW_CHUNK_BYTES])
        del buffer[:ULAW_CHUNK_BYTES]
        next_send += ULAW_CHUNK_INTERVAL
    if buffer:
        yield bytes(buffer)
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# GPU lock: serializes TTS generation to prevent GPU contention
# (pattern from groxaxo vllm_omni backend)
_gpu_lock = asyncio.Lock()

# API key for the WebSocket stream-input endpoint. The Caddy proxy in front of
# this server already enforces auth on the HTTP path; the WS endpoint enforces
# it here too so a direct (non-proxied) connection still requires a credential.
# Accepts either the ElevenLabs-style `xi-api-key` header or `Authorization:
# Bearer <token>`. If no key is configured in the environment, any non-empty
# credential is accepted (the proxy is then the source of truth).
EXPECTED_API_KEY = os.environ.get("TTS_API_KEY") or os.environ.get("API_KEY")

# Voice library directory (same as VOICE_LIBRARY_DIR in start_server.sh)
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Cache for reference audio reads (profile_name -> (audio, sr))
_ref_audio_cache = {}


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library.

    Returns dict with keys: ref_audio_path, ref_text, x_vector_only_mode, language.
    Raises ValueError if not found.
    """
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")

    # Search all profiles
    for child in profiles_dir.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Match by profile_id or name (case-insensitive)
        if meta.get("profile_id") == name_or_id or \
           meta.get("name", "").lower() == name_or_id.lower():
            # Found it -- build result
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }

    raise ValueError(f"Voice profile not found: '{name_or_id}'")


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
# Must map to voices that actually exist in the model:
# aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Aiden",
    "onyx": "Eric",
    "shimmer": "Dylan",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """Extract language from model name if it has a language suffix."""
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        await initialize_backend()

    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate speech from text using the configured TTS backend."""
    backend = await get_tts_backend()
    voice_name = get_voice_name(voice)

    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """OpenAI-compatible endpoint for text-to-speech."""
    logger.info(f"TTS request: model={request.model}, voice={request.voice}, format={request.response_format}, len={len(request.input)}")

    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )

    try:
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")

        # Voice profile: voice name starts with "clone:" -> load from voice library
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[6:].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": "clone: prefix requires a profile name, e.g. 'clone:MyVoice'",
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as e:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "profile_not_found",
                        "message": str(e),
                        "type": "invalid_request_error",
                    },
                )

            backend = await get_tts_backend()

            # Cache reference audio reads per profile
            ref_audio_path = profile["ref_audio_path"]
            if profile_name in _ref_audio_cache:
                ref_audio, ref_sr = _ref_audio_cache[profile_name]
            else:
                ref_audio, ref_sr = sf.read(ref_audio_path)
                if len(ref_audio.shape) > 1:
                    ref_audio = ref_audio.mean(axis=1)
                ref_audio = ref_audio.astype(np.float32)
                _ref_audio_cache[profile_name] = (ref_audio, ref_sr)
                logger.info(f"Reference audio cached: '{profile_name}'")

            clone_lang = language if language != "Auto" else profile["language"]
            logger.info(f"Voice profile '{profile['name']}': lang={clone_lang}, xvec_only={profile['x_vector_only_mode']}, stream={request.stream}")

            if request.stream:
                # True streaming: yield PCM chunks as model generates them
                fmt = request.response_format
                if fmt == "wav":
                    fmt = "pcm"
                content_type = get_content_type(fmt)

                async def clone_audio_stream():
                    gen_start = time.time()
                    first_chunk_time = None
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                        text=normalized_text,
                        ref_audio=ref_audio,
                        ref_audio_sr=ref_sr,
                        ref_text=profile["ref_text"] or None,
                        language=clone_lang,
                        x_vector_only_mode=profile["x_vector_only_mode"],
                        cache_key=profile_name,
                    ):
                        if pcm_chunk is not None and len(pcm_chunk) > 0:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - gen_start
                            total_samples += len(pcm_chunk)
                            sample_rate = sr
                            chunk_count += 1
                            yield encode_audio(pcm_chunk, fmt, sr)
                            await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(f"Voice clone stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

                return StreamingResponse(
                    clone_audio_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )
            else:
                # Non-streaming: generate all audio, then send (better RTF)
                gen_start = time.time()
                audio, sample_rate = await backend.generate_voice_clone(
                    text=normalized_text,
                    ref_audio=ref_audio,
                    ref_audio_sr=ref_sr,
                    ref_text=profile["ref_text"] or None,
                    language=clone_lang,
                    x_vector_only_mode=profile["x_vector_only_mode"],
                    speed=request.speed,
                    cache_key=profile_name,
                )
                gen_time = time.time() - gen_start

                audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

                fmt = request.response_format
                if fmt == "wav":
                    fmt = "mp3"
                audio_bytes = encode_audio(audio, fmt, sample_rate)
                content_type = get_content_type(fmt)

                async def send_audio():
                    for i in range(0, len(audio_bytes), 4096):
                        yield audio_bytes[i:i + 4096]

                return StreamingResponse(
                    send_audio(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # Voice cloning: task_type=Base with ref_audio (Voice Studio)
        if request.task_type == "Base" and request.ref_audio:
            backend = await get_tts_backend()

            # Use voice name as cache key so repeated calls for the same voice
            # reuse the cached voice prompt (skips ~1s create_voice_clone_prompt).
            http_cache_key = request.voice if request.voice else None

            # Decode and cache the reference audio array per voice.
            if http_cache_key and http_cache_key in _ref_audio_cache:
                ref_audio, ref_sr = _ref_audio_cache[http_cache_key]
            else:
                ref_audio_data = request.ref_audio
                if ref_audio_data.startswith("data:"):
                    ref_audio_data = ref_audio_data.split(",", 1)[1]
                audio_bytes_raw = base64.b64decode(ref_audio_data)
                audio_buffer = io.BytesIO(audio_bytes_raw)
                ref_audio, ref_sr = sf.read(audio_buffer)
                if len(ref_audio.shape) > 1:
                    ref_audio = ref_audio.mean(axis=1)
                ref_audio = ref_audio.astype(np.float32)
                if http_cache_key:
                    _ref_audio_cache[http_cache_key] = (ref_audio, ref_sr)

            logger.info(f"Voice clone: lang={language}, ref_text={request.ref_text is not None}, xvec_only={request.x_vector_only_mode}")

            gen_start = time.time()
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_sr,
                ref_text=request.ref_text,
                language=language,
                x_vector_only_mode=request.x_vector_only_mode or False,
                speed=request.speed,
                cache_key=http_cache_key,
            )
            gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"Voice clone: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

        if request.stream:
            backend = await get_tts_backend()
            voice_name = get_voice_name(request.voice)
            fmt = request.response_format
            if fmt == "wav":
                fmt = "pcm"
            content_type = get_content_type(fmt)

            async def audio_stream():
                gen_start = time.time()
                first_chunk_time = None
                total_samples = 0
                chunk_count = 0
                sample_rate = 24000
                async for pcm_chunk, sr in backend.generate_speech_streaming(
                    text=normalized_text,
                    voice=voice_name,
                    language=language,
                    instruct=request.instruct,
                    speed=request.speed,
                    model=request.model,
                ):
                    if pcm_chunk is not None and len(pcm_chunk) > 0:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - gen_start
                        total_samples += len(pcm_chunk)
                        sample_rate = sr
                        chunk_count += 1
                        yield encode_audio(pcm_chunk, fmt, sr)
                        await asyncio.sleep(0)
                gen_time = time.time() - gen_start
                audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(f"TTS stream: First-Byte={first_chunk_time:.2f}s Gesamt={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x Chunks={chunk_count}")

            stream = audio_stream()
            if fmt == "ulaw_8000":
                stream = pace_ulaw_stream(stream)

            return StreamingResponse(
                stream,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{fmt}",
                    "Cache-Control": "no-cache",
                },
            )
        else:
            # Non-streaming: generate in thread pool to keep event loop free
            # GPU lock serializes access (pattern from groxaxo vllm_omni backend)
            async with _gpu_lock:
                gen_start = time.time()
                loop = asyncio.get_event_loop()
                audio, sample_rate = await loop.run_in_executor(
                    None,
                    lambda: asyncio.run(generate_speech(
                        text=normalized_text,
                        voice=request.voice,
                        language=language,
                        instruct=request.instruct,
                        speed=request.speed,
                    ))
                )
                gen_time = time.time() - gen_start

            audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            logger.info(f"TTS: Gen={gen_time:.2f}s Audio={audio_dur:.2f}s RTF={rtf:.2f}x")

            audio_bytes = encode_audio(audio, request.response_format, sample_rate)
            content_type = get_content_type(request.response_format)

            async def send_audio():
                for i in range(0, len(audio_bytes), 4096):
                    yield audio_bytes[i:i + 4096]

            return StreamingResponse(
                send_audio(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"TTS request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.websocket("/audio/speech/stream-input")
@router.websocket("/audio/speech/stream-input/{voice_id}")
async def websocket_tts_stream(websocket: WebSocket, voice_id: str = ""):
    """ElevenLabs-compatible bidirectional streaming TTS over WebSocket.

    Protocol (matches wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input):

      Client -> Server:  {"text": "Hello. ", "try_trigger_generation": true}
                         {"text": "Next sentence. "}
                         {"text": ""}            # empty text = EOS, flush + close
      Server -> Client:  {"audio": "<base64 ulaw_8000 bytes>"}
                         ...
                         {"isFinal": true}

    One connection per LLM turn: text chunks stream in, ulaw_8000 audio streams
    back continuously across sentence boundaries with no inter-sentence gap.

    Supports clone: voice profiles:
      wss://.../audio/speech/stream-input/clone:MyAgentVoice
    Profile is loaded once at connect; voice prompt is cached across connections.
    If the profile is not found the server closes with code 4404.

    Three concurrent tasks form the pipeline:
      _receive  -- WS text frames -> text_queue (None sentinel on EOS/disconnect)
      _generate -- text_queue -> backend.generate_speech_streaming (or
                   generate_voice_clone_streaming for clone: voices) -> audio_queue
                   (encoded ulaw_8000 chunks; None sentinel when text exhausted)
      _send     -- audio_queue -> real-time-paced {"audio": ...} frames, then
                   {"isFinal": true} and close
    """
    # --- Auth: accept `xi-api-key` OR `Authorization: Bearer <token>` ---
    xi_key = websocket.headers.get("xi-api-key")
    auth_header = websocket.headers.get("authorization", "")
    bearer = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else None
    provided = xi_key or bearer
    if EXPECTED_API_KEY:
        authorized = provided == EXPECTED_API_KEY
    else:
        # No server-side key configured: the Caddy proxy is the source of truth,
        # so just require that *some* credential was passed through.
        authorized = bool(provided)
    if not authorized:
        # Complete the handshake first so a real 1008 (policy violation) close
        # frame is delivered to the client, rather than a pre-accept HTTP 403.
        await websocket.accept()
        await websocket.close(code=1008)
        return

    await websocket.accept()

    loop = asyncio.get_running_loop()

    # Shared config populated from the first control frame.
    state = {
        "voice": voice_id or "",
        "language": "Auto",
        "model": "tts-1",
        "ref_audio": None,        # base64 string, set by first frame
        "ref_text": None,
        "x_vector_only_mode": False,
        "voice_id": voice_id,     # cache key hint from URL
        "response_format": "ulaw_8000",  # ulaw_8000 | pcm | mp3 | opus | wav | aac
    }

    # _generate waits on this before reading state so there is no race with
    # _receive processing the first (config) frame.
    config_ready: asyncio.Event = asyncio.Event()

    text_queue: asyncio.Queue = asyncio.Queue()
    audio_queue: asyncio.Queue = asyncio.Queue()

    async def _receive():
        """Read JSON text frames into text_queue; sentinel on EOS/disconnect."""
        first = True
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("WS stream-input: skipping non-JSON frame")
                    if first:
                        config_ready.set()
                    continue
                if first:
                    first = False
                    if msg.get("voice"):
                        state["voice"] = msg["voice"]
                    if msg.get("voice_id"):
                        state["voice_id"] = msg["voice_id"]
                    if msg.get("model_id"):
                        state["model"] = msg["model_id"]
                    if msg.get("language"):
                        state["language"] = msg["language"]
                    if msg.get("ref_audio"):
                        state["ref_audio"] = msg["ref_audio"]
                    if msg.get("ref_text") is not None:
                        state["ref_text"] = msg["ref_text"]
                    if msg.get("x_vector_only_mode") is not None:
                        state["x_vector_only_mode"] = bool(msg["x_vector_only_mode"])
                    if msg.get("response_format"):
                        state["response_format"] = msg["response_format"]
                    config_ready.set()
                text = msg.get("text")  # None = key absent (config-only frame), "" = EOS
                if text is None:
                    continue  # config-only frame, no text to enqueue
                if text == "":
                    break  # EOS
                await text_queue.put(text)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"WS stream-input receive error: {e}")
        finally:
            config_ready.set()  # unblock _generate even on early disconnect
            await text_queue.put(None)

    async def _generate():
        """Generate ulaw_8000 chunks per text segment into audio_queue.

        Three voice modes, resolved after the first frame is received:
          1. ref-at-connect: ref_audio in first frame — decode once, stream all
             sentences through that voice. voice_id (if given) is used as the
             cache key so the voice prompt survives across connections on warm boxes.
          2. clone:Name    : look up a pre-provisioned disk profile.
          3. standard      : CustomVoice / built-in voice (existing behaviour).
        """
        try:
            backend = await get_tts_backend()

            # Wait until _receive has processed the first frame so state is final.
            await config_ready.wait()

            voice = state["voice"]
            fmt = state["response_format"]

            # --- Path 1: ref_audio supplied in first frame ---
            if state.get("ref_audio"):
                ref_b64 = state["ref_audio"]
                if ref_b64.startswith("data:"):
                    ref_b64 = ref_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(ref_b64)
                ref_audio, ref_sr = sf.read(io.BytesIO(raw_bytes))
                if len(ref_audio.shape) > 1:
                    ref_audio = ref_audio.mean(axis=1)
                ref_audio = ref_audio.astype(np.float32)
                ref_text = state.get("ref_text") or None
                x_vector_only = state.get("x_vector_only_mode", False)
                clone_lang = state["language"]
                # Optional stable cache key: voice_id from URL or first frame.
                cache_key = state.get("voice_id") or None
                if cache_key:
                    _ref_audio_cache[cache_key] = (ref_audio, ref_sr)
                logger.info(f"WS clone (ref-at-connect): cache_key={cache_key} lang={clone_lang} xvec={x_vector_only}")

                while True:
                    text = await text_queue.get()
                    if text is None:
                        break
                    if not text.strip():
                        continue
                    async with _gpu_lock:
                        async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                            text=text,
                            ref_audio=ref_audio,
                            ref_audio_sr=ref_sr,
                            ref_text=ref_text,
                            language=clone_lang,
                            x_vector_only_mode=x_vector_only,
                            cache_key=cache_key,
                        ):
                            if pcm_chunk is not None and len(pcm_chunk) > 0:
                                encoded = encode_audio(pcm_chunk, fmt, sr)
                                await audio_queue.put(encoded)
                                await asyncio.sleep(0)

            # --- Path 2: clone:Name disk profile ---
            elif voice.lower().startswith("clone:"):
                profile_name = voice[6:].strip()
                try:
                    profile = _load_voice_profile(profile_name)
                except ValueError as e:
                    logger.error(f"WS clone: profile not found: {e}")
                    await websocket.send_json({"error": "profile_not_found", "message": str(e)})
                    await websocket.close(code=4404)
                    return

                if profile_name in _ref_audio_cache:
                    ref_audio, ref_sr = _ref_audio_cache[profile_name]
                else:
                    ref_audio, ref_sr = sf.read(profile["ref_audio_path"])
                    if len(ref_audio.shape) > 1:
                        ref_audio = ref_audio.mean(axis=1)
                    ref_audio = ref_audio.astype(np.float32)
                    _ref_audio_cache[profile_name] = (ref_audio, ref_sr)

                clone_lang = state["language"] if state["language"] != "Auto" else profile["language"]
                logger.info(f"WS clone (disk profile): profile='{profile_name}' lang={clone_lang} xvec={profile['x_vector_only_mode']}")

                while True:
                    text = await text_queue.get()
                    if text is None:
                        break
                    if not text.strip():
                        continue
                    async with _gpu_lock:
                        async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                            text=text,
                            ref_audio=ref_audio,
                            ref_audio_sr=ref_sr,
                            ref_text=profile["ref_text"] or None,
                            language=clone_lang,
                            x_vector_only_mode=profile["x_vector_only_mode"],
                            cache_key=profile_name,
                        ):
                            if pcm_chunk is not None and len(pcm_chunk) > 0:
                                encoded = encode_audio(pcm_chunk, fmt, sr)
                                await audio_queue.put(encoded)
                                await asyncio.sleep(0)

            # --- Path 3: standard built-in voice ---
            else:
                voice_name = get_voice_name(voice)
                while True:
                    text = await text_queue.get()
                    if text is None:
                        break
                    if not text.strip():
                        continue
                    async with _gpu_lock:
                        async for pcm_chunk, sr in backend.generate_speech_streaming(
                            text=text,
                            voice=voice_name,
                            language=state["language"],
                            model=state["model"],
                        ):
                            if pcm_chunk is not None and len(pcm_chunk) > 0:
                                encoded = encode_audio(pcm_chunk, fmt, sr)
                                await audio_queue.put(encoded)
                                await asyncio.sleep(0)

        except Exception as e:
            logger.exception(f"WS stream-input generate error: {e}")
        finally:
            await audio_queue.put(None)

    async def _send():
        """Send audio chunks to client, then isFinal + close.

        For ulaw_8000: real-time paced at 200ms intervals so Telnyx doesn't get
        a burst at the start of each utterance.
        For all other formats: send chunks as fast as they arrive — the client
        buffers (mobile AudioTrack, Web Audio API, etc.).
        """
        # Wait for the config frame before reading response_format — otherwise
        # _send races _receive and captures the default ("ulaw_8000"), applying
        # the real-time ulaw pacer to every stream. For pcm_16000 that pacer
        # sleeps 200ms per 1600 bytes (= only 50ms of 16 kHz audio), throttling
        # output to exactly 0.25x real time. _generate already waits here.
        await config_ready.wait()
        fmt = state["response_format"]
        pace = fmt == "ulaw_8000"
        buffer = bytearray()
        next_send = None

        # Coalesce raw-PCM formats into ~COALESCE_MS-of-audio messages so a
        # high-sample-rate format (pcm_16000, pcm) is not fragmented into ~4x as
        # many WS frames as ulaw — each frame carries a fixed per-message cost
        # (json + base64 + send/recv), so fragmentation, not generation, is what
        # made pcm_16000 run at ~0.25x real time. These are delivered as fast as
        # generated (NO real-time sleep): the phone client already paces to
        # Telnyx (20ms framing + prebuffer), and a second real-time pacer here
        # would fight it. ulaw keeps its own real-time pacer below. The first
        # message is emitted the instant it is available so TTFB does not
        # regress. A future config-frame `output_chunk_ms` could override
        # COALESCE_MS for finer-grained transports (e.g. a browser leg).
        COALESCE_MS = 200
        _raw_bytes_per_sec = {"pcm_16000": 16000 * 2, "pcm": 24000 * 2}
        coalesce = fmt in _raw_bytes_per_sec
        target_bytes = int(_raw_bytes_per_sec.get(fmt, 0) * COALESCE_MS / 1000)
        cbuf = bytearray()
        first_sent = False

        async def _emit(data: bytes):
            await websocket.send_json({"audio": base64.b64encode(data).decode()})

        try:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break

                if not pace:
                    if not coalesce:
                        # Container/compressed formats: send as chunks arrive.
                        await _emit(chunk)
                        continue
                    # Raw-PCM: coalesce to ~COALESCE_MS per message, no sleep.
                    cbuf.extend(chunk)
                    if not first_sent:
                        # Emit the first message immediately to preserve TTFB,
                        # then batch the rest up to the duration target.
                        await _emit(bytes(cbuf))
                        cbuf.clear()
                        first_sent = True
                        continue
                    while len(cbuf) >= target_bytes:
                        await _emit(bytes(cbuf[:target_bytes]))
                        del cbuf[:target_bytes]
                    continue

                # ulaw_8000: pace at real-time
                buffer.extend(chunk)
                if next_send is None and len(buffer) >= ULAW_CHUNK_BYTES:
                    next_send = loop.time()
                if next_send is not None:
                    while len(buffer) >= ULAW_CHUNK_BYTES:
                        delay = next_send - loop.time()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        await _emit(bytes(buffer[:ULAW_CHUNK_BYTES]))
                        del buffer[:ULAW_CHUNK_BYTES]
                        next_send += ULAW_CHUNK_INTERVAL

            # Flush ulaw remainder
            if pace:
                if next_send is None:
                    next_send = loop.time()
                while len(buffer) >= ULAW_CHUNK_BYTES:
                    delay = next_send - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await _emit(bytes(buffer[:ULAW_CHUNK_BYTES]))
                    del buffer[:ULAW_CHUNK_BYTES]
                    next_send += ULAW_CHUNK_INTERVAL
                if buffer:
                    await _emit(bytes(buffer))

            # Flush coalesce remainder for raw-PCM formats.
            elif coalesce and cbuf:
                await _emit(bytes(cbuf))

            await websocket.send_json({"isFinal": True})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"WS stream-input send error: {e}")

    recv_task = asyncio.create_task(_receive())
    gen_task = asyncio.create_task(_generate())
    send_task = asyncio.create_task(_send())
    try:
        await asyncio.gather(recv_task, gen_task, send_task)
    except Exception as e:
        logger.warning(f"WS stream-input pipeline error: {e}")
    finally:
        for t in (recv_task, gen_task, send_task):
            if not t.done():
                t.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/audio/models")
async def list_audio_models():
    """List TTS models in OpenWebUI-compatible format."""
    return {
        "models": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()

    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Serena)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Aiden)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Eric)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Dylan)"),
    ]

    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]

    try:
        backend = await get_tts_backend()
        speakers = backend.get_supported_voices()
        languages = backend.get_supported_languages()

        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = []

        # Add clone profiles from voice library
        clone_voices = []
        profiles_dir = VOICE_LIBRARY_DIR / "profiles"
        if profiles_dir.exists():
            for child in sorted(profiles_dir.iterdir()):
                meta_file = child / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("task_type") == "Base" and meta.get("ref_audio_filename"):
                        clone_id = f"clone:{meta['name']}"
                        clone_voices.append(VoiceInfo(
                            id=clone_id,
                            name=clone_id,
                            description=f"Cloned voice: {meta['name']}",
                        ).model_dump())
                except Exception:
                    pass

        return {
            "voices": voices + clone_voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }

    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        return {
            "voices": [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """Get voice cloning capabilities of the current backend."""
    try:
        backend = await get_tts_backend()
        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )
    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """Clone a voice from reference audio and generate speech."""
    try:
        backend = await get_tts_backend()

        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model.",
                    "type": "invalid_request_error",
                },
            )

        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text. Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        try:
            ref_audio_b64 = request.ref_audio
            if ref_audio_b64.startswith("data:"):
                ref_audio_b64 = ref_audio_b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(ref_audio_b64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_audio", "message": f"Failed to decode base64 audio: {e}", "type": "invalid_request_error"},
            )

        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "audio_processing_error", "message": f"Failed to process reference audio: {e}", "type": "invalid_request_error"},
            )

        normalized_text = normalize_text(request.input, request.normalization_options)
        if not normalized_text.strip():
            raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"})

        audio, sample_rate = await backend.generate_voice_clone(
            text=normalized_text, ref_audio=ref_audio, ref_audio_sr=ref_sr,
            ref_text=request.ref_text, language=request.language or "Auto",
            x_vector_only_mode=request.x_vector_only_mode, speed=request.speed,
        )

        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        content_type = get_content_type(request.response_format)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}", "Cache-Control": "no-cache"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})
