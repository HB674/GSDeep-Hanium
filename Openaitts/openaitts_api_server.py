# Openaitts/openaitts_api_server.py
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# Env & Paths
# -------------------------------------------------
load_dotenv()  # .env에서 OPENAI_API_KEY 로딩
client = OpenAI()

SHARED_DIR = Path(os.getenv("SHARED_DIR", "/app/shared_data_workspace"))

# 파이프라인 표준 디렉토리
INPUT_AUDIO_DIR = SHARED_DIR / "input_audio"
INPUT_TEXT_DIR  = SHARED_DIR / "input_text"
WARMUP_DIR      = SHARED_DIR / "warmup"

# 디렉토리 보장 (요청대로 openaitts_output_queue는 만들지 않음)
SHARED_DIR.mkdir(parents=True, exist_ok=True)
INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
INPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="OpenAITTS API Server",
    description="OpenAI TTS(SSML 지원) — input_text 최신 파일을 읽어 input_audio에 음성 저장",
    version="1.2.0",
)

# -------------------------------------------------
# Models
# -------------------------------------------------
class HealthResp(BaseModel):
    status: str = "ok"
    openai_key_loaded: bool = True

# -------------------------------------------------
# Utils
# -------------------------------------------------
def _safe_filename(stem: str, ext: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (".", "_", "-", " ") else "_" for ch in stem).strip()
    safe = safe or "tts_output"
    if not ext.startswith("."):
        ext = "." + ext
    return safe + ext

def _default_name(ext: str, base_stem: Optional[str] = None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = (base_stem or "tts").strip()
    prefix = "".join(ch if ch.isalnum() or ch in (".", "_", "-", " ") else "_" for ch in prefix)
    prefix = prefix or "tts"
    return f"{prefix}_{ts}.{ext}"

def _pick_latest(directory: Path, patterns: List[str]) -> Optional[Path]:
    cand: List[Path] = []
    for pat in patterns:
        cand.extend(directory.glob(pat))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

def _read_text_file(fp: Path) -> str:
    # UTF-8 우선, 실패 시 국제 인코딩 폴백
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        try:
            return fp.read_text(encoding="utf-16")
        except Exception:
            return fp.read_text(errors="ignore")

def _ensure_ssml_wrapped(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s if s.lstrip().startswith("<speak>") else f"<speak>{s}</speak>"

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/health", response_model=HealthResp)
def health():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return HealthResp(status="ok", openai_key_loaded=has_key)

@app.post("/warmup")
def warmup():
    """
    매우 짧은 합성으로 OpenAI TTS 연결 및 초기 로딩 확인.
    파일은 생성하지 않습니다.
    """
    try:
        text = "<speak>테스트.</speak>"
        _ = client.audio.speech.create(
            model="tts-1-hd",
            voice="onyx",
            input=text,
            response_format="mp3",
        )
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/synthesize")
def synthesize(
    # 1) 직접 텍스트 입력(옵션) — 주지 않으면 input_text 최신 파일을 자동 사용
    text: Optional[str] = Form(None, description="합성할 텍스트(SSML 허용). 비우면 input_text 최신 파일 사용"),
    # 2) TTS 옵션
    voice: str = Form("onyx"),
    response_format: str = Form("mp3"),  # "mp3" | "wav"
    output_basename: Optional[str] = Form(None, description="저장 파일명(확장자 제외). 없으면 자동 생성"),
    auto_ssml_wrap: bool = Form(True, description="True면 <speak> 자동 래핑"),
):
    """
    텍스트(직접 입력 또는 input_text 최신 파일) → 음성 파일 생성 후 {SHARED_DIR}/input_audio/에 저장.
    - Applio가 input_audio/를 스캔해 바로 사용 가능.
    """
    # 1) 텍스트 확보: 폼 입력 → 없으면 input_text 최신 파일
    source_path: Optional[Path] = None
    if text is None or not text.strip():
        latest = _pick_latest(INPUT_TEXT_DIR, ["*.txt", "*.ssml", "*.xml"])
        if latest is None:
            return JSONResponse(
                {"status": "error", "error": "No text provided and no files under input_text/."},
                status_code=400,
            )
        text_raw = _read_text_file(latest)
        source_path = latest
    else:
        text_raw = text

    text_raw = text_raw.strip()
    if not text_raw:
        return JSONResponse({"status": "error", "error": "Empty text."}, status_code=400)

    # 2) SSML 처리
    text_to_speak = _ensure_ssml_wrapped(text_raw) if auto_ssml_wrap else text_raw

    # 3) 파일명/확장자
    ext = "mp3" if response_format.lower() == "mp3" else "wav"
    base_stem = (source_path.stem if source_path else "tts")
    fname = output_basename or _default_name(ext, base_stem)
    safe_name = _safe_filename(Path(fname).stem, f".{ext}")

    # 최종 저장 위치: input_audio/
    out_path = INPUT_AUDIO_DIR / safe_name

    try:
        # 4) OpenAI TTS 호출
        resp = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text_to_speak,
            response_format=ext,
        )
        resp.stream_to_file(out_path)

        return {
            "status": "ok",
            "message": "synthesized",
            "output": str(out_path.resolve()),
            "relative": str(out_path.resolve().relative_to(SHARED_DIR.resolve())),
            "voice": voice,
            "format": ext,
            "source_text_file": str(source_path) if source_path else None,
        }
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "error": str(e),
                "hint": [
                    "1) .env의 OPENAI_API_KEY 확인",
                    "2) 크레딧/요금 상태 확인",
                    "3) 입력 텍스트 길이/SSML 검사",
                ],
            },
            status_code=500,
        )
