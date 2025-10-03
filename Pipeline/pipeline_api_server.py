# pipeline_api_server.py
import os
import asyncio
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from fastapi.middleware.cors import CORSMiddleware
from typing import List  # 파일 상단에 이미 있다면 생략

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ==== 추가 import ====
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from datetime import datetime
import re

app = FastAPI(
    title="Pipeline API Server",
    description="마이크로서비스 상태를 종합하고 파이프라인 잡을 오케스트레이션하는 API",
    version="0.3.0",
)

# ★ 이 블록을 app 생성 직후에 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gs-deep-hanium.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://0.0.0.0:5500",
    ],
    allow_origin_regex=r"^https://.*\.vercel\.app$",  # 프리뷰 허용(옵션)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 공용 설정/경로
# -------------------------------------------------
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/app/shared_data_workspace"))
INPUT_AUDIO_DIR = SHARED_DIR / "input_audio"
INPUT_IMAGE_DIR = SHARED_DIR / "input_image"
INPUT_TEXT_DIR  = SHARED_DIR / "input_text"  
for d in [INPUT_AUDIO_DIR, INPUT_IMAGE_DIR, INPUT_TEXT_DIR]: 
    d.mkdir(parents=True, exist_ok=True)

# 각 서비스의 엔드포인트
SERVICE_URLS: Dict[str, str] = {
    "applio":    "http://applio_api:8001/health",
    "sadtalker": "http://sadtalker_api:8002/health",
    "wav2lip":   "http://wav2lip_api:8003/health",
    "gfpgan":    "http://gfpgan_api:8004/health",
    "openaitts": "http://openaitts_api:8005/health",
}
APPLIO_INFER_URL    = "http://applio_api:8001/infer"
SADTALKER_INFER_URL = "http://sadtalker_api:8002/infer"
W2L_INFER_URL       = "http://wav2lip_api:8003/infer"
GFPGAN_ENH_URL      = "http://gfpgan_api:8004/enhance_video"
OPENAITTS_SYN_URL    = "http://openaitts_api:8005/synthesize"


# -------------------------------------------------
# 음성 프로필 매핑 (예시)
#  - 프론트엔드는 voice_profile만 넘기고, 서버가 경로로 매핑
#  - 필요에 맞게 실제 경로로 교체하세요.
# -------------------------------------------------
VOICE_PROFILES: Dict[str, Tuple[str, str]] = {
    # "profile_key": ("voice_model/xxx/model.pth", "voice_model/xxx/model.index")
    "male_young": ("/app/voice_model/male_young/model.pth", "/app/voice_model/male_young/model.index"),
    "male_adult": ("/app/voice_model/swain/swain.pth", "/app/voice_model/swain/swain.index"),
    "female_young": ("/app/voice_model/irelia/irelia.pth", "/app/voice_model/irelia/irelia.index"),
    # 필요 시 더 추가
}

# TTS 경로에서만 사용하는: voice_profile -> 고정 pitch 매핑
VOICE_PROFILE_TO_PITCH = {
    # 형님이 원하시는 값으로 자유롭게 수정하세요
    "female_young": 8,
    "male_adult":   -2,
    "male_young":   6,
    # 필요 시 추가...
}

def resolve_profile_paths(profile: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    if not profile:
        return None, None
    pair = VOICE_PROFILES.get(profile)
    if not pair:
        return None, None
    p_pth, p_idx = map(Path, pair)
    # 절대경로면 그대로, 상대경로면 SHARED_DIR 기준
    if not p_pth.is_absolute():
        p_pth = (SHARED_DIR / p_pth)
    if not p_idx.is_absolute():
        p_idx = (SHARED_DIR / p_idx)
    return p_pth, p_idx

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _ffmpeg_convert_to_wav48k_inplace(src: Path) -> Path:
    """
    입력 오디오(src)를 같은 디렉토리에 48kHz mono 16-bit PCM WAV로 변환하고,
    변환 성공 시 원본(src)을 삭제. 변환본 경로를 반환.
    - 원본 확장자와 무관하게 <stem>_48k.wav로 생성
    """
    dst = src.with_name(f"{src.stem}_48k.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {proc.stderr[:400]}")

    # 변환 성공 시 원본 삭제
    try:
        if dst.exists():
            src.unlink(missing_ok=True)
    except Exception:
        # 삭제 실패는 치명적이지 않음
        pass
    return dst

def _pick_latest(directory: Path, patterns):
    files = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

# -------------------------------------------------
# /health (기존)
# -------------------------------------------------
@app.get("/health")
async def health():
    results = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        tasks = {name: client.get(url) for name, url in SERVICE_URLS.items()}
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for (name, _), resp in zip(SERVICE_URLS.items(), responses):
        if isinstance(resp, Exception):
            results[name] = {"status": "unreachable", "detail": str(resp)}
        elif resp.status_code != 200:
            results[name] = {"status": f"http_{resp.status_code}"}
        else:
            try:
                results[name] = resp.json()
            except Exception:
                results[name] = {"status": "invalid_json"}

    overall_ok = all(
        isinstance(r, dict) and r.get("status") in ("ok", "healthy") for r in results.values()
    )

    return JSONResponse({
        "status": "ok" if overall_ok else "degraded",
        "services": results
    })

# -------------------------------------------------
# /warmup (기존)
# -------------------------------------------------
@app.post("/warmup")
async def warmup():
    warmup_urls = {name: f"{url.rsplit('/', 1)[0]}/warmup"
                   for name, url in SERVICE_URLS.items()}
    async def _post_with_timing(client: httpx.AsyncClient, name: str, url: str):
        t0 = time.perf_counter()
        try:
            resp = await client.post(url)
            ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except Exception:
                    payload = None
                return name, {"ok": True, "ms": ms, "response": payload}
            else:
                return name, {"ok": False, "ms": ms, "status_code": resp.status_code, "text": resp.text[:300]}
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            return name, {"ok": False, "ms": ms, "error": str(e)}

    async with httpx.AsyncClient(timeout=None) as client:
        pairs = await asyncio.gather(
            *[_post_with_timing(client, name, url) for name, url in warmup_urls.items()]
        )
    results = dict(pairs)

    overall_ok = all(v.get("ok") for v in results.values())
    return JSONResponse({"status": "ok" if overall_ok else "partial", "warmup": results})

# -------------------------------------------------
# 잡 모델/상태
# -------------------------------------------------
class AudioJobRequest(BaseModel):
    # 경로 모드
    audio_path: str = Field(..., description="SHARED_DIR 내부의 입력 오디오 경로 (예: input_audio/foo.mp3)")
    image_path: Optional[str] = Field(None, description="없으면 input_image 최신 사용")

    # Applio 제어 (나머지 단계는 기본값)
    use_applio: bool = True
    pitch: Optional[float] = None

    # (A) 권장: voice_profile로 모델 지정 → 서버에서 경로 매핑
    voice_profile: Optional[str] = None

    # (B) 직접 경로 지정(voice_profile이 있으면 무시하거나, 없을 때만 사용)
    pth_path: Optional[str] = None
    index_path: Optional[str] = None

class TTSJobRequest(BaseModel):
    # 1) TTS 입력: text를 생략하면 OpenAITTS가 input_text/의 최신 파일을 자동 사용
    tts_text: Optional[str] = Field(None, description="합성할 텍스트(SSML 허용). 비우면 input_text 최신 파일 사용")
    voice: Optional[str] = Field("nova", description="OpenAITTS 보이스")
    response_format: Optional[str] = Field("mp3", description='"mp3" | "wav"')
    output_basename: Optional[str] = Field(None, description="출력 파일명(확장자 제외)")
    auto_ssml_wrap: Optional[bool] = Field(True, description="True면 <speak> 자동 래핑")

    # 2) 영상 입력
    image_path: Optional[str] = Field(None, description="없으면 input_image 최신 사용")

    # 3) Applio 제어 (기존과 동일)
    use_applio: bool = True
    pitch: Optional[float] = None
    voice_profile: Optional[str] = None
    pth_path: Optional[str] = None
    index_path: Optional[str] = None

class JobState(BaseModel):
    job_id: str
    status: str               # queued | running | failed | done
    step: str                 # current step name
    created_at: float
    updated_at: float
    params: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    error: Optional[str] = None
    timings: Dict[str, int] = {}   # ★ 추가: 단계별/총 소요시간(ms)

JOBS: Dict[str, JobState] = {}
JOB_SEM = asyncio.Semaphore(1)  # 전체 파이프라인 동시 1개 (순차 처리)

def _new_job_state(req: AudioJobRequest) -> JobState:
    now = time.time()
    return JobState(
        job_id=uuid.uuid4().hex,
        status="queued",
        step="init",
        created_at=now,
        updated_at=now,
        params=req.dict(),
        artifacts={},
        error=None,
        timings={},   # ★ 추가
    )

def _update_job(job: JobState, **patch):
    for k, v in patch.items():
        setattr(job, k, v)
    job.updated_at = time.time()
    JOBS[job.job_id] = job

# 경로 정규화(SHARED_DIR 기준)
def _as_shared_path(p: str) -> Path:
    if p.startswith("/"):
        # 컨테이너 내부 절대경로면 그대로 쓰되 SHARED_DIR 하위인지 최소한 확인하고 싶으면 추가 검증 가능
        return Path(p)
    return SHARED_DIR / p

# -------------------------------------------------
# 잡 워커
# -------------------------------------------------
async def _run_audio_job(job: JobState):
    _update_job(job, status="running", step="prepare")
    t_total0 = time.perf_counter()       # ★ 총 소요시간 측정 시작

    # 입력 경로
    audio_in = _as_shared_path(job.params["audio_path"])
    img_param = job.params.get("image_path")
    if img_param:
        image_in = _as_shared_path(img_param)
    else:
        latest_img = _pick_latest(INPUT_IMAGE_DIR, ["*.png","*.jpg","*.jpeg","*.webp"])
        if not latest_img:
            return _update_job(job, status="failed", step="prepare", error="no image under input_image/")
        image_in = latest_img
    if not audio_in.exists():
        return _update_job(job, status="failed", step="prepare", error=f"audio not found: {audio_in}")
    if not image_in.exists():
        return _update_job(job, status="failed", step="prepare", error=f"image not found: {image_in}")

    # 0) 오디오 48k 변환(같은 디렉토리에 <stem>_48k.wav 생성) + 원본 삭제
    try:
        t = time.perf_counter()          # ★ 변환 단계 시작
        audio_48k = _ffmpeg_convert_to_wav48k_inplace(audio_in)
        job.artifacts["audio_48k"] = str(audio_48k)
        job.timings["audio_convert"] = int((time.perf_counter() - t) * 1000)  # ★ 기록
        _update_job(job, step="applio" if job.params["use_applio"] else "sadtalker",
                    artifacts=job.artifacts, timings=job.timings)
    except Exception as e:
        job.timings["audio_convert"] = int((time.perf_counter() - t) * 1000)  # 실패여도 기록
        job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
        return _update_job(job, status="failed", step="audio_convert", error=str(e), timings=job.timings)

    # Applio 모델 경로 결정: voice_profile 우선 → 없으면 pth/index → 둘 다 없으면 Applio 기본
    pth_path: Optional[Path] = None
    index_path: Optional[Path] = None
    if job.params.get("voice_profile"):
        pth_path, index_path = resolve_profile_paths(job.params["voice_profile"])
    if (not pth_path or not index_path) and job.params.get("pth_path") and job.params.get("index_path"):
        # 명시 경로가 있으면(둘 다 있을 때만) 그것으로 덮어쓰기
        pth_path = _as_shared_path(job.params["pth_path"])
        index_path = _as_shared_path(job.params["index_path"])

    # 전체 파이프라인 순차 실행 보장
    async with JOB_SEM:
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                # 1) Applio (선택)
                if job.params.get("use_applio", True):
                    form = { "input_wav_path": str(audio_48k) }
                    if job.params.get("pitch") is not None:
                        form["pitch"] = str(job.params["pitch"])
                    if pth_path and index_path:
                        form["pth_path"] = str(pth_path)
                        form["index_path"] = str(index_path)

                    _update_job(job, step="applio", timings=job.timings)
                    t = time.perf_counter()                                  # ★ applio 시작
                    r = await client.post(APPLIO_INFER_URL, data=form)
                    job.timings["applio"] = int((time.perf_counter() - t) * 1000)  # ★ 기록
                    if r.status_code != 200:
                        job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                        return _update_job(job, status="failed", step="applio",
                                           error=f"applio http_{r.status_code}: {r.text[:300]}",
                                           timings=job.timings)
                    jr = r.json()
                    audio_for_video = Path(jr.get("output") or jr.get("out") or jr.get("audio") or "")
                    if not audio_for_video or not audio_for_video.exists():
                        job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                        return _update_job(job, status="failed", step="applio",
                                           error=f"applio output not found in response: {jr}",
                                           timings=job.timings)
                    job.artifacts["applio_audio"] = str(audio_for_video)
                else:
                    audio_for_video = audio_48k

                _update_job(job, step="sadtalker", artifacts=job.artifacts, timings=job.timings)

                # 2) SadTalker
                t = time.perf_counter()                                      # ★ sadtalker 시작
                s_form = {"driven_audio": str(audio_for_video), "source_image": str(image_in)}
                r = await client.post(SADTALKER_INFER_URL, data=s_form)
                job.timings["sadtalker"] = int((time.perf_counter() - t) * 1000)  # ★ 기록
                if r.status_code != 200:
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="sadtalker",
                                       error=f"sadtalker http_{r.status_code}: {r.text[:300]}",
                                       timings=job.timings)
                jr = r.json()
                face_video = Path(jr.get("output") or jr.get("face") or jr.get("result") or "")
                if not face_video or not face_video.exists():
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="sadtalker",
                                       error=f"sadtalker output not found in response: {jr}",
                                       timings=job.timings)
                job.artifacts["sadtalker_face"] = str(face_video)
                _update_job(job, step="wav2lip", artifacts=job.artifacts, timings=job.timings)

                # 3) Wav2Lip
                t = time.perf_counter()                                      # ★ wav2lip 시작
                w_form = {"face": str(face_video), "audio": str(audio_for_video)}
                r = await client.post(W2L_INFER_URL, data=w_form)
                job.timings["wav2lip"] = int((time.perf_counter() - t) * 1000)  # ★ 기록
                if r.status_code != 200:
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="wav2lip",
                                       error=f"wav2lip http_{r.status_code}: {r.text[:300]}",
                                       timings=job.timings)
                jr = r.json()
                w2l_out = Path(jr.get("output") or jr.get("result") or "")
                if not w2l_out or not w2l_out.exists():
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="wav2lip",
                                       error=f"wav2lip output not found in response: {jr}",
                                       timings=job.timings)
                job.artifacts["wav2lip_out"] = str(w2l_out)
                _update_job(job, step="gfpgan", artifacts=job.artifacts, timings=job.timings)

                # 4) GFPGAN
                t = time.perf_counter()                                      # ★ gfpgan 시작
                g_form = {"input_video_path": str(w2l_out)}
                r = await client.post(GFPGAN_ENH_URL, data=g_form)
                job.timings["gfpgan"] = int((time.perf_counter() - t) * 1000)     # ★ 기록
                if r.status_code != 200:
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="gfpgan",
                                    error=f"gfpgan http_{r.status_code}: {r.text[:300]}",
                                    timings=job.timings)
                jr = r.json()

                # ★ GFPGAN이 돌려준 최종 출력 경로 (보통 절대경로)
                final_out = Path(jr.get("output") or "")
                if not final_out or not final_out.exists():
                    job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                    return _update_job(job, status="failed", step="gfpgan",
                                    error=f"gfpgan output not found in response: {jr}",
                                    timings=job.timings)

                # ★ 여기서부터 핵심: SHARED_DIR 기준 '상대경로'로 변환해서 artifacts에 저장
                try:
                    # 정석: realpath 기준으로 상대경로 계산
                    final_rel = str(final_out.resolve().relative_to(SHARED_DIR.resolve()))
                except Exception:
                    # 드물게 상대화 실패 시, 문자열로 강제 절단 (보수적 폴백)
                    s = str(final_out.resolve())
                    prefix = str(SHARED_DIR.resolve()) + os.sep
                    final_rel = s[len(prefix):] if s.startswith(prefix) else s

                # 프런트는 /files/<상대경로> 로 접근합니다.
                job.artifacts["final"] = final_rel         # 예: "gfpgan_output_queue/xxx.mp4"
                # (선택) 디버깅용으로 절대경로도 남기고 싶으면 아래 한 줄 추가:
                # job.artifacts["final_abs"] = str(final_out.resolve())

                # ★ 총 소요시간 기록 후 완료
                job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)
                return _update_job(job, status="done", step="done",
                                artifacts=job.artifacts, timings=job.timings)

            except Exception as e:
                job.timings["total"] = int((time.perf_counter() - t_total0) * 1000)  # ★ 실패여도 total 기록
                return _update_job(job, status="failed", error=str(e), timings=job.timings)
# -------------------------------------------------
# 엔드포인트: 잡 생성/조회
# -------------------------------------------------
from typing import Iterable

def _pick_latest_image() -> Optional[Path]:
    return _pick_latest(INPUT_IMAGE_DIR, ["*.png","*.jpg","*.jpeg","*.webp"])

@app.post("/jobs/tts")
async def create_tts_audio_job(req: TTSJobRequest):
    # 0) 이미지 확보: 주어지면 검증, 없으면 최신 자동
    if req.image_path and req.image_path.strip():
        image_in = _as_shared_path(req.image_path)
        if not image_in.exists():
            raise HTTPException(status_code=404, detail=f"image not found: {image_in}")
        image_auto = False
    else:
        latest_img = _pick_latest_image()
        if not latest_img:
            raise HTTPException(status_code=404, detail="no image found under input_image/")
        image_in = latest_img
        image_auto = True

    # 1) OpenAITTS 호출 준비
    form = {
        "voice": (req.voice or "nova"),
        "response_format": (req.response_format or "mp3"),
        "auto_ssml_wrap": "true" if (req.auto_ssml_wrap is None or req.auto_ssml_wrap) else "false",
    }
    text_source = "inline"
    if req.tts_text is not None and req.tts_text.strip():
        form["text"] = req.tts_text
    else:
        text_source = "latest_file"
    if req.output_basename:
        form["output_basename"] = req.output_basename

    # 2) 합성 실행 + 시간 측정
    async with httpx.AsyncClient(timeout=None) as client:
        t0 = time.perf_counter()
        r = await client.post(OPENAITTS_SYN_URL, data=form)
        ms = int((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"openaitts http_{r.status_code}: {r.text[:300]}")
        jr = r.json()
        rel = jr.get("relative")
        out = jr.get("output")
        if not rel and out:
            try:
                rel = str(Path(out).resolve().relative_to(SHARED_DIR.resolve()))
            except Exception:
                rel = out
        if not rel:
            raise HTTPException(status_code=500, detail=f"openaitts response missing path: {jr}")

    mapped_pitch = VOICE_PROFILE_TO_PITCH.get(req.voice_profile, 0)

    # 3) 기존 /jobs/audio와 동일한 요청으로 변환
    audio_req = AudioJobRequest(
        audio_path=rel,
        image_path=str(image_in.resolve().relative_to(SHARED_DIR.resolve())),
        use_applio=req.use_applio,
        pitch=mapped_pitch,
        voice_profile=req.voice_profile,
        pth_path=req.pth_path,
        index_path=req.index_path,
    )

    # 4) 잡 생성 + TTS 메타/타이밍/아티팩트 선기록
    job = _new_job_state(audio_req)
    job.params["tts_mapped_pitch"] = mapped_pitch
    job.params["tts"] = {
        "voice": req.voice or "nova",
        "response_format": req.response_format or "mp3",
        "auto_ssml_wrap": bool(req.auto_ssml_wrap if req.auto_ssml_wrap is not None else True),
        "text_source": text_source,  # "inline" | "latest_file"
        "output_basename": req.output_basename,
    }
    job.params["image_auto_selected"] = image_auto
    job.artifacts["tts_audio"] = rel                 # 상대경로
    job.artifacts["tts_audio_abs"] = out or None     # 절대경로(있으면)
    job.timings["openaitts"] = ms

    JOBS[job.job_id] = job
    asyncio.create_task(_run_audio_job(job))
    return {"job_id": job.job_id, "status": job.status}

@app.post("/jobs/audio")
async def create_audio_job(req: AudioJobRequest):
    """
    경로 모드: 웹에서 저장한 image/audio 경로를 받아 파이프라인 실행(비동기).
    - 입력 오디오는 input_audio/ 같은 위치에서 주시면 됩니다.
    - 서버는 해당 오디오를 같은 디렉토리에 48kHz WAV로 변환하고, 원본 파일은 삭제합니다.
    - Applio는 voice_profile 또는 pth/index 경로로 제어(voice_profile 우선).
    """
    job = _new_job_state(req)
    JOBS[job.job_id] = job
    asyncio.create_task(_run_audio_job(job))
    return {"job_id": job.job_id, "status": job.status}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.dict()

@app.get("/jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=500, description="가져올 최대 개수"),
    status: Optional[str] = Query(None, pattern="^(queued|running|failed|done)$", description="상태 필터"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="created_at 정렬"),
    since: Optional[float] = Query(None, description="updated_at >= since (epoch seconds)"),
    detailed: bool = Query(False, description="True면 전체 필드 반환"),
):
    """
    잡 목록을 최신순(default)으로 반환합니다.
    - /jobs?limit=20
    - /jobs?status=running
    - /jobs?order=asc
    - /jobs?since=1695380000
    - /jobs?detailed=true
    """
    items: List[JobState] = list(JOBS.values())

    # 필터링
    if status:
        items = [j for j in items if j.status == status]
    if since is not None:
        items = [j for j in items if j.updated_at >= float(since)]

    # 정렬
    rev = (order != "asc")
    items.sort(key=lambda j: j.created_at, reverse=rev)

    # 개수 제한
    items = items[:limit]

    if detailed:
        # pydantic 모델 그대로(dict) 반환
        payload = [j.dict() for j in items]
    else:
        # 요약본(주로 확인하는 필드만)
        payload = [{
            "job_id": j.job_id,
            "status": j.status,
            "step": j.step,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
            "final": j.artifacts.get("final"),
            "error": j.error,
        } for j in items]

    return {"count": len(payload), "items": payload}

# ==== 공용 유틸 ====
_valid_name_re = re.compile(r"[^A-Za-z0-9._-]+")

def _safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = _valid_name_re.sub("_", name)
    return name or "upload"

def _save_upload_to(dirpath: Path, up: UploadFile, prefix: str = "") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    original = _safe_filename(up.filename or f"{prefix}_upload")
    stem = Path(original).stem
    suffix = Path(original).suffix.lower()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}{stem}_{ts}{suffix}"
    dst = dirpath / fname

    with dst.open("wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dst

def _save_text_to(dirpath: Path, content: str, basename: Optional[str] = None) -> Path:
    """
    텍스트 내용을 UTF-8 .txt 파일로 저장.
    - basename이 있으면 그 이름(확장자 제외)을 기반으로 저장
    - 없으면 timestamp로 파일명 생성
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if basename:
        stem = _safe_filename(Path(basename).stem)
        fname = f"{stem}_{ts}.txt"
    else:
        fname = f"text_{ts}.txt"
    dst = dirpath / fname
    dst.write_text(content, encoding="utf-8", errors="ignore")
    return dst

def _ensure_under_shared_any(p: Path) -> Path:
    """
    절대/상대 경로를 모두 받아 SHARED_DIR 하위의 실제 파일로 확정.
    - 절대 경로면 그대로 resolve 후 SHARED_DIR 하위 여부 확인
    - 상대 경로면 SHARED_DIR/상대경로 로 resolve
    """
    base = SHARED_DIR.resolve()
    rp = p if p.is_absolute() else (SHARED_DIR / p)
    rp = rp.resolve()
    try:
        rp.relative_to(base)
    except Exception:
        raise HTTPException(status_code=403, detail="path not allowed")
    if not rp.exists() or not rp.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return rp

def _file_response(p: Path) -> FileResponse:
    ext = p.suffix.lower()
    media = "application/octet-stream"
    if ext in (".mp4", ".m4v"): media = "video/mp4"
    elif ext in (".mov",):       media = "video/quicktime"
    elif ext in (".png",):       media = "image/png"
    elif ext in (".jpg", ".jpeg"): media = "image/jpeg"
    elif ext in (".wav",):       media = "audio/wav"
    elif ext in (".mp3",):       media = "audio/mpeg"

    return FileResponse(
        path=str(p),
        media_type=media,
        filename=p.name,
        headers={"Content-Disposition": f'inline; filename="{p.name}"'}
    )

# ==== 1) 업로드 엔드포인트 ====
@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(..., description="소스 이미지 (jpg/png 등)"),
    audio: Optional[UploadFile] = File(None, description="옵션: 48k 변환 전 원본 오디오(wav 등)"),
    message: Optional[str] = Form(None),
):
    """
    - 이미지 파일은 SHARED_DIR/input_image/ 에 저장
    - 오디오 파일은 SHARED_DIR/input_audio/ 에 저장
    - 저장된 상대경로(파이프라인이 요구하는 형태)를 반환
    """
    # 이미지 저장
    try:
        img_path = _save_upload_to(INPUT_IMAGE_DIR, file, prefix="")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # 오디오 저장(선택)
    audio_rel: Optional[str] = None
    if audio is not None:
        try:
            a_path = _save_upload_to(INPUT_AUDIO_DIR, audio, prefix="")
            # 파이프라인은 경로 문자열을 상대/절대 모두 처리하지만
            # 일관성을 위해 SHARED_DIR 기준 상대 경로로 반환
            audio_rel = str(a_path.relative_to(SHARED_DIR))
        finally:
            try:
                audio.file.close()
            except Exception:
                pass

    image_rel = str(img_path.relative_to(SHARED_DIR))

    # 업로드 메타 메시지는 현재 저장만 하지 않고 응답에 포함
    return JSONResponse({
        "ok": True,
        "image_path": image_rel,   # 예: "input_image/You_20250922_190012_123456.png"
        "audio_path": audio_rel,   # 예: "input_audio/011_20250922_190012_123456.wav" (없으면 null)
        "message": message
    })

# ==== 1-b) 업로드 엔드포인트(이미지 + 텍스트) ====
@app.post("/api/upload_with_text")
async def api_upload_with_text(
    file: UploadFile = File(..., description="소스 이미지 (jpg/png 등)"),
    text: str = Form(..., description="합성에 사용할 텍스트(SSML 허용 가능, 일반 문장도 OK)"),
    text_basename: Optional[str] = Form(None, description="텍스트 파일명(확장자 제외). 없으면 자동"),
    message: Optional[str] = Form(None),
):
    """
    - 이미지는 SHARED_DIR/input_image/ 에 저장
    - 텍스트는 SHARED_DIR/input_text/ 에 .txt로 저장
    - 저장된 상대경로를 반환 (OpenAITTS는 input_text/의 최신을 자동 사용 가능)
    """
    # 1) 이미지 저장
    try:
        img_path = _save_upload_to(INPUT_IMAGE_DIR, file, prefix="")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # 2) 텍스트 저장
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="text is empty")
    txt_path = _save_text_to(INPUT_TEXT_DIR, text, basename=text_basename)

    # 3) 상대경로로 반환
    image_rel = str(img_path.relative_to(SHARED_DIR))
    text_rel  = str(txt_path.relative_to(SHARED_DIR))

    return JSONResponse({
        "ok": True,
        "image_path": image_rel,  # 예: "input_image/face_20251003_....png"
        "text_path": text_rel,    # 예: "input_text/prompt_20251003_....txt"
        "message": message
    })

# ==== 2) 정적 파일 서빙 ====
@app.get("/files/{path:path}")
async def get_shared_file_by_rel(path: str):
    """
    상대경로 버전: /files/gfpgan_output_queue/final.mp4
    """
    target = _ensure_under_shared_any(Path(path))  # 상대경로 처리
    return _file_response(target)

@app.get("/files/")
async def get_shared_file_by_query(path: str = Query(..., description="절대 또는 상대 경로")):
    """
    절대/상대 경로 쿼리 버전: /files/?path=/app/shared_data_workspace/...
    """
    target = _ensure_under_shared_any(Path(path))  # 절대/상대 모두 처리
    return _file_response(target)
