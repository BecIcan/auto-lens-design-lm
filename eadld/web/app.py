"""Small, hardened web boundary around the private seed backend."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import hmac
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eadld.initialization import DesignSpec, InitialStructureBackend, load_backend, run_generation_audit


LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class WebSettings:
    """Server-owned settings. None of these values are accepted from a browser."""

    backend_factory: str | None
    backend_config: Path | None
    access_token: str | None
    public_origin: str
    allowed_hosts: tuple[str, ...]
    result_root: Path
    result_ttl_seconds: int = 1800
    rate_limit_per_minute: int = 3
    max_concurrent: int = 1
    generation_timeout_seconds: int = 180
    max_body_bytes: int = 16_384
    trust_proxy_headers: bool = False

    @classmethod
    def from_env(cls) -> "WebSettings":
        hosts = tuple(
            item.strip()
            for item in os.getenv("EADLD_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
            if item.strip()
        )
        config = os.getenv("EADLD_BACKEND_CONFIG")
        return cls(
            backend_factory=os.getenv("EADLD_BACKEND_FACTORY"),
            backend_config=Path(config) if config else None,
            access_token=os.getenv("EADLD_ACCESS_TOKEN"),
            public_origin=os.getenv("EADLD_PUBLIC_ORIGIN", "http://127.0.0.1:8000").rstrip("/"),
            allowed_hosts=hosts,
            result_root=Path(os.getenv("EADLD_RESULT_ROOT", "outputs/web-results")),
            result_ttl_seconds=int(os.getenv("EADLD_RESULT_TTL_SECONDS", "1800")),
            rate_limit_per_minute=int(os.getenv("EADLD_RATE_LIMIT", "3")),
            max_concurrent=int(os.getenv("EADLD_MAX_CONCURRENT", "1")),
            generation_timeout_seconds=int(os.getenv("EADLD_GENERATION_TIMEOUT_SECONDS", "180")),
            max_body_bytes=int(os.getenv("EADLD_MAX_BODY_BYTES", "16384")),
            trust_proxy_headers=os.getenv("EADLD_TRUST_PROXY_HEADERS", "0") == "1",
        )


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efl_mm: float = Field(ge=5.0, le=1000.0)
    f_number: float = Field(ge=1.0, le=32.0)
    half_field_deg: float = Field(ge=0.0, lt=60.0)
    wavelengths_nm: list[float]
    elements: int = Field(ge=3, le=10)
    candidate_count: int = Field(default=3, ge=1, le=5)
    min_image_clearance_mm: float | None = Field(default=None, ge=0.0, le=200.0)
    max_package_length_mm: float | None = Field(default=None, gt=0.0, le=2000.0)
    max_distortion_fraction: float | None = Field(default=None, ge=0.0, le=0.2)
    target_cra_deg: float | None = Field(default=None, ge=0.0, le=60.0)

    def to_spec(self) -> DesignSpec:
        if not 1 <= len(self.wavelengths_nm) <= 9:
            raise ValueError("波长数量必须为 1..9")
        if any(not 350.0 <= item <= 2500.0 for item in self.wavelengths_nm):
            raise ValueError("波长范围必须在 350..2500 nm")
        return DesignSpec(
            effective_focal_length_mm=self.efl_mm,
            f_number=self.f_number,
            max_field_angle_deg=self.half_field_deg,
            wavelengths_nm=tuple(self.wavelengths_nm),
            elements=self.elements,
            candidate_count=self.candidate_count,
            min_image_clearance_mm=self.min_image_clearance_mm,
            max_package_length_mm=self.max_package_length_mm,
            max_distortion_fraction=self.max_distortion_fraction,
            target_chief_ray_angle_deg=self.target_cra_deg,
        )


class _RateLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def accept(self, key: str) -> bool:
        now = time.monotonic()
        async with self.lock:
            events = self.events[key]
            while events and now - events[0] >= 60.0:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


class _ConcurrencyGate:
    def __init__(self, maximum: int) -> None:
        self.maximum = max(1, maximum)
        self.active = 0
        self.lock = asyncio.Lock()

    async def reserve(self) -> bool:
        async with self.lock:
            if self.active >= self.maximum:
                return False
            self.active += 1
            return True

    async def release(self) -> None:
        async with self.lock:
            self.active -= 1


def _client_key(request: Request, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _authorize(settings: WebSettings, authorization: str | None) -> None:
    if not settings.access_token:
        raise HTTPException(status_code=503, detail="服务尚未配置")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, settings.access_token):
        raise HTTPException(status_code=401, detail="访问码无效")


def _cleanup_expired(settings: WebSettings) -> None:
    root = settings.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max(60, settings.result_ttl_seconds)
    for item in root.iterdir():
        if not item.is_dir() or not JOB_ID_PATTERN.fullmatch(item.name):
            continue
        if item.resolve().parent == root and item.stat().st_mtime < cutoff:
            shutil.rmtree(item)


def _safe_metrics(manifest: dict) -> dict:
    selected = manifest["selected_candidate_id"]
    record = next(row for row in manifest["candidates"] if row["candidate_id"] == selected)
    metrics = record["metrics"]
    return {
        "efl_mm": round(metrics["efl_mm"], 3),
        "bfl_mm": round(metrics["bfl_mm"], 3),
        "ttl_mm": round(metrics["ttl_mm"], 3),
        "mean_rms_um": round(metrics["mean_rms_radius_um"], 3),
        "worst_rms_um": round(metrics["worst_rms_radius_um"], 3),
        "valid_ray_percent": round(100.0 * metrics["valid_ray_fraction"], 2),
    }


def create_app(
    settings: WebSettings | None = None,
    backend: InitialStructureBackend | None = None,
    audit_runner: Callable[[DesignSpec, InitialStructureBackend, Path], dict] = run_generation_audit,
) -> FastAPI:
    settings = settings or WebSettings.from_env()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.backend = backend
    app.state.backend_lock = asyncio.Lock()
    limiter = _RateLimiter(settings.rate_limit_per_minute)
    gate = _ConcurrencyGate(settings.max_concurrent)

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > settings.max_body_bytes
            except ValueError:
                too_large = True
            if too_large:
                response = JSONResponse({"detail": "请求过大"}, status_code=413)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError):
        return JSONResponse({"detail": "设计指标格式不正确"}, status_code=422)

    async def get_backend() -> InitialStructureBackend:
        if app.state.backend is not None:
            return app.state.backend
        async with app.state.backend_lock:
            if app.state.backend is None:
                if not settings.backend_factory:
                    raise HTTPException(status_code=503, detail="服务尚未配置")
                app.state.backend = await asyncio.to_thread(
                    load_backend, settings.backend_factory, settings.backend_config
                )
        return app.state.backend

    @app.get("/")
    async def home():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.post("/api/generate")
    async def generate(
        payload: GenerateRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_eadld_request: str | None = Header(default=None),
        origin: str | None = Header(default=None),
    ):
        if x_eadld_request != "1" or origin != settings.public_origin:
            raise HTTPException(status_code=403, detail="请求来源无效")
        key = _client_key(request, settings.trust_proxy_headers)
        if not await limiter.accept(key):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        _authorize(settings, authorization)
        try:
            spec = payload.to_spec()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        if not await gate.reserve():
            raise HTTPException(status_code=429, detail="服务器正在生成另一组结构")

        job_id = uuid4().hex
        output_dir = settings.result_root.resolve() / job_id

        async def work() -> dict:
            try:
                _cleanup_expired(settings)
                private_backend = await get_backend()
                manifest = await asyncio.to_thread(audit_runner, spec, private_backend, output_dir)
                if not (output_dir / "layout.png").is_file() or not (output_dir / "spots.png").is_file():
                    raise RuntimeError("生成结果不完整")
                return manifest
            finally:
                await gate.release()

        task = asyncio.create_task(work())
        try:
            manifest = await asyncio.wait_for(
                asyncio.shield(task), timeout=max(10, settings.generation_timeout_seconds)
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="生成超时") from None
        except Exception:
            LOGGER.exception("Initial-structure generation failed for job %s", job_id)
            if output_dir.exists() and output_dir.resolve().parent == settings.result_root.resolve():
                shutil.rmtree(output_dir)
            raise HTTPException(status_code=422, detail="当前指标未找到可用初始结构") from None

        return {
            "job_id": job_id,
            "metrics": _safe_metrics(manifest),
            "images": {
                "layout": f"/api/results/{job_id}/layout",
                "spots": f"/api/results/{job_id}/spots",
            },
        }

    @app.get("/api/results/{job_id}/{artifact}")
    async def result_image(
        job_id: str,
        artifact: str,
        authorization: str | None = Header(default=None),
        x_eadld_request: str | None = Header(default=None),
    ):
        _authorize(settings, authorization)
        if x_eadld_request != "1" or not JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="结果不存在")
        filenames = {"layout": "layout.png", "spots": "spots.png"}
        filename = filenames.get(artifact)
        if filename is None:
            raise HTTPException(status_code=404, detail="结果不存在")
        root = settings.result_root.resolve()
        job_dir = root / job_id
        if (
            job_dir.exists()
            and job_dir.resolve().parent == root
            and job_dir.stat().st_mtime < time.time() - max(60, settings.result_ttl_seconds)
        ):
            shutil.rmtree(job_dir)
            raise HTTPException(status_code=404, detail="结果已过期")
        path = job_dir / filename
        if path.resolve().parent.parent != settings.result_root.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="结果不存在")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("eadld.web.app:app", host="127.0.0.1", port=8000, proxy_headers=False)
