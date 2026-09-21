"""Aplicacao FastAPI: rotas HTTP, SSE de status e lifespan.

Entrypoint ASGI: ``uvicorn app.main:app``.

Rotas (shapes exatos do contrato):
    POST   /api/v1/jobs                  -> 202 JobView
    GET    /api/v1/jobs/{job_id}         -> 200 JobView | 404
    GET    /api/v1/jobs/{job_id}/events  -> text/event-stream (evento "status")
    GET    /api/v1/jobs                  -> 200 {items, total}
    DELETE /api/v1/jobs/{job_id}         -> 204
    GET    /api/v1/images/{image_id}     -> binario (attachment)
    GET    /api/v1/health                -> {ok, provider, model_id, queue_depth, version}
    GET    /api/v1/config                -> {provider, model_id, providers_available, ...}
    GET    /                             -> web/index.html (StaticFiles html=True)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.db import Database
from app.queue import (
    JobQueue,
    ProviderError,
    available_providers,
    provider_label,
    resolve_provider,
)
from app.schemas import (
    ACTIVE_STATUSES,
    MEDIA_TYPES,
    TERMINAL_STATUSES,
    ConfigView,
    HealthView,
    JobCreate,
    JobListView,
    JobView,
)

logger = logging.getLogger(__name__)

#: Diretorio do frontend estatico (servido em /).
WEB_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "web"

#: Intervalo de polling do SSE (1 evento por segundo, conforme contrato).
SSE_INTERVAL_SECONDS: Final[float] = 1.0

#: Teto de itens por pagina em GET /api/v1/jobs (limits maiores sao truncados).
MAX_JOBS_PAGE: Final[int] = 200

#: Origens liberadas para desenvolvimento local (a UI de producao e mesma origem).
DEV_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

JOB_NOT_FOUND: Final[str] = "Job nao encontrado."
IMAGE_NOT_FOUND: Final[str] = "Imagem nao encontrada."


def create_app(settings: Settings | None = None) -> FastAPI:
    """Constroi a aplicacao (permite injetar Settings nos testes)."""
    app_settings = settings or get_settings()
    db = Database(app_settings.db_file)
    queue = JobQueue(app_settings, db)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app_settings.ensure_dirs()
        await db.connect()
        await db.init_schema()
        recovered = await db.recover_running_jobs()
        if recovered:
            logger.info("%d job(s) recuperado(s) de crash", recovered)
        await queue.start()
        try:
            yield
        finally:
            await queue.stop()
            await db.close()

    app = FastAPI(
        title="Image Gen",
        version=app_settings.app_version,
        description="Servico de geracao de imagens com fila assincrona (Qwen-Image 2.1).",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.db = db
    app.state.queue = queue

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # erros em pt-BR
    # ------------------------------------------------------------------ #
    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        parts: list[str] = []
        for error in exc.errors():
            location = ".".join(
                str(piece) for piece in error.get("loc", ()) if piece != "body"
            )
            message = error.get("msg", "valor invalido")
            parts.append(f"{location}: {message}" if location else message)
        detail = "Dados invalidos: " + "; ".join(parts) if parts else "Dados invalidos."
        return JSONResponse(status_code=422, content={"detail": detail})

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #
    @app.post("/api/v1/jobs", response_model=JobView, status_code=202)
    async def create_job(payload: JobCreate, request: Request) -> JobView:
        """Cria um job e o coloca na fila. 400 se o provider nao estiver configurado."""
        try:
            provider = resolve_provider(app_settings)
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = uuid.uuid4().hex
        job = await request.app.state.db.create_job(
            job_id,
            payload,
            provider=str(getattr(provider, "name", app_settings.image_provider)),
            model_id=str(
                getattr(
                    provider,
                    "model_id",
                    app_settings.model_id_for(app_settings.image_provider),
                )
            ),
        )
        request.app.state.queue.enqueue(job_id)
        return job

    @app.get("/api/v1/jobs", response_model=JobListView)
    async def list_jobs(
        request: Request,
        limit: int = Query(default=20),
        offset: int = Query(default=0),
    ) -> JobListView:
        """Historico de jobs, mais recentes primeiro.

        ``limit``/``offset`` sao tolerantes: valores fora da faixa sao
        normalizados (1..MAX_JOBS_PAGE e >= 0) em vez de virar 422.
        """
        page = max(1, min(int(limit), MAX_JOBS_PAGE))
        skip = max(0, int(offset))
        items, total = await request.app.state.db.list_jobs(limit=page, offset=skip)
        return JobListView(items=items, total=total)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobView)
    async def get_job(job_id: str, request: Request) -> JobView:
        """Estado atual de um job."""
        job = await request.app.state.db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)
        return job

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        """SSE com um evento "status" por segundo ate o status ser terminal."""
        db: Database = request.app.state.db
        if await db.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)

        async def stream() -> AsyncIterator[str]:
            while True:
                job = await db.get_job(job_id)
                if job is None:
                    payload = json.dumps(
                        {"job_id": job_id, "error": JOB_NOT_FOUND}, ensure_ascii=False
                    )
                    yield f"event: error\ndata: {payload}\n\n"
                    return
                # model_dump_json evita re-encodar datetime/bytes manualmente.
                yield f"event: status\ndata: {job.model_dump_json()}\n\n"
                if job.status in TERMINAL_STATUSES:
                    return
                await asyncio.sleep(SSE_INTERVAL_SECONDS)
                if await request.is_disconnected():
                    logger.debug("Cliente desconectou do SSE do job %s", job_id)
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str, request: Request) -> Response:
        """Cancela o job (se queued/running) e apaga as imagens dele."""
        db: Database = request.app.state.db
        job = await db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)

        await request.app.state.queue.cancel(job_id)
        removed = await db.delete_images_for_job(job_id)
        for record in removed:
            await asyncio.to_thread(_unlink_quietly, Path(record.path))
        return Response(status_code=204)

    # ------------------------------------------------------------------ #
    # imagens
    # ------------------------------------------------------------------ #
    @app.get("/api/v1/images/{image_id}")
    async def get_image(image_id: str, request: Request) -> FileResponse:
        """Bytes da imagem, com Content-Type do formato e download como attachment."""
        record = await request.app.state.db.get_image(image_id)
        if record is None:
            raise HTTPException(status_code=404, detail=IMAGE_NOT_FOUND)

        path = Path(record.path)
        images_root = app_settings.images_path.resolve()
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - caminho invalido
            raise HTTPException(status_code=404, detail=IMAGE_NOT_FOUND) from None
        if not resolved.is_relative_to(images_root) or not resolved.is_file():
            logger.error(
                "Imagem %s com arquivo ausente ou fora de %s", image_id, images_root
            )
            raise HTTPException(status_code=404, detail=IMAGE_NOT_FOUND)

        media_type = MEDIA_TYPES.get(record.format, "application/octet-stream")
        return FileResponse(
            resolved,
            media_type=media_type,
            filename=f"{image_id}.{record.format}",
            headers={
                "Content-Disposition": f'attachment; filename="{image_id}.{record.format}"'
            },
        )

    # ------------------------------------------------------------------ #
    # diagnostico
    # ------------------------------------------------------------------ #
    @app.get("/api/v1/health", response_model=HealthView)
    async def health(request: Request) -> HealthView:
        """Status do servico (sempre 200: e o healthcheck do container)."""
        provider_name, model_id = provider_label(app_settings)
        # Profundidade da fila = jobs queued + running (fonte de verdade: SQLite).
        depth = await request.app.state.db.count_jobs(ACTIVE_STATUSES)
        return HealthView(
            ok=True,
            provider=provider_name,
            model_id=model_id,
            queue_depth=depth,
            version=app_settings.app_version,
        )

    @app.get("/api/v1/config", response_model=ConfigView)
    async def config() -> ConfigView:
        """Configuracao efetiva exposta para a UI."""
        provider_name, model_id = provider_label(app_settings)
        return ConfigView(
            provider=provider_name,
            model_id=model_id,
            providers_available=available_providers(app_settings),
            min_size=app_settings.min_size,
            max_size=app_settings.max_size,
            job_timeout_seconds=app_settings.job_timeout_seconds,
        )

    # ------------------------------------------------------------------ #
    # frontend estatico (por ultimo: nao pode sombrear /api/*)
    # ------------------------------------------------------------------ #
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:  # pragma: no cover - o frontend e escrito separadamente
        logger.warning(
            "Diretorio web/ nao encontrado em %s; interface nao sera servida", WEB_DIR
        )

        @app.get("/")
        async def index() -> JSONResponse:
            return JSONResponse(
                status_code=200,
                content={
                    "detail": "Interface nao instalada (web/ ausente). API em /api/v1."
                },
            )

    return app


def _unlink_quietly(path: Path) -> None:
    """Remove um arquivo ignorando ausencia/erro (best effort)."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - permissao/IO
        logger.warning("Nao foi possivel remover %s: %s", path, exc)


app = create_app()
