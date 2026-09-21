"""Fila de jobs (worker asyncio FIFO) e resolucao de providers.

O SQLite e a fonte de verdade do status; a fila em memoria so carrega ids.
Um job e "reivindicado" lendo o status no banco no momento do processamento
(se nao estiver mais 'queued', o worker descarta o id) -- assim o
cancelamento de um job ainda nao iniciado nao precisa remover nada da fila.

Concorrencia = MAX_PARALLEL_JOBS (default 1) workers, cada um executando um
job por vez. Timeout = JOB_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Final

from app.config import Settings
from app.db import Database, ImageRecord, utcnow_iso
from app.schemas import JobCreate

if TYPE_CHECKING:  # pragma: no cover - apenas para type checkers
    from app.providers.base import GeneratedImage, ImageProvider, ProgressFn

logger = logging.getLogger(__name__)

try:  # pragma: no cover - caminho normal quando app/providers existe
    from app.providers.base import ProviderError
except ModuleNotFoundError:  # pragma: no cover - providers ainda nao instalados

    class ProviderError(Exception):  # type: ignore[no-redef]
        """Shim temporario: o provider real vive em app/providers/base.py."""


#: Intervalo minimo entre escritas de progresso no SQLite.
PROGRESS_INTERVAL_SECONDS: Final[float] = 0.5

#: Extensao de arquivo por formato de imagem.
FORMAT_EXTENSIONS: Final[dict[str, str]] = {
    "png": "png",
    "jpeg": "jpeg",
    "webp": "webp",
}

#: Mensagem quando o provider nao devolve nenhuma imagem.
EMPTY_RESULT_MESSAGE: Final[str] = "O provider nao retornou nenhuma imagem."


def resolve_provider(
    settings: Settings, *, require_available: bool = True
) -> ImageProvider:
    """Instancia o provider configurado, traduzindo falhas em ``ProviderError``.

    Com ``require_available=True`` (default) o provider indisponivel vira
    ProviderError -> HTTP 400 no POST /api/v1/jobs (fail fast do contrato).
    O import e tardio para que o core possa ser importado/testado sozinho.
    """
    try:
        from app.providers import get_provider
    except ModuleNotFoundError as exc:  # pragma: no cover - defensivo
        raise ProviderError(
            f"Modulo de providers indisponivel (app/providers): {exc}"
        ) from exc
    try:
        return get_provider(settings, require_available=require_available)
    except ProviderError:
        raise
    except NotImplementedError as exc:
        raise ProviderError(
            f"Provider '{settings.image_provider}' nao implementado: {exc}"
        ) from exc


def available_providers(settings: Settings) -> list[str]:
    """Providers configurados agora (delegado ao pacote app/providers)."""
    try:
        from app.providers import providers_available
    except ModuleNotFoundError as exc:  # pragma: no cover - defensivo
        logger.warning("Nao foi possivel listar providers: %s", exc)
        return []
    return list(providers_available(settings))


async def call_generate(
    provider: ImageProvider, request: JobCreate, progress: ProgressFn, job_id: str
) -> list[GeneratedImage]:
    """Chama ``provider.generate`` passando job_id quando ele aceita.

    O contrato define ``generate(req, progress)``; o provider fake (e outros)
    aceitam um terceiro parametro opcional ``job_id`` -- usado para rotular a
    imagem. Inspecionamos a assinatura para nao quebrar quem segue o contrato
    estrito de dois argumentos.
    """
    generate = provider.generate
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C-extension
        parameters = {}
    if "job_id" in parameters:
        return await generate(request, progress, job_id=job_id)
    return await generate(request, progress)


class JobQueue:
    """Fila FIFO com N workers assincronos."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._running: dict[str, asyncio.Task[None]] = {}
        self._stopping = False

    # ------------------------------------------------------------------ #
    # ciclo de vida
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Sobe os workers e re-enfileira o que ficou pendente no banco."""
        await self.stop()
        self._stopping = False

        pending = await self.db.list_job_ids(("queued",))
        for job_id in pending:
            self._queue.put_nowait(job_id)
        if pending:
            logger.info("Re-enfileirando %d job(s) pendente(s)", len(pending))

        workers = max(1, self.settings.max_parallel_jobs)
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"job-worker-{index}")
            for index in range(workers)
        ]
        logger.info("Fila iniciada com %d worker(s)", workers)

    async def stop(self) -> None:
        """Cancela workers e jobs em execucao (idempotente).

        Jobs interrompidos ficam com status 'running' no banco e voltam para
        'queued' na proxima inicializacao (recuperacao de crash).
        """
        self._stopping = True
        tasks = list(self._workers) + list(self._running.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._running.clear()

    def enqueue(self, job_id: str) -> None:
        """Coloca um job no fim da fila."""
        self._queue.put_nowait(job_id)
        logger.debug(
            "Job %s enfileirado (profundidade=%d)", job_id, self._queue.qsize()
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancela um job queued/running. True se algo foi cancelado."""
        running_task = self._running.get(job_id)
        if running_task is not None:
            # Marca antes de cancelar para que o worker nao sobrescreva o status.
            cancelled = await self.db.cancel_job(job_id)
            running_task.cancel()
            return cancelled
        return await self.db.cancel_job(job_id)

    # ------------------------------------------------------------------ #
    # worker
    # ------------------------------------------------------------------ #
    async def _worker(self, index: int) -> None:
        """Loop do worker: pega um id da fila e executa o job em uma task propria.

        O job roda em uma task separada para que cancelar um job nao cancele o
        worker (com MAX_PARALLEL_JOBS=1 isso mataria a unica thread de trabalho).
        ``asyncio.wait`` observa a task sem propagar o cancelamento dela.
        """
        logger.debug("Worker %d iniciado", index)
        while True:
            job_id = await self._queue.get()
            job_task = asyncio.create_task(self._process(job_id), name=f"job-{job_id}")
            self._running[job_id] = job_task
            try:
                await asyncio.wait({job_task})
            except asyncio.CancelledError:
                # Shutdown: interrompe o job em andamento e sai.
                job_task.cancel()
                await asyncio.gather(job_task, return_exceptions=True)
                logger.debug("Worker %d encerrado", index)
                raise
            except Exception:
                logger.exception("Erro inesperado ao processar o job %s", job_id)
                await self._mark_failed_safely(job_id, "Erro inesperado no worker.")
            else:
                if not job_task.cancelled():
                    error = job_task.exception()
                    if error is not None:  # pragma: no cover - _process trata tudo
                        logger.error(
                            "Job %s terminou com erro nao tratado: %r", job_id, error
                        )
            finally:
                self._running.pop(job_id, None)
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = await self.db.get_job(job_id)
        if job is None:
            logger.debug("Job %s nao existe mais; descartando", job_id)
            return
        if job.status != "queued":
            logger.debug(
                "Job %s nao esta mais 'queued' (%s); descartando", job_id, job.status
            )
            return

        # O registro em self._running e responsabilidade do worker (dono da task).
        provider_name = job.provider
        try:
            await self.db.mark_status(
                job_id,
                "running",
                progress=0,
                message="Iniciando a geracao...",
                error=None,
                started_at=utcnow_iso(),
            )

            provider = resolve_provider(self.settings)
            provider_name = getattr(provider, "name", job.provider)
            progress = self._make_progress_reporter(job_id)
            timeout = self.settings.job_timeout_seconds

            images = await asyncio.wait_for(
                call_generate(provider, job.request, progress, job_id), timeout=timeout
            )

            if not images:
                raise ProviderError(EMPTY_RESULT_MESSAGE)

            saved: list[ImageRecord] = []
            for image in images:
                saved.append(await self._save_image(job_id, image))

            await self.db.mark_status(
                job_id,
                "succeeded",
                progress=100,
                message=f"{len(saved)} imagem(ns) gerada(s)",
                finished_at=utcnow_iso(),
            )
            logger.info("Job %s concluido com %d imagem(ns)", job_id, len(saved))

        except asyncio.CancelledError:
            logger.info("Job %s cancelado", job_id)
            if not self._stopping:
                # Cancelamento pedido pelo usuario: garante o status no banco.
                try:
                    await asyncio.shield(self._mark_cancelled_safely(job_id))
                except asyncio.CancelledError:  # pragma: no cover - best effort
                    pass
            raise
        except (asyncio.TimeoutError, TimeoutError):
            message = (
                f"Tempo limite de {self.settings.job_timeout_seconds}s excedido "
                f"(provider: {provider_name})."
            )
            logger.warning("Job %s excedeu o timeout", job_id)
            await self._fail(job_id, message)
        except ProviderError as exc:
            logger.warning(
                "Job %s falhou no provider %s: %s", job_id, provider_name, exc
            )
            await self._fail(job_id, str(exc))
        except Exception as exc:
            logger.exception("Job %s falhou", job_id)
            await self._fail(job_id, f"Erro inesperado ao gerar a imagem: {exc}")

    # ------------------------------------------------------------------ #
    # helpers internos
    # ------------------------------------------------------------------ #
    def _make_progress_reporter(self, job_id: str) -> ProgressFn:
        """Callback de progresso com throttle (0.5 s) para nao inundar o SQLite."""
        state = {"value": -1, "last_write": 0.0}

        async def report(value: int, message: str) -> None:
            try:
                percent = max(0, min(100, int(value)))
            except (TypeError, ValueError):
                return
            now = asyncio.get_running_loop().time()
            is_endpoint = percent in (0, 100)
            if percent == state["value"] and not is_endpoint:
                return
            if (
                not is_endpoint
                and (now - state["last_write"]) < PROGRESS_INTERVAL_SECONDS
            ):
                return
            state["value"] = percent
            state["last_write"] = now
            await self.db.set_progress(job_id, percent, message or None)

        return report

    async def _save_image(self, job_id: str, image: GeneratedImage) -> ImageRecord:
        """Grava os bytes em IMAGES_DIR/<image_id>.<fmt> e insere a linha."""
        image_id = uuid.uuid4().hex
        fmt = str(getattr(image, "format", "png") or "png").lower().lstrip(".")
        if fmt == "jpg":
            fmt = "jpeg"
        extension = FORMAT_EXTENSIONS.get(fmt, "png")
        if fmt not in FORMAT_EXTENSIONS:
            fmt = extension

        path = self.settings.images_path / f"{image_id}.{extension}"
        data = bytes(image.data)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

        width = int(getattr(image, "width", 0) or 0)
        height = int(getattr(image, "height", 0) or 0)
        record = ImageRecord(
            image_id=image_id,
            job_id=job_id,
            path=str(path),
            width=width,
            height=height,
            bytes=len(data),
            seed=getattr(image, "seed", None),
            format=fmt,
            created_at=utcnow_iso(),
        )
        return await self.db.add_image(record)

    async def _fail(self, job_id: str, message: str) -> None:
        """Marca falha e descarta imagens parciais (estado consistente)."""
        changed = await self.db.mark_status(
            job_id,
            "failed",
            error=(message or "Falha desconhecida.")[:2000],
            finished_at=utcnow_iso(),
        )
        if changed:
            await self._discard_images(job_id)

    async def _mark_failed_safely(self, job_id: str, message: str) -> None:
        try:
            await self._fail(job_id, message)
        except Exception:
            logger.exception("Nao foi possivel marcar o job %s como failed", job_id)

    async def _mark_cancelled_safely(self, job_id: str) -> None:
        try:
            await self.db.cancel_job(job_id)
        except Exception:
            logger.exception("Nao foi possivel marcar o job %s como cancelled", job_id)

    async def _discard_images(self, job_id: str) -> None:
        records = await self.db.delete_images_for_job(job_id)
        for record in records:
            await asyncio.to_thread(_unlink_quietly, Path(record.path))


def _unlink_quietly(path: Path) -> None:
    """Remove um arquivo ignorando ausencia/erro (best effort)."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - permissao/IO
        logger.warning("Nao foi possivel remover %s: %s", path, exc)


def provider_label(settings: Settings) -> tuple[str, str]:
    """(nome canonico, model_id) do provider configurado, sem exigir config valida.

    Usado por /health e /config: mesmo sem chave/URL o endpoint de diagnostico
    responde (devolvendo o model_id estatico das settings) em vez de estourar.
    """
    name = settings.image_provider
    try:
        provider = resolve_provider(settings, require_available=False)
    except Exception as exc:  # noqa: BLE001 - endpoint de diagnostico nunca falha
        logger.debug("Provider '%s' indisponivel para label: %s", name, exc)
        return name, settings.model_id_for(name)
    return (
        str(getattr(provider, "name", name) or name),
        str(getattr(provider, "model_id", "") or settings.model_id_for(name)),
    )
