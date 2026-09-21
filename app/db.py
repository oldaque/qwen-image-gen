"""Persistencia SQLite (aiosqlite) -- fonte de verdade do status dos jobs.

Uma unica conexao e compartilhada pela aplicacao (aiosqlite serializa as
chamadas em uma thread propria); um ``asyncio.Lock`` protege as escritas
compostas para manter as transicoes de status atomicas no processo.

Schema (criado no boot, idempotente):

    jobs(job_id PK, status, progress, progress_message, error, provider,
         model_id, request_json, created_at, started_at, finished_at)
    images(image_id PK, job_id FK -> jobs.job_id, path, width, height, bytes,
           seed, format, created_at)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import aiosqlite

from app.schemas import ImageView, JobCreate, JobView

logger = logging.getLogger(__name__)

#: Sentinela para distinguir "nao alterar" de "gravar NULL".
_UNSET: Final = object()

SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        progress_message TEXT,
        error TEXT,
        provider TEXT NOT NULL,
        model_id TEXT NOT NULL,
        request_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS images (
        image_id TEXT PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id),
        path TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        bytes INTEGER NOT NULL,
        seed INTEGER,
        format TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_images_job ON images(job_id, created_at)",
)


def utcnow_iso() -> str:
    """Timestamp UTC em ISO-8601 com sufixo ``Z`` (ordenavel lexicograficamente)."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def parse_iso(value: str | None) -> datetime | None:
    """Parse tolerante de timestamp; ``None`` se ausente/invalido."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("timestamp invalido no banco: %r", value)
        return None


@dataclass(slots=True, frozen=True)
class ImageRecord:
    """Linha da tabela ``images``."""

    image_id: str
    job_id: str
    path: str
    width: int
    height: int
    bytes: int
    seed: int | None
    format: str
    created_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> ImageRecord:
        return cls(
            image_id=row["image_id"],
            job_id=row["job_id"],
            path=row["path"],
            width=row["width"],
            height=row["height"],
            bytes=row["bytes"],
            seed=row["seed"],
            format=row["format"],
            created_at=row["created_at"],
        )

    def to_view(self) -> ImageView:
        """Converte para o shape publico (``url`` derivada do id)."""
        return ImageView(
            image_id=self.image_id,
            url=f"/api/v1/images/{self.image_id}",
            width=self.width,
            height=self.height,
            bytes=self.bytes,
            seed=self.seed,
            format=self.format,
        )


class Database:
    """Acesso async ao SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # ciclo de vida
    # ------------------------------------------------------------------ #
    @property
    def connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Banco nao inicializado: chame connect() antes de usar.")
        return self._conn

    async def connect(self) -> None:
        """Abre a conexao e aplica os pragmas."""
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.commit()
        self._conn = conn
        logger.info("SQLite conectado em %s", self._db_path)

    async def close(self) -> None:
        """Fecha a conexao (idempotente)."""
        if self._conn is None:
            return
        conn, self._conn = self._conn, None
        await conn.close()
        logger.info("SQLite fechado (%s)", self._db_path)

    async def init_schema(self) -> None:
        """Cria tabelas e indices no boot."""
        conn = self.connection
        async with self._write_lock:
            for statement in SCHEMA_STATEMENTS:
                await conn.execute(statement)
            await conn.commit()

    async def recover_running_jobs(self) -> int:
        """Recuperacao de crash: jobs 'running' voltam para 'queued'.

        Zera progresso/erro/timestamps de execucao para que o job possa ser
        replays com estado limpo. Retorna quantos jobs foram requeueados.
        """
        conn = self.connection
        async with self._write_lock:
            cursor = await conn.execute(
                """
                UPDATE jobs
                   SET status = 'queued',
                       progress = 0,
                       progress_message = NULL,
                       error = NULL,
                       started_at = NULL,
                       finished_at = NULL
                 WHERE status = 'running'
                """
            )
            await conn.commit()
            count = cursor.rowcount or 0
            await cursor.close()
        if count:
            logger.warning(
                "Recuperacao de crash: %d job(s) 'running' -> 'queued'", count
            )
        return count

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #
    async def create_job(
        self,
        job_id: str,
        request: JobCreate,
        provider: str,
        model_id: str,
        *,
        status: str = "queued",
        created_at: str | None = None,
    ) -> JobView:
        """Insere um job e devolve o JobView correspondente."""
        conn = self.connection
        created = created_at or utcnow_iso()
        async with self._write_lock:
            await conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, progress, progress_message, error, provider,
                    model_id, request_json, created_at, started_at, finished_at
                ) VALUES (?, ?, 0, NULL, NULL, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    job_id,
                    status,
                    provider,
                    model_id,
                    request.model_dump_json(),
                    created,
                ),
            )
            await conn.commit()
        job = await self.get_job(job_id)
        if job is None:  # pragma: no cover - so acontece se o INSERT falhar
            raise RuntimeError(f"Falha ao persistir o job {job_id}.")
        return job

    async def get_job(self, job_id: str) -> JobView | None:
        """Job + imagens, ou ``None`` se nao existir."""
        conn = self.connection
        async with conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self._build_job_view(row)

    async def list_jobs(
        self, limit: int = 20, offset: int = 0
    ) -> tuple[list[JobView], int]:
        """Jobs mais recentes primeiro + total de jobs existentes."""
        conn = self.connection
        async with conn.execute(
            """
            SELECT * FROM jobs
             ORDER BY created_at DESC, rowid DESC
             LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
        items = [await self._build_job_view(row) for row in rows]

        async with conn.execute("SELECT COUNT(*) AS total FROM jobs") as cursor:
            total_row = await cursor.fetchone()
        total = int(total_row["total"]) if total_row else 0
        return items, total

    async def list_job_ids(self, statuses: Sequence[str] = ("queued",)) -> list[str]:
        """Ids de jobs nos status dados, em ordem FIFO (mais antigos primeiro)."""
        placeholders = ", ".join("?" for _ in statuses)
        conn = self.connection
        async with conn.execute(
            f"""
            SELECT job_id FROM jobs
             WHERE status IN ({placeholders})
             ORDER BY created_at ASC, rowid ASC
            """,
            tuple(statuses),
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["job_id"] for row in rows]

    async def count_jobs(self, statuses: Sequence[str]) -> int:
        """Quantos jobs existem nos status dados."""
        placeholders = ", ".join("?" for _ in statuses)
        conn = self.connection
        async with conn.execute(
            f"SELECT COUNT(*) AS total FROM jobs WHERE status IN ({placeholders})",
            tuple(statuses),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["total"]) if row else 0

    async def mark_status(
        self,
        job_id: str,
        status: str,
        *,
        progress: int | None = None,
        message: Any = _UNSET,
        error: Any = _UNSET,
        started_at: Any = _UNSET,
        finished_at: Any = _UNSET,
        allow_terminal_overwrite: bool = False,
    ) -> bool:
        """Transiciona o status do job, respeitando estados terminais.

        Por padrao um job terminal ('succeeded'/'failed'/'cancelled') nao e
        sobrescrito -- isso impede que um cancelamento seja trocado por
        'failed' por uma corrida com o worker.
        """
        fields: dict[str, Any] = {"status": status}
        if progress is not None:
            fields["progress"] = progress
        fields["progress_message"] = message
        fields["error"] = error
        fields["started_at"] = started_at
        fields["finished_at"] = finished_at

        clause = (
            ""
            if allow_terminal_overwrite
            else " AND status NOT IN ('succeeded', 'failed', 'cancelled')"
        )
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in fields.items():
            if value is _UNSET:
                continue
            assignments.append(f"{column} = ?")
            values.append(value)
        values.append(job_id)
        conn = self.connection
        async with self._write_lock:
            cursor = await conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?{clause}",
                values,
            )
            await conn.commit()
            changed = bool(cursor.rowcount)
            await cursor.close()
        return changed

    async def set_progress(
        self, job_id: str, progress: int, message: str | None
    ) -> bool:
        """Atualiza progresso (0..100) e mensagem, so se o job estiver ativo."""
        conn = self.connection
        async with self._write_lock:
            cursor = await conn.execute(
                """
                UPDATE jobs
                   SET progress = ?, progress_message = ?
                 WHERE job_id = ? AND status = 'running'
                """,
                (progress, message, job_id),
            )
            await conn.commit()
            changed = bool(cursor.rowcount)
            await cursor.close()
        return changed

    async def cancel_job(self, job_id: str, *, finished_at: str | None = None) -> bool:
        """Marca como 'cancelled' se o job estiver queued/running."""
        conn = self.connection
        async with self._write_lock:
            cursor = await conn.execute(
                """
                UPDATE jobs
                   SET status = 'cancelled',
                       progress_message = 'Cancelado pelo usuario',
                       finished_at = COALESCE(?, finished_at)
                 WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (finished_at or utcnow_iso(), job_id),
            )
            await conn.commit()
            changed = bool(cursor.rowcount)
            await cursor.close()
        return changed

    # ------------------------------------------------------------------ #
    # imagens
    # ------------------------------------------------------------------ #
    async def add_image(self, record: ImageRecord) -> ImageRecord:
        """Persiste uma imagem (os bytes ja foram gravados em disco)."""
        conn = self.connection
        async with self._write_lock:
            await conn.execute(
                """
                INSERT INTO images (
                    image_id, job_id, path, width, height, bytes, seed, format, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.image_id,
                    record.job_id,
                    record.path,
                    record.width,
                    record.height,
                    record.bytes,
                    record.seed,
                    record.format,
                    record.created_at,
                ),
            )
            await conn.commit()
        return record

    async def get_image(self, image_id: str) -> ImageRecord | None:
        """Uma imagem pelo id, ou ``None``."""
        conn = self.connection
        async with conn.execute(
            "SELECT * FROM images WHERE image_id = ?", (image_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return ImageRecord.from_row(row) if row is not None else None

    async def list_images(self, job_id: str) -> list[ImageRecord]:
        """Imagens de um job em ordem de criacao."""
        conn = self.connection
        async with conn.execute(
            "SELECT * FROM images WHERE job_id = ? ORDER BY created_at ASC, rowid ASC",
            (job_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [ImageRecord.from_row(row) for row in rows]

    async def delete_images_for_job(self, job_id: str) -> list[ImageRecord]:
        """Remove as linhas de imagem do job e devolve o que foi removido."""
        records = await self.list_images(job_id)
        if not records:
            return []
        conn = self.connection
        async with self._write_lock:
            await conn.execute("DELETE FROM images WHERE job_id = ?", (job_id,))
            await conn.commit()
        return records

    # ------------------------------------------------------------------ #
    # internos
    # ------------------------------------------------------------------ #
    async def _build_job_view(self, row: aiosqlite.Row) -> JobView:
        request = JobCreate.model_validate_json(row["request_json"])
        images = await self.list_images(row["job_id"])
        started_at = row["started_at"]
        finished_at = row["finished_at"]
        return JobView(
            job_id=row["job_id"],
            status=row["status"],
            progress=int(row["progress"] or 0),
            progress_message=row["progress_message"],
            error=row["error"],
            provider=row["provider"],
            model_id=row["model_id"],
            request=request,
            images=[record.to_view() for record in images],
            created_at=row["created_at"],
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=_elapsed_seconds(started_at, finished_at),
        )


def _elapsed_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    """Segundos entre inicio e fim; se ainda rodando, ate agora."""
    start = parse_iso(started_at)
    if start is None:
        return None
    end = parse_iso(finished_at) or datetime.now(timezone.utc)
    return round(max(0.0, (end - start).total_seconds()), 3)
