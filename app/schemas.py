"""Modelos pydantic da API (shapes exatos do contrato).

``JobCreate`` e o corpo de POST /api/v1/jobs. ``JobView`` e a representacao
de um job em todas as respostas (inclusive no evento SSE "status").
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Entrada tolerante (campos extras sao ignorados) e saida exata: o contrato
#: fixa os shapes, mas payloads antigos no SQLite nunca quebram a leitura.
_MODEL_CONFIG = ConfigDict(extra="ignore")

from app.config import get_settings

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
OutputFormat = Literal["png", "jpeg", "webp"]
PromptExpander = Literal["none", "quality"]

#: Status em que o job nao evolui mais.
TERMINAL_STATUSES: Final[tuple[JobStatus, ...]] = ("succeeded", "failed", "cancelled")

#: Status em que o job ainda aceita cancelamento.
ACTIVE_STATUSES: Final[tuple[JobStatus, ...]] = ("queued", "running")

#: Content-Type por formato de saida.
MEDIA_TYPES: Final[dict[str, str]] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class JobCreate(BaseModel):
    """Corpo de POST /api/v1/jobs."""

    model_config = _MODEL_CONFIG

    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_images: int = Field(default=1, ge=1, le=4)
    seed: int | None = None
    output_format: OutputFormat = "png"
    prompt_expander: PromptExpander = "quality"
    reference_images: list[str] = Field(default_factory=list)

    @field_validator("width", "height")
    @classmethod
    def _check_size(cls, value: int, info) -> int:
        settings = get_settings()
        if not (settings.min_size <= value <= settings.max_size):
            raise ValueError(
                f"{info.field_name} precisa estar entre {settings.min_size} e "
                f"{settings.max_size} (recebido {value})."
            )
        return value

    @field_validator("prompt")
    @classmethod
    def _check_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt nao pode ser vazio.")
        return value

    @field_validator("negative_prompt")
    @classmethod
    def _check_negative_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value if value.strip() else None

    @field_validator("reference_images")
    @classmethod
    def _check_references(cls, value: list[str]) -> list[str]:
        allowed_prefixes = ("data:", "http://", "https://")
        for item in value:
            if item.startswith(allowed_prefixes):
                continue
            raise ValueError(
                "reference_images aceita apenas URLs http(s) ou data URI base64 "
                f"(recebido: {item[:60]!r})."
            )
        return value


class ImageView(BaseModel):
    """Uma imagem persistida, como devolvida em ``JobView.images``."""

    model_config = _MODEL_CONFIG

    image_id: str
    url: str
    width: int
    height: int
    bytes: int
    seed: int | None = None
    format: str


class JobView(BaseModel):
    """Representacao de um job. Shape exato do contrato."""

    model_config = _MODEL_CONFIG

    job_id: str
    status: JobStatus
    progress: int
    progress_message: str | None = None
    error: str | None = None
    provider: str
    model_id: str
    request: JobCreate
    images: list[ImageView] = Field(default_factory=list)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None


class JobListView(BaseModel):
    """Resposta de GET /api/v1/jobs."""

    model_config = _MODEL_CONFIG

    items: list[JobView]
    total: int


class HealthView(BaseModel):
    """Resposta de GET /api/v1/health."""

    model_config = _MODEL_CONFIG

    ok: bool
    provider: str
    model_id: str
    queue_depth: int
    version: str


class ConfigView(BaseModel):
    """Resposta de GET /api/v1/config."""

    model_config = _MODEL_CONFIG

    provider: str
    model_id: str
    providers_available: list[str]
    min_size: int
    max_size: int
    job_timeout_seconds: int
