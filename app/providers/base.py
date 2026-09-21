"""Contratos e utilitarios da camada de providers de geracao de imagem.

CONTRATO v1 (nomes fixos, nao renomear):
    GeneratedImage, ProgressFn, ProviderError, ImageProvider

Todos os providers devolvem `list[GeneratedImage]` com os BYTES ja baixados
(a camada de fila/API e que grava em disco em IMAGES_DIR).
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas import JobCreate

__all__ = [
    "GeneratedImage",
    "ProgressFn",
    "ProviderError",
    "ImageProvider",
    "MIME_BY_FORMAT",
    "mime_for_format",
    "normalize_format",
    "image_meta",
    "decode_base64_image",
    "validate_reference_images",
    "noop_progress",
]


# --------------------------------------------------------------------------
# Formatos / mimetypes
# --------------------------------------------------------------------------

MIME_BY_FORMAT: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}

_FORMAT_ALIASES = {
    "jpg": "jpeg",
    "jpe": "jpeg",
    "jfif": "jpeg",
    "tif": "tiff",
}

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<params>(?:;[^,]*)*),(?P<payload>.*)$", re.DOTALL)


def normalize_format(value: str | None, default: str = "png") -> str:
    """Normaliza 'JPG', '.jpeg', 'image/png' etc. para 'png' | 'jpeg' | 'webp'."""
    if not value:
        return default
    fmt = str(value).strip().lower()
    if "/" in fmt:
        fmt = fmt.rsplit("/", 1)[-1]
    fmt = fmt.lstrip(".").strip()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    return fmt or default


def mime_for_format(value: str | None, default: str = "image/png") -> str:
    """Content-Type para um formato de imagem ('png' -> 'image/png')."""
    return MIME_BY_FORMAT.get(normalize_format(value), default)


# --------------------------------------------------------------------------
# Contratos
# --------------------------------------------------------------------------


class GeneratedImage(BaseModel):
    """Uma imagem gerada, ja em memoria."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: bytes
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seed: int | None = None
    format: str = "png"

    @field_validator("format", mode="before")
    @classmethod
    def _normalize_format(cls, v: object) -> str:
        return normalize_format(v if isinstance(v, str) else None)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        format: str | None = None,
    ) -> GeneratedImage:
        """Monta um GeneratedImage; se width/height/format faltarem, le do binario."""
        if width and height and format:
            return cls(data=data, width=width, height=height, seed=seed, format=format)
        real_w, real_h, real_fmt = image_meta(data)
        return cls(
            data=data,
            width=width or real_w,
            height=height or real_h,
            seed=seed,
            format=normalize_format(format or real_fmt),
        )


ProgressFn = Callable[[int, str], Awaitable[None]]
"""Assinatura do callback de progresso: await progress(percentual 0..100, mensagem)."""


class ProviderError(Exception):
    """Erro de provider (config ausente, HTTP, resposta invalida, offline...).

    A API HTTP traduz isso em 400/502 com `str(exc)` no campo `detail`.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else "ProviderError"


@runtime_checkable
class ImageProvider(Protocol):
    """Interface de provider conforme o contrato v1."""

    name: str
    model_id: str

    async def generate(self, req: JobCreate, progress: ProgressFn) -> list[GeneratedImage]:
        ...


# --------------------------------------------------------------------------
# Utilitarios compartilhados pelos providers
# --------------------------------------------------------------------------


async def noop_progress(percent: int, message: str) -> None:
    """ProgressFn que nao faz nada (util em testes/manual)."""
    return None


def image_meta(data: bytes) -> tuple[int, int, str]:
    """(width, height, format) lidos do proprio binario via Pillow."""
    if not data:
        raise ProviderError("imagem vazia (0 bytes) recebida do provider")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = normalize_format(im.format, default="png")
            return int(im.width), int(im.height), fmt
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - qualquer falha do Pillow
        raise ProviderError(f"resposta do provider nao e uma imagem valida: {exc}") from exc


def decode_base64_image(payload: str) -> bytes:
    """Decodifica base64 (aceita data URI e base64 sem padding)."""
    if not payload:
        raise ProviderError("payload base64 vazio na resposta do provider")
    raw = payload.strip()
    match = _DATA_URI_RE.match(raw)
    if match:
        raw = match.group("payload")
    raw = re.sub(r"\s+", "", raw)
    raw += "=" * (-len(raw) % 4)
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(f"base64 invalido na resposta do provider: {exc}") from exc
    if not data:
        raise ProviderError("base64 decodificou para 0 bytes")
    return data


def validate_reference_images(refs: list[str]) -> list[str]:
    """Aceita apenas URLs http(s) ou data URI base64 (ordem preservada)."""
    out: list[str] = []
    for index, ref in enumerate(refs or []):
        value = (ref or "").strip()
        if not value:
            raise ProviderError(f"reference_images[{index}] esta vazia")
        if value.startswith(("http://", "https://", "data:")):
            out.append(value)
            continue
        raise ProviderError(
            f"reference_images[{index}] deve ser URL http(s) ou data URI base64 "
            f"(recebido: {value[:48]!r})"
        )
    return out
