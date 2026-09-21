"""Fabrica de providers de geracao de imagem.

Contrato v1:
    get_provider(settings: Settings) -> ImageProvider

Valores de IMAGE_PROVIDER (env): fake | local_comfy | fal | remote_gpu  (default: fake).
`providers_available(settings)` lista os providers CONFIGURADOS (usado por /api/v1/config).

Fail-fast: `get_provider` levanta ProviderError quando o provider escolhido esta
indisponivel (sem FAL_KEY, sem REMOTE_GPU_URL, sem COMFY_URL...), e a API HTTP
traduz isso em 400 no POST /api/v1/jobs.
"""

from __future__ import annotations

from app.config import Settings
from app.providers import fake as fake_module
from app.providers import fal as fal_module
from app.providers import local_comfy as local_comfy_module
from app.providers import remote_gpu as remote_gpu_module
from app.providers.base import (
    GeneratedImage,
    ImageProvider,
    ProgressFn,
    ProviderError,
    decode_base64_image,
    image_meta,
    mime_for_format,
    normalize_format,
    validate_reference_images,
)
from app.providers.fake import FakeProvider
from app.providers.fal import FalProvider
from app.providers.local_comfy import LocalComfyProvider
from app.providers.remote_gpu import RemoteGPUProvider

__all__ = [
    # contrato
    "get_provider",
    "providers_available",
    # extras uteis
    "PROVIDER_NAMES",
    "AVAILABLE_CHECKERS",
    "normalize_provider_name",
    "provider_status",
    "is_provider_available",
    # base
    "GeneratedImage",
    "ImageProvider",
    "ProgressFn",
    "ProviderError",
    "decode_base64_image",
    "image_meta",
    "mime_for_format",
    "normalize_format",
    "validate_reference_images",
    # implementacoes
    "FakeProvider",
    "FalProvider",
    "LocalComfyProvider",
    "RemoteGPUProvider",
]

#: Ordem canonica (tambem a ordem exibida na UI).
PROVIDER_NAMES: tuple[str, ...] = ("fake", "local_comfy", "fal", "remote_gpu")

#: nome -> classe concreta
PROVIDER_CLASSES: dict[str, type] = {
    "fake": FakeProvider,
    "local_comfy": LocalComfyProvider,
    "fal": FalProvider,
    "remote_gpu": RemoteGPUProvider,
}

#: nome -> funcao is_available(settings) -> (ok, motivo)
AVAILABLE_CHECKERS: dict[str, object] = {
    "fake": fake_module.is_available,
    "local_comfy": local_comfy_module.is_available,
    "fal": fal_module.is_available,
    "remote_gpu": remote_gpu_module.is_available,
}

#: Apelidos tolerados em IMAGE_PROVIDER (ex.: "local_cpu" -> local_comfy).
_PROVIDER_ALIASES: dict[str, str] = {
    "": "fake",
    "none": "fake",
    "offline": "fake",
    "mock": "fake",
    "comfy": "local_comfy",
    "comfyui": "local_comfy",
    "local": "local_comfy",
    "local_cpu": "local_comfy",
    "local_comfyui": "local_comfy",
    "fal_ai": "fal",
    "falai": "fal",
    "gpu": "remote_gpu",
    "remote": "remote_gpu",
    "vllm": "remote_gpu",
}


def normalize_provider_name(name: str | None) -> str:
    """Canoniza o valor de IMAGE_PROVIDER (aceita apelidos)."""
    value = (name or "").strip().lower().replace("-", "_")
    if value in PROVIDER_CLASSES:
        return value
    if value in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[value]
    raise ProviderError(
        f"IMAGE_PROVIDER invalido: {name!r}. Use um de: {', '.join(PROVIDER_NAMES)}"
    )


def is_provider_available(name: str, settings: Settings) -> tuple[bool, str]:
    """(disponivel, motivo) para um nome (aceita apelidos)."""
    canonical = normalize_provider_name(name)
    checker = AVAILABLE_CHECKERS.get(canonical)
    if checker is None:  # pragma: no cover - defensivo
        return False, f"provider {canonical} desconhecido"
    return checker(settings)  # type: ignore[operator]


def provider_status(settings: Settings) -> dict[str, dict[str, object]]:
    """Mapa nome -> {available, reason, model_id} (para /api/v1/config e /health)."""
    status: dict[str, dict[str, object]] = {}
    for name in PROVIDER_NAMES:
        ok, reason = is_provider_available(name, settings)
        model_id: str | None = None
        try:
            model_id = str(getattr(PROVIDER_CLASSES[name](settings), "model_id", "") or "") or None
        except Exception:  # noqa: BLE001 - so metadados
            model_id = None
        status[name] = {"available": ok, "reason": reason, "model_id": model_id}
    return status


def providers_available(settings: Settings) -> list[str]:
    """Nomes dos providers configurados (ordem canonica)."""
    return [name for name in PROVIDER_NAMES if is_provider_available(name, settings)[0]]


def get_provider(settings: Settings, *, require_available: bool = True) -> ImageProvider:
    """Instancia o provider de IMAGE_PROVIDER.

    Com require_available=True (default) levanta ProviderError se ele nao estiver
    configurado — e o fail-fast pedido pelo contrato (POST /api/v1/jobs -> 400).
    """
    name = normalize_provider_name(getattr(settings, "image_provider", "fake"))
    if require_available:
        ok, reason = is_provider_available(name, settings)
        if not ok:
            raise ProviderError(f"provider {name} indisponivel: {reason}", provider=name)
    return PROVIDER_CLASSES[name](settings)  # type: ignore[return-value]
