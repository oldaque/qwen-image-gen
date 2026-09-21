"""Provider `remote_gpu` — GPU propria servindo rota OpenAI-compatible.

Recipe de referencia: `vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091`
-> expoe POST /v1/images/generations (vLLM-Omni; SGLang/LightX2V tem rota equivalente).

Corpo enviado (campos do contrato — o shape campo a campo nao esta documentado,
por isso NAO inventamos campos obrigatorios extras):

    {
      "model": $REMOTE_GPU_MODEL,
      "prompt": "...",
      "size": "1024x1024",
      "n": 1,
      "response_format": "b64_json"
    }

`seed` so vai quando o usuario pede (campo extra tolerado/ignorado por servidores
OpenAI-compat; se o servidor reclamar, remova-o).

Resposta aceita: {"data":[{"b64_json"|"url"}]} (padrao OpenAI), {"images":[{"url"}]}
e {"data":{"images":[...]}}. VRAM de referencia: 48 GB bf16 ou 16-20 GB quantizado.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Settings
from app.providers.base import (
    GeneratedImage,
    ProgressFn,
    ProviderError,
    decode_base64_image,
    image_meta,
    normalize_format,
)
from app.schemas import JobCreate

__all__ = ["RemoteGPUProvider", "Provider", "is_available", "build_endpoint"]

DEFAULT_PORT_HINT = 8091


def is_available(settings: Settings) -> tuple[bool, str]:
    """Disponivel quando REMOTE_GPU_URL esta preenchida."""
    if not (getattr(settings, "remote_gpu_url", "") or "").strip():
        return False, "REMOTE_GPU_URL nao configurada (ex.: http://gpu.internal:8091 ou .../v1)"
    return True, "REMOTE_GPU_URL configurada"


def build_endpoint(base_url: str) -> str:
    """Normaliza a base URL para o endpoint /v1/images/generations.

    Aceita:  http://host:8091 | http://host:8091/ | http://host:8091/v1 |
             http://host:8091/v1/images/generations
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ProviderError("remote_gpu: REMOTE_GPU_URL vazia", provider="remote_gpu")
    if base.endswith("/v1/images/generations"):
        return base
    if base.endswith("/v1"):
        return f"{base}/images/generations"
    return f"{base}/v1/images/generations"


class RemoteGPUProvider:
    """Cliente da rota OpenAI-compatible /v1/images/generations (vLLM-Omni/SGLang)."""

    name = "remote_gpu"

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        #: Injetavel para testes (httpx.MockTransport). None = rede real.
        self._transport = transport
        self.model_id = settings.remote_gpu_model

    # ------------------------------------------------------------------
    def _key(self) -> str:
        return (getattr(self.settings, "remote_gpu_key", "") or "").strip()

    def _endpoint(self) -> str:
        url = (getattr(self.settings, "remote_gpu_url", "") or "").strip()
        if not url:
            raise ProviderError(
                "remote_gpu indisponivel: REMOTE_GPU_URL nao configurada "
                f"(ex.: http://gpu.internal:{DEFAULT_PORT_HINT})",
                provider=self.name,
            )
        return build_endpoint(url)

    def _timeout(self) -> httpx.Timeout:
        job_timeout = float(getattr(self.settings, "job_timeout_seconds", 7200) or 7200)
        return httpx.Timeout(connect=10.0, read=job_timeout, write=120.0, pool=10.0)

    # ------------------------------------------------------------------
    # API do protocolo ImageProvider
    # ------------------------------------------------------------------
    async def generate(self, req: JobCreate, progress: ProgressFn) -> list[GeneratedImage]:
        if req.reference_images:
            raise ProviderError(
                "remote_gpu nao suporta reference_images: a rota /v1/images/generations "
                "e text-to-image apenas (edicao por imagem exige outro endpoint)",
                provider=self.name,
            )

        endpoint = self._endpoint()
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": req.prompt,
            "size": f"{int(req.width)}x{int(req.height)}",
            "n": int(req.num_images),
            "response_format": "b64_json",
        }
        if req.seed is not None:
            # campo extra: servidores OpenAI-compat costumam ignorar; nao e critico.
            payload["seed"] = int(req.seed)

        headers = {"Content-Type": "application/json"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        await progress(2, f"enviando para a GPU remota ({endpoint})")

        async with httpx.AsyncClient(
            timeout=self._timeout(), follow_redirects=True, transport=self._transport
        ) as client:
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise ProviderError(
                    f"remote_gpu: timeout apos "
                    f"{getattr(self.settings, 'job_timeout_seconds', 7200)}s em {endpoint}",
                    provider=self.name,
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"remote_gpu: falha de rede em {endpoint}: {exc} — a GPU remota esta no ar?",
                    provider=self.name,
                ) from exc

            if response.status_code >= 400:
                body = self._truncate(response.text)
                hint = ""
                if response.status_code in (401, 403):
                    hint = " (REMOTE_GPU_KEY ausente/invalida)"
                elif response.status_code == 404:
                    hint = f" (confirme que o servidor expoe /v1/images/generations; o recipe e `vllm serve Qwen/Qwen-Image-2.1 --omni --port {DEFAULT_PORT_HINT}`)"
                raise ProviderError(
                    f"remote_gpu: HTTP {response.status_code} em {endpoint}{hint}. {body}",
                    provider=self.name,
                    status_code=response.status_code,
                    body=body,
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError(
                    f"remote_gpu: resposta nao-JSON ({self._truncate(response.text)})",
                    provider=self.name,
                ) from exc

            await progress(60, "GPU remota respondeu; decodificando imagens")

            items = self._extract_items(data, endpoint)
            images: list[GeneratedImage] = []
            for index, item in enumerate(items):
                raw = await self._materialize(client, item, endpoint)
                width = item.get("width")
                height = item.get("height")
                fmt = normalize_format(item.get("output_format") or item.get("format"), default="")
                if not (width and height) or not fmt:
                    det_w, det_h, det_fmt = image_meta(raw)
                    width = width or det_w
                    height = height or det_h
                    fmt = fmt or det_fmt
                images.append(
                    GeneratedImage(
                        data=raw,
                        width=int(width),
                        height=int(height),
                        seed=req.seed,
                        format=normalize_format(fmt or req.output_format),
                    )
                )
                percent = 60 + int(round(35 * (index + 1) / len(items)))
                await progress(percent, f"imagem {index + 1}/{len(items)} decodificada")

        await progress(100, f"remote_gpu: {len(images)} imagem(ns) gerada(s)")
        return images

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------
    def _extract_items(self, data: Any, endpoint: str) -> list[dict[str, Any]]:
        candidates: Any = None
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                candidates = data["data"]
            elif isinstance(data.get("images"), list):
                candidates = data["images"]
            elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("images"), list):
                candidates = data["data"]["images"]
            elif isinstance(data.get("data"), dict):
                candidates = [data["data"]]
            elif isinstance(data.get("image"), dict):
                candidates = [data["image"]]

        items = [i for i in candidates or [] if isinstance(i, dict)]
        if not items:
            raise ProviderError(
                f"remote_gpu: resposta sem imagens em {endpoint}. "
                f"Corpo: {self._truncate(json.dumps(data) if not isinstance(data, str) else data)}",
                provider=self.name,
            )
        return items

    async def _materialize(
        self, client: httpx.AsyncClient, item: dict[str, Any], endpoint: str
    ) -> bytes:
        for key in ("b64_json", "b64", "base64", "image"):
            value = item.get(key)
            if isinstance(value, str) and value and not value.startswith("http"):
                return decode_base64_image(value)

        url = item.get("url")
        if not isinstance(url, str) or not url:
            # variante vista em alguns servidores: {"image": "https://..."}
            candidate = item.get("image") or item.get("image_url")
            url = candidate if isinstance(candidate, str) and candidate.startswith("http") else None
        if isinstance(url, str) and url:
            try:
                response = await client.get(url)
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"remote_gpu: falha ao baixar a imagem ({url}): {exc}", provider=self.name
                ) from exc
            if response.status_code >= 400:
                raise ProviderError(
                    f"remote_gpu: HTTP {response.status_code} ao baixar {url}: "
                    f"{self._truncate(response.text)}",
                    provider=self.name,
                    status_code=response.status_code,
                )
            return response.content

        raise ProviderError(
            f"remote_gpu: item sem b64_json/url: {self._truncate(json.dumps(item))}",
            provider=self.name,
        )

    @staticmethod
    def _truncate(text: str, limit: int = 1500) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"


#: Alias de compatibilidade (nome curto).
Provider = RemoteGPUProvider
