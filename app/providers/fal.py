"""Provider `fal` — fal.ai (provider gerenciado, sem GPU local, servindo o Qwen-Image-2.1).

Contrato da API:
  endpoint t2i  : POST https://fal.run/alibaba/qwen-image-2.1/text-to-image
  endpoint edit : POST https://fal.run/alibaba/qwen-image-2.1/edit
  header        : "Authorization: Key $FAL_KEY"  +  Content-Type: application/json
  body t2i      : prompt, prompt_expander(none|quality), image_size{width,height},
                  negative_prompt, seed, num_images(1..4), output_format(png|jpeg|webp),
                  enable_safety_checker (default true -> mandamos false)
  body edit     : prompt + image_urls (lista ORDENADA, max 10) + output_format
  preco         : $0.02 por megapixel de saida (t2i) / ~$0.0733 por MP in+out (edit)

IMPORTANTE: fal.run e SINCRONO — o POST bloqueia ate a imagem sair. O timeout de
leitura e JOB_TIMEOUT_SECONDS (default 7200 s). O shape exato do JSON de resposta
nao esta documentado campo a campo; `_extract_images` aceita as variantes
conhecidas (images[] / image{} / data.images[] / file_data) e, se nenhuma casar,
levanta ProviderError com o corpo truncado.
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
    validate_reference_images,
)
from app.schemas import JobCreate

__all__ = ["FalProvider", "Provider", "is_available", "FAL_BASE_URL"]

FAL_BASE_URL = "https://fal.run"
#: Teto de referencias do schema de edit (maxItems=10 no OpenAPI).
MAX_REFERENCE_IMAGES = 10


def is_available(settings: Settings) -> tuple[bool, str]:
    """Disponivel quando FAL_KEY esta preenchida."""
    if not (getattr(settings, "fal_key", "") or "").strip():
        return False, "FAL_KEY nao configurada (defina FAL_KEY no .env)"
    return True, "FAL_KEY configurada"


class FalProvider:
    """Cliente fal.ai (t2i e edit)."""

    name = "fal"

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.t2i_model = settings.fal_model
        self.edit_model = settings.fal_edit_model
        #: Injetavel para testes (httpx.MockTransport). None = rede real.
        self._transport = transport
        # model_id padrao = t2i (usado em /health e /config)
        self.model_id = self.t2i_model

    # ------------------------------------------------------------------
    def model_id_for_request(self, req: JobCreate) -> str:
        """Edit model quando ha reference_images, senao o t2i model."""
        return self.edit_model if req.reference_images else self.t2i_model

    def _api_key(self) -> str:
        key = (getattr(self.settings, "fal_key", "") or "").strip()
        if not key:
            raise ProviderError(
                "fal indisponivel: FAL_KEY nao configurada (defina FAL_KEY no .env)",
                provider=self.name,
            )
        return key

    def _timeout(self) -> httpx.Timeout:
        job_timeout = float(getattr(self.settings, "job_timeout_seconds", 7200) or 7200)
        return httpx.Timeout(connect=10.0, read=job_timeout, write=120.0, pool=10.0)

    # ------------------------------------------------------------------
    # API do protocolo ImageProvider
    # ------------------------------------------------------------------
    async def generate(self, req: JobCreate, progress: ProgressFn) -> list[GeneratedImage]:
        key = self._api_key()
        model = self.model_id_for_request(req)
        payload = self._build_payload(req, model)

        await progress(
            2,
            f"enviando para fal ({model}) — endpoint sincrono, pode levar de segundos a minutos",
        )

        async with httpx.AsyncClient(
            timeout=self._timeout(), follow_redirects=True, transport=self._transport
        ) as client:
            data = await self._post(client, model, payload, key)

            items = self._extract_images(data, model)
            await progress(60, f"fal respondeu: {len(items)} imagem(ns); baixando bytes")

            images: list[GeneratedImage] = []
            for index, item in enumerate(items):
                raw = await self._materialize(client, item, model)
                width = item.get("width") if isinstance(item, dict) else None
                height = item.get("height") if isinstance(item, dict) else None
                fmt = self._item_format(item)  # None quando a resposta nao informa
                if not (width and height):
                    # sem dimensoes na resposta: os bytes (Pillow) sao a fonte da verdade
                    width, height, fmt = image_meta(raw)
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
                await progress(percent, f"imagem {index + 1}/{len(items)} recebida do fal")

        await progress(100, f"fal: {len(images)} imagem(ns) gerada(s) com {model}")
        return images

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------
    def _build_payload(self, req: JobCreate, model: str) -> dict[str, Any]:
        if req.reference_images:
            refs = validate_reference_images(req.reference_images)
            if len(refs) > MAX_REFERENCE_IMAGES:
                raise ProviderError(
                    f"fal edit aceita no maximo {MAX_REFERENCE_IMAGES} reference_images "
                    f"(recebido: {len(refs)})",
                    provider=self.name,
                )
            # Campos documentados do schema de edit: prompt, image_urls, output_format.
            # num_images/seed NAO estao documentados nesse schema -> nao enviamos
            # (nem inventamos) para evitar HTTP 422.
            payload: dict[str, Any] = {
                "prompt": req.prompt,
                "image_urls": refs,
                "output_format": req.output_format,
            }
            if req.negative_prompt:
                # nao documentado no edit; enviamos apenas se o usuario pedir
                payload["negative_prompt"] = req.negative_prompt
            return payload

        payload = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt or "",
            "image_size": {"width": int(req.width), "height": int(req.height)},
            "num_images": int(req.num_images),
            "output_format": req.output_format,
            "prompt_expander": req.prompt_expander,
            "enable_safety_checker": False,
        }
        if req.seed is not None:
            payload["seed"] = int(req.seed)
        return payload

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _post(
        self, client: httpx.AsyncClient, model: str, payload: dict[str, Any], key: str
    ) -> dict[str, Any]:
        url = f"{FAL_BASE_URL}/{model}"
        headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"fal: timeout apos {getattr(self.settings, 'job_timeout_seconds', 7200)}s em {url} "
                f"(o endpoint fal.run e sincrono; aumente JOB_TIMEOUT_SECONDS se precisar)",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"fal: falha de rede em {url}: {exc}", provider=self.name) from exc

        if response.status_code >= 400:
            body = self._truncate(response.text)
            if response.status_code in (401, 403):
                raise ProviderError(
                    f"fal: HTTP {response.status_code} — FAL_KEY invalida/sem permissao. {body}",
                    provider=self.name,
                    status_code=response.status_code,
                    body=body,
                )
            raise ProviderError(
                f"fal: HTTP {response.status_code} em {url}. {body}",
                provider=self.name,
                status_code=response.status_code,
                body=body,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"fal: resposta nao-JSON em {url}: {self._truncate(response.text)}",
                provider=self.name,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"fal: resposta inesperada ({type(data).__name__})", provider=self.name)
        if data.get("error"):
            raise ProviderError(f"fal: erro no corpo da resposta: {data['error']}", provider=self.name)
        return data

    # ------------------------------------------------------------------
    # Parse da resposta
    # ------------------------------------------------------------------
    def _extract_images(self, data: dict[str, Any], model: str) -> list[dict[str, Any] | str]:
        candidates: Any = None
        for key in ("images", "image", "output", "data"):
            if key in data and data[key]:
                candidates = data[key]
                break

        items: list[dict[str, Any] | str] = []
        if isinstance(candidates, list):
            items = [i for i in candidates if i]
        elif isinstance(candidates, dict):
            # {"data": {"images": [...]}} ou {"image": {...}}
            if isinstance(candidates.get("images"), list):
                items = [i for i in candidates["images"] if i]
            else:
                items = [candidates]
        elif isinstance(candidates, str):
            items = [candidates]

        if not items:
            raise ProviderError(
                f"fal ({model}): resposta sem imagens. Corpo: {self._truncate(json.dumps(data))}",
                provider=self.name,
            )
        return items

    def _item_format(self, item: dict[str, Any] | str) -> str | None:
        """Formato declarado pelo fal na resposta (ou None, para cair no Pillow)."""
        if not isinstance(item, dict):
            return None
        for key in ("output_format", "format", "content_type"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return normalize_format(value)
        file_name = item.get("file_name")
        if isinstance(file_name, str) and "." in file_name:
            return normalize_format(file_name.rsplit(".", 1)[-1])
        return None

    async def _materialize(
        self, client: httpx.AsyncClient, item: dict[str, Any] | str, model: str
    ) -> bytes:
        """Devolve os bytes da imagem: base64 inline ou download da URL."""
        if isinstance(item, str):
            return await self._download(client, item, model)

        for key in ("b64_json", "base64", "image_base64", "b64"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return decode_base64_image(value)

        for key in ("file_data", "data"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("data:"):
                return decode_base64_image(value)

        url = item.get("url") or item.get("image_url")
        if isinstance(url, str) and url:
            return await self._download(client, url, model)

        raise ProviderError(
            f"fal ({model}): item de resposta sem url/b64_json: {self._truncate(json.dumps(item))}",
            provider=self.name,
        )

    async def _download(self, client: httpx.AsyncClient, url: str, model: str) -> bytes:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise ProviderError(f"fal: falha ao baixar a imagem ({url}): {exc}", provider=self.name) from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"fal: HTTP {response.status_code} ao baixar {url}: {self._truncate(response.text)}",
                provider=self.name,
                status_code=response.status_code,
            )
        return response.content

    @staticmethod
    def _truncate(text: str, limit: int = 1500) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"


#: Alias de compatibilidade (nome curto).
Provider = FalProvider
