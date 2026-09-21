"""Provider `local_comfy` — ComfyUI local (servico "comfy" do docker-compose, profile local).

Fluxo (rotas do ComfyUI):
    POST /prompt            {"prompt": <API-format>, "client_id": "<uuid4>"} -> {"prompt_id", "number"}
    GET  /history/{id}      -> {"<id>": {"outputs": {"<node>": {"images": [{filename, subfolder, type}]}},
                                          "status": {"messages": [...], "status_str": "..."}}}
    GET  /view?filename=&subfolder=&type=output -> bytes da imagem
    GET  /queue             -> {"queue_running": [...], "queue_pending": [...]}
    POST /interrupt         {"prompt_id": id}  (best-effort no cancelamento)
    GET  /system_stats      -> usado como health-check

PROGRESSO: steps concluidos / total, com ETA medida entre updates de step
("gerando: step 12/40 (~35 min restantes)"). Enquanto o sampler nao comeca, a
mensagem informa o que o ComfyUI esta fazendo (fila, carregando).

TEMPO DE EXECUCAO: em um servidor ARM64 sem GPU a geracao roda em CPU e um
1024²/25 steps fica na casa de HORAS. A estimativa varia com o numero de vCPU e
com a RAM do host — meca no servico real antes de prometer prazo. O tempo limite
real e JOB_TIMEOUT_SECONDS.
Edicao com reference_images NAO esta implementada: alem de exigir o workflow de edit
(subgraph), o encode de 2+ referencias esgota a RAM disponivel em qualquer tamanho
— por isso da erro explicito.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import httpx

from app import comfy_workflow
from app.config import Settings
from app.providers.base import (
    GeneratedImage,
    ProgressFn,
    ProviderError,
    image_meta,
    normalize_format,
)
from app.schemas import JobCreate

__all__ = ["LocalComfyProvider", "Provider", "is_available", "POLL_INTERVAL_SECONDS"]

#: Intervalo de poll do /history (contrato: 5 s).
POLL_INTERVAL_SECONDS = 5.0
#: Percentual reservado para o sampler (o resto e fila/carregamento/download).
_STEP_FLOOR = 5
_STEP_CEILING = 90


def is_available(settings: Settings) -> tuple[bool, str]:
    """Disponivel quando COMFY_URL esta preenchida (nao checa se o servidor responde)."""
    if not (getattr(settings, "comfy_url", "") or "").strip():
        return False, "COMFY_URL nao configurada (ex.: http://comfy:8188)"
    return True, "COMFY_URL configurada"


# ---------------------------------------------------------------------------
# Helpers de mensagem (pt-BR)
# ---------------------------------------------------------------------------


def format_eta(seconds: float) -> str:
    """ETA legivel: '~45 s restantes' | '~12 min restantes' | '~3 h 05 min restantes'."""
    if seconds <= 0:
        return "menos de 1 min restante"
    if seconds < 60:
        return f"~{int(seconds)} s restantes"
    if seconds < 3600:
        return f"~{int(seconds // 60)} min restantes"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"~{hours} h {minutes:02d} min restantes"


def _message_parts(message: Any) -> tuple[str, dict[str, Any]]:
    """Normaliza as entradas de status.messages do /history (dict ou lista)."""
    if isinstance(message, dict):
        kind = str(message.get("type") or message.get("kind") or "")
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        return kind, data
    if isinstance(message, (list, tuple)) and message:
        kind = str(message[0])
        data = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
        return kind, data
    return "", {}


class _RunState:
    """Estado para calcular percentual e ETA a partir do progresso reportado."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, total_steps)
        self.step_done: int | None = None
        self.step_max: int | None = None
        self._rate_anchor: tuple[float, int] | None = None
        self.executed_nodes: set[str] = set()
        self.ahead: int | None = None
        self.last_emitted: tuple[int, str] | None = None

    def observe_steps(self, done: int, maximum: int | None) -> None:
        self.step_done = max(0, int(done))
        if maximum and int(maximum) > 0:
            self.step_max = int(maximum)
        if self.step_done > 0 and self._rate_anchor is None:
            self._rate_anchor = (time.monotonic(), self.step_done)

    def seconds_per_step(self) -> float | None:
        if self._rate_anchor is None or self.step_done is None:
            return None
        started, anchor_step = self._rate_anchor
        progressed = self.step_done - anchor_step
        elapsed = time.monotonic() - started
        if progressed <= 0 or elapsed <= 0:
            return None
        return elapsed / progressed

    def snapshot(self) -> tuple[int, str]:
        """(percentual, mensagem) no estado atual.

        Os percentuais sao monotonos ao longo de um job (submit 2 -> fila 4 ->
        steps 5..90 -> download 92..99 -> 100), para a barra da UI nunca voltar.
        """
        if self.ahead is not None and self.step_done in (None, 0):
            if self.ahead <= 0:
                return 4, "ComfyUI iniciando a execucao (carregando modelos)"
            return 4, f"na fila do ComfyUI: {self.ahead} job(s) a frente"

        if self.step_done and self.step_max:
            total = self.step_max
            done = min(self.step_done, total)
            percent = _STEP_FLOOR + int(
                round((_STEP_CEILING - _STEP_FLOOR) * done / max(1, total))
            )
            message = f"gerando: step {done}/{total}"
            rate = self.seconds_per_step()
            if rate and done < total:
                message += f" ({format_eta(rate * (total - done))})"
            elif done >= total:
                message += " (ultimos steps, decodificando VAE)"
            return percent, message

        # Sem mensagens de progresso (ComfyUI antigo ou ainda carregando modelos):
        # percentual baixo, com a mensagem dizendo o que esta acontecendo.
        percent = min(_STEP_CEILING - 10, _STEP_FLOOR + 5 * len(self.executed_nodes))
        if self.executed_nodes:
            return percent, (
                "carregando/executando no ComfyUI "
                f"({len(self.executed_nodes)} node(s) concluido(s)) — no CPU ARM isso demora horas"
            )
        return max(4, percent), "aguardando o ComfyUI executar o workflow"


class LocalComfyProvider:
    """Cliente do ComfyUI local (t2i apenas)."""

    name = "local_comfy"

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        #: Injetavel para testes (httpx.MockTransport). None = rede real.
        self._transport = transport
        #: Modelo do DiT que este workflow usa (aparece em /health e /config).
        self.model_id = comfy_workflow.DIFFUSION_MODEL_GGUF
        self.steps: int = comfy_workflow.DEFAULT_STEPS
        self.cfg: float = comfy_workflow.DEFAULT_CFG

    # ------------------------------------------------------------------
    def _base_url(self) -> str:
        url = (getattr(self.settings, "comfy_url", "") or "").strip()
        if not url:
            raise ProviderError(
                "local_comfy indisponivel: COMFY_URL nao configurada (ex.: http://comfy:8188)",
                provider=self.name,
            )
        return url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        # A geracao em si fica no poll do /history: os timeouts HTTP aqui sao curtos.
        timeout = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=5.0)
        return httpx.AsyncClient(
            base_url=self._base_url(),
            timeout=timeout,
            follow_redirects=True,
            transport=self._transport,
        )

    def _offline_error(self, base: str, exc: Exception) -> ProviderError:
        return ProviderError(
            f"ComfyUI indisponivel em {base}: {exc}. "
            "Suba o servico com `docker compose --profile local up comfy` (o set GGUF/TE/VAE "
            f"tem ~{comfy_workflow.MINIMUM_MODEL_SET_GB:.1f} GB para baixar). "
            "ATENCAO: em um host ARM64 sem GPU o ComfyUI roda so em CPU — "
            "1024²/25 steps leva horas; prefira IMAGE_PROVIDER=fal.",
            provider=self.name,
        )

    # ------------------------------------------------------------------
    # API do protocolo ImageProvider
    # ------------------------------------------------------------------
    async def generate(self, req: JobCreate, progress: ProgressFn) -> list[GeneratedImage]:
        if req.reference_images:
            raise ProviderError(
                "local_comfy nao suporta reference_images: o workflow de edit do 2.1 usa um "
                "subgraph proprio e o encode de 2+ referencias esgota a RAM disponivel em "
                "qualquer tamanho. Use IMAGE_PROVIDER=fal com o FAL_EDIT_MODEL para edicao.",
                provider=self.name,
            )

        base = self._base_url()
        workflow = comfy_workflow.build_workflow(
            prompt=req.prompt,
            negative=req.negative_prompt or "",
            width=int(req.width),
            height=int(req.height),
            steps=self.steps,
            cfg=self.cfg,
            seed=req.seed if req.seed is not None else 0,
            filename_prefix=f"{comfy_workflow.DEFAULT_FILENAME_PREFIX}_{uuid.uuid4().hex[:8]}",
            batch_size=int(req.num_images),
        )
        total_steps = comfy_workflow.total_steps(workflow)
        client_id = str(uuid.uuid4())

        async with self._client() as client:
            await self._health_check(client, base)
            prompt_id = await self._submit(client, workflow, client_id, base, progress)
            try:
                images = await self._await_result(client, prompt_id, total_steps, req, progress)
            except asyncio.CancelledError:
                await self._interrupt(client, prompt_id)
                raise

        await progress(100, f"ComfyUI: {len(images)} imagem(ns) gerada(s)")
        return images

    # ------------------------------------------------------------------
    async def _health_check(self, client: httpx.AsyncClient, base: str) -> None:
        try:
            response = await client.get("/system_stats")
        except httpx.HTTPError as exc:
            raise self._offline_error(base, exc) from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"ComfyUI respondeu HTTP {response.status_code} em {base}/system_stats",
                provider=self.name,
                status_code=response.status_code,
                body=(response.text or "")[:500],
            )

    async def _submit(
        self,
        client: httpx.AsyncClient,
        workflow: dict[str, Any],
        client_id: str,
        base: str,
        progress: ProgressFn,
    ) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        try:
            response = await client.post("/prompt", json=payload)
        except httpx.HTTPError as exc:
            raise self._offline_error(base, exc) from exc

        if response.status_code >= 400:
            body = (response.text or "").strip()[:1500]
            raise ProviderError(
                f"ComfyUI recusou o workflow (HTTP {response.status_code}): {body} "
                "(confira GET /object_info: classes de node e nomes de arquivo em "
                "ComfyUI/models/diffusion_models|text_encoders|vae)",
                provider=self.name,
                status_code=response.status_code,
                body=body,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("ComfyUI devolveu resposta nao-JSON no POST /prompt", provider=self.name) from exc
        if data.get("node_errors"):
            raise ProviderError(
                f"ComfyUI reportou node_errors: {str(data['node_errors'])[:1200]}",
                provider=self.name,
            )
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ProviderError(f"ComfyUI nao devolveu prompt_id: {str(data)[:500]}", provider=self.name)

        position = data.get("number")
        suffix = f" (posicao {position} na fila)" if isinstance(position, int) else ""
        await progress(3, f"workflow enviado ao ComfyUI{suffix}; aguardando execucao")
        return str(prompt_id)

    async def _await_result(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
        total_steps: int,
        req: JobCreate,
        progress: ProgressFn,
    ) -> list[GeneratedImage]:
        state = _RunState(total_steps)
        output_nodes = comfy_workflow.DEFAULT_OUTPUT_NODE_IDS

        while True:
            entry = await self._history_entry(client, prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                status_str = status.get("status_str")
                if status_str == "error":
                    raise ProviderError(
                        f"ComfyUI falhou ao executar o workflow: {self._error_detail(status)}",
                        provider=self.name,
                    )
                images = self._entry_images(entry, output_nodes)
                if images:
                    return await self._download_all(client, images, req, progress)
                if status.get("completed") is True or status_str == "success":
                    raise ProviderError(
                        "ComfyUI terminou o workflow sem imagens (o node "
                        f"{comfy_workflow.NODE_SAVE} nao produziu saida)",
                        provider=self.name,
                    )
                self._observe(status, state)

            if entry is None:
                ahead = await self._queue_position(client, prompt_id)
                if ahead is not None:
                    state.ahead = ahead

            await self._emit(progress, state)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _history_entry(self, client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any] | None:
        try:
            response = await client.get(f"/history/{prompt_id}")
        except httpx.HTTPError as exc:
            raise self._offline_error(self._base_url(), exc) from exc
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        entry = (data or {}).get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def _queue_position(self, client: httpx.AsyncClient, prompt_id: str) -> int | None:
        """Quantos jobs estao a frente do nosso (0 = executando agora)."""
        try:
            response = await client.get("/queue")
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        running = data.get("queue_running") or []
        pending = data.get("queue_pending") or []
        for item in running:
            if self._item_prompt_id(item) == prompt_id:
                return 0
        for index, item in enumerate(pending):
            if self._item_prompt_id(item) == prompt_id:
                return index + 1
        return len(running) + len(pending)

    @staticmethod
    def _item_prompt_id(item: Any) -> str | None:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            return str(item[1])
        if isinstance(item, dict):
            value = item.get("prompt_id")
            return str(value) if value else None
        return None

    @staticmethod
    def _observe(status: dict[str, Any], state: _RunState) -> None:
        for raw in status.get("messages") or []:
            kind, data = _message_parts(raw)
            if kind == "progress":
                value = data.get("value")
                if isinstance(value, (int, float)):
                    maximum = data.get("max")
                    state.observe_steps(
                        int(value), int(maximum) if isinstance(maximum, (int, float)) else None
                    )
            elif kind == "executing":
                node = data.get("node")
                if node is not None:
                    state.executed_nodes.add(str(node))
            elif kind == "execution_cached":
                for node in data.get("nodes") or []:
                    state.executed_nodes.add(str(node))

    @staticmethod
    async def _emit(progress: ProgressFn, state: _RunState) -> None:
        percent, message = state.snapshot()
        if state.last_emitted == (percent, message):
            return None
        state.last_emitted = (percent, message)
        await progress(percent, message)
        return None

    @staticmethod
    def _error_detail(status: dict[str, Any]) -> str:
        details: list[str] = []
        for raw in status.get("messages") or []:
            kind, data = _message_parts(raw)
            if kind in ("execution_error", "execution_interrupted"):
                details.append(str(data.get("exception_message") or data.get("exception_type") or data))
        return " | ".join(details)[:1200] if details else "sem detalhes no /history"

    @staticmethod
    def _entry_images(entry: dict[str, Any], output_nodes: tuple[str, ...]) -> list[dict[str, Any]]:
        outputs = entry.get("outputs") or {}
        found: list[dict[str, Any]] = []
        for node_id in output_nodes:
            node_output = outputs.get(node_id) or {}
            for image in node_output.get("images") or []:
                if isinstance(image, dict) and image.get("filename"):
                    found.append(image)
        if found:
            return found
        # fallback: qualquer node que tenha produzido uma imagem
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for image in node_output.get("images") or []:
                if isinstance(image, dict) and image.get("filename"):
                    found.append(image)
        return found

    async def _download_all(
        self,
        client: httpx.AsyncClient,
        images: list[dict[str, Any]],
        req: JobCreate,
        progress: ProgressFn,
    ) -> list[GeneratedImage]:
        total = len(images)
        result: list[GeneratedImage] = []
        for index, image in enumerate(images):
            params = {
                "filename": image.get("filename"),
                "subfolder": image.get("subfolder") or "",
                "type": image.get("type") or "output",
            }
            try:
                response = await client.get("/view", params=params)
            except httpx.HTTPError as exc:
                raise self._offline_error(self._base_url(), exc) from exc
            if response.status_code >= 400:
                raise ProviderError(
                    f"ComfyUI: HTTP {response.status_code} ao baixar /view "
                    f"({params['filename']}): {(response.text or '')[:300]}",
                    provider=self.name,
                    status_code=response.status_code,
                )
            raw = response.content
            width, height, fmt = image_meta(raw)
            result.append(
                GeneratedImage(
                    data=raw,
                    width=width,
                    height=height,
                    seed=req.seed,
                    format=normalize_format(fmt),
                )
            )
            await progress(
                92 + int(round(7 * (index + 1) / max(1, total))),
                f"baixando imagem {index + 1}/{total} do ComfyUI",
            )
        return result

    async def _interrupt(self, client: httpx.AsyncClient, prompt_id: str) -> None:
        """Best-effort: avisa o ComfyUI para parar o job cancelado."""
        try:
            await asyncio.shield(client.post("/interrupt", json={"prompt_id": prompt_id}))
        except Exception:  # noqa: BLE001 - cancelamento nao pode falhar
            return None
        return None


#: Alias de compatibilidade (nome curto).
Provider = LocalComfyProvider
