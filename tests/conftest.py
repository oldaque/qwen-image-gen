"""Fixtures compartilhadas dos testes.

Principios:
  * ZERO rede, ZERO Docker, ZERO GPU: o provider dos testes de API e sempre o
    ``fake`` (Pillow puro). Os testes de ``fal``/``remote_gpu`` injetam
    ``httpx.MockTransport`` — nunca abrem socket.
  * Isolamento total de disco: ``DATA_DIR`` aponta para ``tmp_path`` (SQLite e
    imagens vivem la e desaparecem no fim do teste).
  * Hermetismo de ambiente: as variaveis do contrato sao removidas do ambiente
    do processo e os valores entram como kwargs explicito de ``Settings`` (a
    precedencia do pydantic-settings e init kwargs > env > .env), de modo que
    um ``.env`` ou um ``IMAGE_PROVIDER=fal`` exportado no shell do dev nao
    contamine a suite.
  * O lifespan do FastAPI e executado de verdade (``app.router.lifespan_context``):
    sem isso o banco nao seria criado e a fila nao subiria. O ``ASGITransport``
    do httpx, sozinho, NAO dispara eventos de lifespan.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Awaitable

import httpx
import pytest
import pytest_asyncio

# O repo tem tests/ sem __init__.py: sem isto o `import app` falha quando o
# pytest e chamado de fora da raiz (ex.: `pytest -q tests/`) em import-mode prepend.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402

#: Assinatura binaria de um PNG (usada para provar que o download e uma imagem real).
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
WEBP_MAGIC = b"RIFF"

#: Status em que o job nao evolui mais (contrato).
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

#: Base URL ficticia: o transporte e ASGI, nada sai para a rede.
BASE_URL = "http://testserver"

#: Valor do SIMULATED_SECONDS do provider fake durante a suite. O default real e
#: 3.0 s (contrato); acelerar aqui mantem a suite em segundos, nao em minutos.
FAST_FAKE_SECONDS = 0.02

#: Variaveis do contrato que NUNCA devem vazar do shell para os testes.
_ENV_VARS = (
    "IMAGE_PROVIDER",
    "DATA_DIR",
    "DB_PATH",
    "IMAGES_DIR",
    "FAL_KEY",
    "FAL_MODEL",
    "FAL_EDIT_MODEL",
    "COMFY_URL",
    "REMOTE_GPU_URL",
    "REMOTE_GPU_KEY",
    "REMOTE_GPU_MODEL",
    "JOB_TIMEOUT_SECONDS",
    "MAX_PARALLEL_JOBS",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "MIN_SIZE",
    "MAX_SIZE",
    "APP_VERSION",
)


# ---------------------------------------------------------------------------
# ambiente
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove do processo qualquer env var do contrato (hermetismo)."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def fake_rapido(monkeypatch: pytest.MonkeyPatch) -> float:
    """Acelera o provider fake e devolve o default real (para assertar o contrato)."""
    fake_module = importlib.import_module("app.providers.fake")
    original = float(fake_module.SIMULATED_SECONDS)
    monkeypatch.setattr(fake_module, "SIMULATED_SECONDS", FAST_FAKE_SECONDS, raising=False)
    return original


@pytest.fixture
def fake_simulated_seconds_default(fake_rapido: float) -> float:
    """Default de fábrica do SIMULATED_SECONDS (o patch do ``fake_rapido`` nao o altera)."""
    return fake_rapido


# ---------------------------------------------------------------------------
# settings / app / client
# ---------------------------------------------------------------------------


def build_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings hermeticas apontando para um DATA_DIR temporario."""
    data_dir = tmp_path / "data"
    values: dict[str, Any] = {
        "image_provider": "fake",
        "data_dir": str(data_dir),
        "db_path": str(data_dir / "jobs.db"),
        "images_dir": str(data_dir / "images"),
        "fal_key": "",
        "remote_gpu_url": "",
        "remote_gpu_key": "",
        "comfy_url": "http://comfy:8188",
        "job_timeout_seconds": 120,
        "max_parallel_jobs": 1,
        "default_width": 512,
        "default_height": 512,
        "min_size": 512,
        "max_size": 2048,
        "app_version": "0.1.0",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings do app de teste: provider fake, disco em tmp_path."""
    return build_settings(tmp_path)


@pytest.fixture
def app_instance(settings: Settings):
    """Aplicacao FastAPI construida com as settings de teste."""
    main_module = importlib.import_module("app.main")
    return main_module.create_app(settings)


@pytest_asyncio.fixture
async def client(app_instance) -> AsyncIterator[httpx.AsyncClient]:
    """Cliente httpx sobre ASGITransport COM lifespan em execucao (fila + SQLite)."""
    async with app_instance.router.lifespan_context(app_instance):
        transport = httpx.ASGITransport(app=app_instance)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as http:
            yield http


@pytest.fixture
def settings_provider_indisponivel(tmp_path: Path) -> Settings:
    """Settings de um provider real sem credencial (fal sem FAL_KEY)."""
    return build_settings(tmp_path, image_provider="fal", fal_key="")


@pytest.fixture
def app_provider_indisponivel(settings_provider_indisponivel: Settings):
    main_module = importlib.import_module("app.main")
    return main_module.create_app(settings_provider_indisponivel)


@pytest_asyncio.fixture
async def client_provider_indisponivel(app_provider_indisponivel) -> AsyncIterator[httpx.AsyncClient]:
    """Cliente de um app com IMAGE_PROVIDER=fal e sem chave (fail fast)."""
    async with app_provider_indisponivel.router.lifespan_context(app_provider_indisponivel):
        transport = httpx.ASGITransport(app=app_provider_indisponivel)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as http:
            yield http


# ---------------------------------------------------------------------------
# helpers de fluxo (fixtures que devolvem funcoes async)
# ---------------------------------------------------------------------------

#: Payload minimo valido (dentro de MIN_SIZE..MAX_SIZE e de num_images 1..4).
PAYLOAD_MINIMO: dict[str, Any] = {
    "prompt": "gato astronauta em aquarela",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "output_format": "png",
}


@pytest.fixture
def create_job(client: httpx.AsyncClient) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Cria um job via POST /api/v1/jobs e devolve o JobView (dict)."""

    async def _create(*, expect: int = 202, **overrides: Any) -> dict[str, Any]:
        payload = {**PAYLOAD_MINIMO, **overrides}
        response = await client.post("/api/v1/jobs", json=payload)
        assert response.status_code == expect, f"POST /api/v1/jobs -> {response.status_code}: {response.text}"
        return response.json()

    return _create


@pytest.fixture
def wait_for_terminal(
    client: httpx.AsyncClient,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Faz poll de GET /api/v1/jobs/{id} ate o status ser terminal."""

    async def _wait(job_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = await client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200, response.text
            last = response.json()
            if last["status"] in TERMINAL_STATUSES:
                return last
            await asyncio.sleep(0.05)
        raise AssertionError(
            f"job {job_id} nao chegou a um status terminal em {timeout}s (ultimo: {last!r})"
        )

    return _wait
