"""Testes da API HTTP (app/main.py) — fluxo completo, validacao, 404, health/config.

Tudo roda sobre ASGITransport (sem socket, sem Docker) com o provider ``fake``
(Pillow, offline). A fila e o SQLite reais participam do teste: o lifespan do app
sobe no fixture ``client``.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

#: Assinatura binaria de um PNG.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: Status em que o job nao evolui mais.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# fluxo completo
# ---------------------------------------------------------------------------


async def test_fluxo_completo_criar_gerar_baixar_historico_apagar(
    client: httpx.AsyncClient, create_job, wait_for_terminal
) -> None:
    """create (202) -> poll ate succeeded -> bytes PNG -> historico -> delete (204)."""
    job = await create_job(prompt="gato astronauta em aquarela", seed=1234)

    # --- JobView no 202 -------------------------------------------------
    assert set(job) >= {
        "job_id",
        "status",
        "progress",
        "progress_message",
        "error",
        "provider",
        "model_id",
        "request",
        "images",
        "created_at",
        "started_at",
        "finished_at",
        "elapsed_seconds",
    }
    assert job["job_id"]
    assert job["status"] in {"queued", "running"}
    assert job["provider"] == "fake"
    assert job["model_id"]
    assert job["request"]["prompt"] == "gato astronauta em aquarela"
    assert job["request"]["width"] == 512
    assert job["images"] == []
    assert job["error"] is None
    assert job["created_at"]

    # --- execucao ------------------------------------------------------
    done = await wait_for_terminal(job["job_id"])
    assert done["status"] == "succeeded", done.get("error")
    assert done["progress"] == 100
    assert done["progress_message"]
    assert done["error"] is None
    assert done["started_at"] and done["finished_at"]
    assert isinstance(done["elapsed_seconds"], (int, float))
    assert done["elapsed_seconds"] >= 0

    assert len(done["images"]) == 1
    image = done["images"][0]
    assert set(image) >= {"image_id", "url", "width", "height", "bytes", "seed", "format"}
    assert image["url"] == f"/api/v1/images/{image['image_id']}"
    assert (image["width"], image["height"]) == (512, 512)
    assert image["format"] == "png"
    assert image["bytes"] > 0
    assert image["seed"] == 1234

    # --- download do binario -------------------------------------------
    download = await client.get(image["url"])
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("image/png")
    disposition = download.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert image["image_id"] in disposition
    body = download.content
    assert body.startswith(PNG_MAGIC), "o corpo nao comeca com a assinatura PNG"
    assert len(body) == image["bytes"], "bytes declarados != bytes devolvidos"
    with Image.open(io.BytesIO(body)) as img:
        img.load()
        assert img.size == (512, 512)
        assert img.format == "PNG"

    # --- historico ------------------------------------------------------
    listing = await client.get("/api/v1/jobs", params={"limit": 20, "offset": 0})
    assert listing.status_code == 200
    payload = listing.json()
    assert set(payload) == {"items", "total"}
    assert payload["total"] >= 1
    assert payload["items"][0]["job_id"] == job["job_id"]  # mais recente primeiro
    assert any(item["job_id"] == job["job_id"] for item in payload["items"])

    # --- delete: apaga as imagens e o job segue no historico ------------
    deleted = await client.delete(f"/api/v1/jobs/{job['job_id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""

    after = await client.get(f"/api/v1/jobs/{job['job_id']}")
    assert after.status_code == 200
    assert after.json()["images"] == []
    gone = await client.get(image["url"])
    assert gone.status_code == 404

    still_listed = await client.get("/api/v1/jobs")
    assert still_listed.json()["total"] >= 1


async def test_output_format_jpeg_muda_content_type(
    client: httpx.AsyncClient, create_job, wait_for_terminal
) -> None:
    """output_format e honrado de ponta a ponta (nome do arquivo + Content-Type)."""
    job = await create_job(prompt="retrato", output_format="jpeg")
    done = await wait_for_terminal(job["job_id"])
    assert done["status"] == "succeeded", done.get("error")

    image = done["images"][0]
    assert image["format"] == "jpeg"
    download = await client.get(image["url"])
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("image/jpeg")
    assert download.content.startswith(b"\xff\xd8\xff")


async def test_multiplas_imagens(client: httpx.AsyncClient, create_job, wait_for_terminal) -> None:
    """num_images=3 gera 3 imagens distintas, cada uma baixavel."""
    job = await create_job(prompt="tres variacoes", num_images=3)
    done = await wait_for_terminal(job["job_id"])
    assert done["status"] == "succeeded", done.get("error")
    assert len(done["images"]) == 3
    assert len({image["image_id"] for image in done["images"]}) == 3
    for image in done["images"]:
        assert (await client.get(image["url"])).status_code == 200


# ---------------------------------------------------------------------------
# validacao (422)
# ---------------------------------------------------------------------------


async def test_prompt_vazio_e_recusado(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/jobs", json={"prompt": ""})
    assert response.status_code == 422
    assert "prompt" in response.json()["detail"]


async def test_prompt_ausente_e_recusado(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/jobs", json={"width": 512, "height": 512})
    assert response.status_code == 422
    assert "prompt" in response.json()["detail"]


@pytest.mark.parametrize("value", [511, 100, 2049, 4096])
async def test_width_fora_da_faixa_e_recusado(client: httpx.AsyncClient, value: int) -> None:
    response = await client.post(
        "/api/v1/jobs", json={"prompt": "teste de limite", "width": value, "height": 512}
    )
    assert response.status_code == 422, response.text
    assert "width" in response.json()["detail"]


async def test_height_fora_da_faixa_e_recusado(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/jobs", json={"prompt": "teste de limite", "width": 512, "height": 5000}
    )
    assert response.status_code == 422
    assert "height" in response.json()["detail"]


@pytest.mark.parametrize("value", [0, 5, -1])
async def test_num_images_fora_da_faixa_e_recusado(client: httpx.AsyncClient, value: int) -> None:
    response = await client.post(
        "/api/v1/jobs", json={"prompt": "teste de limite", "num_images": value}
    )
    assert response.status_code == 422, response.text


async def test_formato_invalido_e_recusado(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/jobs", json={"prompt": "teste", "output_format": "tiff"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 404
# ---------------------------------------------------------------------------


async def test_job_inexistente_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/jobs/nao-existe")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_imagem_inexistente_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/images/nao-existe")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_delete_de_job_inexistente_404(client: httpx.AsyncClient) -> None:
    response = await client.delete("/api/v1/jobs/nao-existe")
    assert response.status_code == 404


async def test_sse_de_job_inexistente_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/jobs/nao-existe/events")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# cancelamento
# ---------------------------------------------------------------------------


async def test_cancelamento_de_job_na_fila(
    client: httpx.AsyncClient, create_job, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Com 1 worker, o 2o job fica em 'queued'; DELETE nele tem de cancelar."""
    fake_module = importlib.import_module("app.providers.fake")
    # Override do patch do conftest: job longo o bastante para o 2o ficar na fila.
    monkeypatch.setattr(fake_module, "SIMULATED_SECONDS", 30.0, raising=False)

    primeiro = await create_job(prompt="job longo que segura o worker")
    await asyncio.sleep(0.2)  # deixa o worker pegar o primeiro

    segundo = await create_job(prompt="job que vai ser cancelado na fila")
    assert segundo["status"] == "queued"

    deleted = await client.delete(f"/api/v1/jobs/{segundo['job_id']}")
    assert deleted.status_code == 204

    estado = (await client.get(f"/api/v1/jobs/{segundo['job_id']}")).json()
    assert estado["status"] == "cancelled"
    assert estado["finished_at"] is not None
    assert estado["images"] == []

    # cleanup: cancela o job longo para o lifespan nao esperar por ele
    assert (await client.delete(f"/api/v1/jobs/{primeiro['job_id']}")).status_code == 204


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


async def test_sse_envia_evento_status_ate_o_fim(
    client: httpx.AsyncClient, create_job
) -> None:
    """GET /events: text/event-stream, evento 'status' com JobView JSON, fecha no terminal."""
    job = await create_job(prompt="sse")

    eventos: list[dict[str, Any]] = []
    async with asyncio.timeout(30):
        async with client.stream("GET", f"/api/v1/jobs/{job['job_id']}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            nome_evento: str | None = None
            async for linha in response.aiter_lines():
                if linha.startswith("event:"):
                    nome_evento = linha.split(":", 1)[1].strip()
                elif linha.startswith("data:"):
                    assert nome_evento == "status", f"evento inesperado: {nome_evento}"
                    eventos.append(json.loads(linha.split(":", 1)[1].strip()))
                    if eventos[-1]["status"] in TERMINAL_STATUSES:
                        break

    assert eventos, "nenhum evento 'status' recebido"
    for evento in eventos:
        assert evento["job_id"] == job["job_id"]
        assert set(evento) >= {"status", "progress", "provider", "model_id", "request", "images"}
    assert eventos[-1]["status"] == "succeeded", eventos[-1].get("error")


# ---------------------------------------------------------------------------
# health / config
# ---------------------------------------------------------------------------


async def test_health_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"ok", "provider", "model_id", "queue_depth", "version"}
    assert body["ok"] is True
    assert body["provider"] == "fake"
    assert isinstance(body["model_id"], str) and body["model_id"]
    assert isinstance(body["queue_depth"], int) and body["queue_depth"] >= 0
    assert body["version"] == "0.1.0"


async def test_config_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "provider",
        "model_id",
        "providers_available",
        "min_size",
        "max_size",
        "job_timeout_seconds",
    }
    assert body["provider"] == "fake"
    assert isinstance(body["model_id"], str) and body["model_id"]
    assert isinstance(body["providers_available"], list)
    assert "fake" in body["providers_available"]
    assert body["min_size"] == 512
    assert body["max_size"] == 2048
    assert body["job_timeout_seconds"] == 120


async def test_health_conta_fila(client: httpx.AsyncClient, create_job) -> None:
    """queue_depth espelha jobs ativos (queued + running) no SQLite."""
    await create_job(prompt="conta fila")
    response = await client.get("/api/v1/health")
    assert response.json()["queue_depth"] >= 0


# ---------------------------------------------------------------------------
# frontend estatico e fail-fast do provider
# ---------------------------------------------------------------------------


async def test_index_servido_na_raiz(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"<html" in response.content.lower()


async def test_provider_indisponivel_fail_fast(
    client_provider_indisponivel: httpx.AsyncClient,
) -> None:
    """IMAGE_PROVIDER=fal sem FAL_KEY: POST /jobs -> 400, mas /health segue 200."""
    response = await client_provider_indisponivel.post(
        "/api/v1/jobs", json={"prompt": "sem chave", "width": 512, "height": 512}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "FAL_KEY" in detail or "fal" in detail.lower()

    health = await client_provider_indisponivel.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["provider"] == "fal"
    assert body["model_id"] == "alibaba/qwen-image-2.1/text-to-image"

    config = await client_provider_indisponivel.get("/api/v1/config")
    assert config.status_code == 200
    assert "fal" not in config.json()["providers_available"]


async def test_base_url_do_cliente_e_apenas_o_asgi(client: httpx.AsyncClient) -> None:
    """Guarda-corpo: a suite nao depende de rede (o transporte e ASGI, nao socket)."""
    assert isinstance(client._transport, httpx.ASGITransport)
    assert client.base_url.host == "testserver"
