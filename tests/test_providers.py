"""Testes dos providers de geracao (app/providers/*).

Sem rede e sem Docker em nenhum caso:
  * ``fake`` roda Pillow de verdade e checa a assinatura/dimensoes do PNG;
  * ``fal`` e ``remote_gpu`` recebem um ``httpx.MockTransport`` injetado (o
    provider aceita ``transport=`` no construtor) e o payload/resposta sao
    inspecionados como objetos httpx reais;
  * ``get_provider`` e testado por env (nome do provider -> classe certa) e por
    fail-fast (falta de chave/URL -> ProviderError).
"""

from __future__ import annotations

import base64
import importlib
import io
from typing import Any

import httpx
import pytest
from PIL import Image, ImageFont

from app.config import Settings
from app.providers import get_provider, is_provider_available
from app.providers.base import (
    ImageProvider,
    ProviderError,
    decode_base64_image,
    image_meta,
    noop_progress,
)
from app.providers.fake import FakeProvider
from app.providers.fal import FalProvider
from app.providers.remote_gpu import RemoteGPUProvider
from app.schemas import JobCreate

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

#: Endpoints conforme a documentacao oficial do provider fal (base e modelos).
FAL_T2I_MODEL = "alibaba/qwen-image-2.1/text-to-image"
FAL_EDIT_MODEL = "alibaba/qwen-image-2.1/edit"
FAL_BASE = "https://fal.run"
CDN_URL = "https://cdn.fal.exemplo/out.png"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _png(width: int = 64, height: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 148, 136)).save(buffer, format="PNG")
    return buffer.getvalue()


def _settings(**overrides: Any) -> Settings:
    """Settings de teste (sem disco, sem .env: valores por kwargs explícito)."""
    values: dict[str, Any] = {
        "image_provider": "fake",
        "fal_key": "",
        "fal_model": FAL_T2I_MODEL,
        "fal_edit_model": FAL_EDIT_MODEL,
        "remote_gpu_url": "",
        "remote_gpu_key": "",
        "remote_gpu_model": "Qwen/Qwen-Image-2.1",
        "comfy_url": "http://comfy:8188",
        "job_timeout_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


def _req(**overrides: Any) -> JobCreate:
    values: dict[str, Any] = {
        "prompt": "montanha ao amanhecer",
        "width": 512,
        "height": 512,
        "num_images": 1,
        "output_format": "png",
    }
    values.update(overrides)
    return JobCreate(**values)


class _Progresso:
    """ProgressFn coletora: guarda (percentual, mensagem) na ordem das chamadas."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, percent: int, message: str) -> None:
        self.calls.append((int(percent), str(message)))

    @property
    def valores(self) -> list[int]:
        return [value for value, _ in self.calls]


# ---------------------------------------------------------------------------
# fake (Pillow, offline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_gera_png_valido_com_pillow() -> None:
    provider = get_provider(_settings(image_provider="fake"))
    assert isinstance(provider, FakeProvider)
    assert isinstance(provider, ImageProvider)
    assert provider.name == "fake"

    progresso = _Progresso()
    imagens = await provider.generate(
        _req(prompt="praia ao por do sol", num_images=2, seed=42), progresso
    )

    assert len(imagens) == 2
    for index, imagem in enumerate(imagens):
        assert imagem.data.startswith(PNG_MAGIC), "nao e um PNG"
        with Image.open(io.BytesIO(imagem.data)) as img:
            img.load()
            assert img.size == (512, 512)
            assert img.format == "PNG"
        assert (imagem.width, imagem.height) == (512, 512)
        assert imagem.format == "png"
        assert imagem.seed == 42 + index

    # progresso: monotono, comeca em >0 e termina em 100, mensagens pt-BR nao vazias
    valores = progresso.valores
    assert valores == sorted(valores), "progresso voltou atras"
    assert valores[-1] == 100
    assert 0 < valores[0] <= 100
    assert all(mensagem.strip() for _, mensagem in progresso.calls)


@pytest.mark.parametrize(
    ("output_format", "magic"),
    [("jpeg", JPEG_MAGIC), ("webp", b"RIFF")],
)
@pytest.mark.asyncio
async def test_fake_respeita_output_format(output_format: str, magic: bytes) -> None:
    provider = FakeProvider(_settings())
    imagens = await provider.generate(_req(output_format=output_format), noop_progress)
    imagem = imagens[0]
    assert imagem.format == output_format
    assert imagem.data.startswith(magic), f"magic bytes errados para {output_format}"
    with Image.open(io.BytesIO(imagem.data)) as img:
        assert img.format == ("JPEG" if output_format == "jpeg" else "WEBP")


@pytest.mark.asyncio
async def test_fake_nao_toca_a_rede(monkeypatch: pytest.MonkeyPatch) -> None:
    """Se o fake tentasse HTTP, este teste explodiria."""

    def _proibido(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - so falha
        raise AssertionError("o provider fake nao pode abrir conexao HTTP")

    monkeypatch.setattr(httpx, "AsyncClient", _proibido)
    imagens = await FakeProvider(_settings()).generate(
        _req(prompt="offline, por favor"), noop_progress
    )
    assert imagens[0].data.startswith(PNG_MAGIC)


def test_fake_simula_os_3_segundos_do_contrato(fake_simulated_seconds_default: float) -> None:
    """O default de fabrica do SIMULATED_SECONDS e ~3 s (o conftest so acelera nos testes)."""
    assert fake_simulated_seconds_default == pytest.approx(3.0, abs=0.5)


def test_fake_carrega_fonte_portatil_sem_fontes_do_sistema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem fonte do sistema E sem `load_default(size=...)`, a fonte ainda resolve.

    Regressao: a lista antiga so tinha caminhos do macOS e o fallback so tratava
    TypeError — qualquer outra falha do Pillow (OSError/ImportError quando o
    FreeType nao esta disponivel) abortava a geracao no primeiro deploy.
    """
    fake_module = importlib.import_module("app.providers.fake")
    monkeypatch.setattr(
        fake_module, "_FONT_CANDIDATES", ("/nao/existe/DejaVuSans.ttf",), raising=False
    )
    original = ImageFont.load_default

    def sem_freetype(*args: Any, **kwargs: Any) -> Any:
        if args or kwargs:  # load_default(size=...) precisa do FreeType
            raise OSError("FreeType indisponivel")
        return original()

    monkeypatch.setattr(ImageFont, "load_default", sem_freetype)
    fonte = fake_module._load_font(18)
    assert fonte.getbbox("texto") is not None, "a fonte de fallback nao desenha nada"


@pytest.mark.asyncio
async def test_fake_gera_imagem_sem_fontes_do_sistema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nenhum caminho de fonte pode quebrar a geracao (bug do deploy Linux)."""
    fake_module = importlib.import_module("app.providers.fake")
    monkeypatch.setattr(
        fake_module, "_FONT_CANDIDATES", ("/nao/existe/Arial.ttf",), raising=False
    )
    imagens = await FakeProvider(_settings()).generate(_req(prompt="sem fontes"), noop_progress)
    assert imagens[0].data.startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# fal (httpx.MockTransport)
# ---------------------------------------------------------------------------


def _fal_transport(
    *,
    response: httpx.Response,
    posts: list[httpx.Request] | None = None,
    gets: list[httpx.Request] | None = None,
    download: bytes | None = None,
) -> httpx.MockTransport:
    """MockTransport: POST registra e devolve `response`; GET devolve `download`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if posts is not None:
                posts.append(request)
            return response
        if gets is not None:
            gets.append(request)
        body = download if download is not None else _png()
        return httpx.Response(200, content=body, headers={"Content-Type": "image/png"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fal_monta_payload_t2i_e_baixa_a_imagem() -> None:
    png = _png(512, 512)
    posts: list[httpx.Request] = []
    gets: list[httpx.Request] = []
    transport = _fal_transport(
        response=httpx.Response(
            200,
            json={
                "images": [
                    {"url": CDN_URL, "width": 512, "height": 512, "content_type": "image/png"}
                ]
            },
        ),
        posts=posts,
        gets=gets,
        download=png,
    )
    provider = FalProvider(_settings(image_provider="fal", fal_key="test-fal-key"), transport=transport)
    assert provider.name == "fal"

    progresso = _Progresso()
    imagens = await provider.generate(
        _req(
            prompt="retrato de um gato",
            negative_prompt="borrado",
            num_images=2,
            seed=7,
            output_format="png",
            prompt_expander="none",
        ),
        progresso,
    )

    # --- request ---------------------------------------------------------
    assert len(posts) == 1
    request = posts[0]
    assert str(request.url) == f"{FAL_BASE}/{FAL_T2I_MODEL}"
    assert request.headers["Authorization"] == "Key test-fal-key"
    assert request.headers["Content-Type"].startswith("application/json")

    payload = json_payload(request)
    assert payload["prompt"] == "retrato de um gato"
    assert payload["negative_prompt"] == "borrado"
    assert payload["image_size"] == {"width": 512, "height": 512}
    assert payload["num_images"] == 2
    assert payload["seed"] == 7
    assert payload["output_format"] == "png"
    assert payload["prompt_expander"] == "none"
    assert payload["enable_safety_checker"] is False

    # --- download --------------------------------------------------------
    assert [str(r.url) for r in gets] == [CDN_URL]

    # --- resposta --------------------------------------------------------
    assert len(imagens) == 1
    imagem = imagens[0]
    assert imagem.data == png
    assert (imagem.width, imagem.height) == (512, 512)
    assert imagem.format == "png"
    assert imagem.seed == 7
    assert progresso.valores[-1] == 100


@pytest.mark.asyncio
async def test_fal_usa_o_modelo_de_edit_com_reference_images() -> None:
    png = _png(64, 64)
    inline = base64.b64encode(png).decode()
    posts: list[httpx.Request] = []
    transport = _fal_transport(
        response=httpx.Response(200, json={"images": [{"b64_json": inline, "width": 64, "height": 64}]}),
        posts=posts,
    )
    provider = FalProvider(_settings(image_provider="fal", fal_key="k"), transport=transport)

    referencias = [
        "https://exemplo.com/base.png",
        f"data:image/png;base64,{inline}",
    ]
    imagens = await provider.generate(_req(reference_images=referencias), noop_progress)

    assert len(posts) == 1
    assert str(posts[0].url) == f"{FAL_BASE}/{FAL_EDIT_MODEL}"
    payload = json_payload(posts[0])
    assert payload["image_urls"] == referencias  # ordem preservada
    assert payload.get("image_size") is None  # o schema de edit nao tem image_size
    assert imagens[0].data == png


@pytest.mark.asyncio
async def test_fal_sem_chave_levanta_provider_error() -> None:
    provider = FalProvider(_settings(image_provider="fal", fal_key="   "))
    with pytest.raises(ProviderError) as erro:
        await provider.generate(_req(), noop_progress)
    assert "FAL_KEY" in str(erro.value)


@pytest.mark.asyncio
async def test_fal_http_401_vira_provider_error() -> None:
    transport = _fal_transport(response=httpx.Response(401, json={"detail": "invalid key"}))
    provider = FalProvider(_settings(image_provider="fal", fal_key="errada"), transport=transport)
    with pytest.raises(ProviderError) as erro:
        await provider.generate(_req(), noop_progress)
    assert "401" in str(erro.value)
    assert erro.value.status_code == 401


@pytest.mark.asyncio
async def test_fal_resposta_sem_imagens_vira_provider_error() -> None:
    transport = _fal_transport(response=httpx.Response(200, json={"detail": "nada aqui"}))
    provider = FalProvider(_settings(image_provider="fal", fal_key="k"), transport=transport)
    with pytest.raises(ProviderError):
        await provider.generate(_req(), noop_progress)


# ---------------------------------------------------------------------------
# remote_gpu (httpx.MockTransport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_gpu_envia_payload_openai_e_decodifica_b64() -> None:
    png = _png(512, 512)
    posts: list[httpx.Request] = []
    transport = _fal_transport(
        response=httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]}),
        posts=posts,
    )
    provider = RemoteGPUProvider(
        _settings(
            image_provider="remote_gpu",
            remote_gpu_url="http://gpu.local:8091",
            remote_gpu_key="test-gpu-key",
            remote_gpu_model="Qwen/Qwen-Image-2.1",
        ),
        transport=transport,
    )
    assert provider.name == "remote_gpu"
    assert provider.model_id == "Qwen/Qwen-Image-2.1"

    progresso = _Progresso()
    imagens = await provider.generate(
        _req(prompt="deserto", num_images=2, seed=99), progresso
    )

    assert len(posts) == 1
    request = posts[0]
    assert str(request.url) == "http://gpu.local:8091/v1/images/generations"
    assert request.headers["Authorization"] == "Bearer test-gpu-key"

    payload = json_payload(request)
    assert payload["model"] == "Qwen/Qwen-Image-2.1"
    assert payload["prompt"] == "deserto"
    assert payload["size"] == "512x512"
    assert payload["n"] == 2
    assert payload["response_format"] == "b64_json"

    imagem = imagens[0]
    assert imagem.data == png
    assert (imagem.width, imagem.height) == (512, 512)
    assert imagem.format == "png"
    assert progresso.valores[-1] == 100


@pytest.mark.parametrize(
    "base_url",
    ["http://gpu.local:8091", "http://gpu.local:8091/", "http://gpu.local:8091/v1"],
)
@pytest.mark.asyncio
async def test_remote_gpu_normaliza_a_url_para_v1_images_generations(base_url: str) -> None:
    posts: list[httpx.Request] = []
    transport = _fal_transport(
        response=httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(_png()).decode()}]}),
        posts=posts,
    )
    provider = RemoteGPUProvider(
        _settings(image_provider="remote_gpu", remote_gpu_url=base_url), transport=transport
    )
    await provider.generate(_req(), noop_progress)
    assert str(posts[0].url) == "http://gpu.local:8091/v1/images/generations"


@pytest.mark.asyncio
async def test_remote_gpu_sem_url_levanta_provider_error() -> None:
    provider = RemoteGPUProvider(_settings(image_provider="remote_gpu", remote_gpu_url=""))
    with pytest.raises(ProviderError) as erro:
        await provider.generate(_req(), noop_progress)
    assert "REMOTE_GPU_URL" in str(erro.value)


@pytest.mark.asyncio
async def test_remote_gpu_http_404_vira_provider_error() -> None:
    transport = _fal_transport(response=httpx.Response(404, text="Not Found"))
    provider = RemoteGPUProvider(
        _settings(image_provider="remote_gpu", remote_gpu_url="http://gpu.local:8091"),
        transport=transport,
    )
    with pytest.raises(ProviderError) as erro:
        await provider.generate(_req(), noop_progress)
    assert "404" in str(erro.value)


@pytest.mark.asyncio
async def test_remote_gpu_recusa_reference_images() -> None:
    provider = RemoteGPUProvider(
        _settings(image_provider="remote_gpu", remote_gpu_url="http://gpu.local:8091")
    )
    with pytest.raises(ProviderError):
        await provider.generate(_req(reference_images=["https://exemplo.com/a.png"]), noop_progress)


# ---------------------------------------------------------------------------
# get_provider / disponibilidade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nome", "classe", "overrides"),
    [
        ("fake", "FakeProvider", {}),
        ("local_comfy", "LocalComfyProvider", {"comfy_url": "http://comfy:8188"}),
        ("fal", "FalProvider", {"fal_key": "k"}),
        ("remote_gpu", "RemoteGPUProvider", {"remote_gpu_url": "http://gpu:8091"}),
    ],
)
@pytest.mark.asyncio
async def test_get_provider_escolhe_a_classe_por_env(
    nome: str, classe: str, overrides: dict[str, Any]
) -> None:
    provider = get_provider(_settings(image_provider=nome, **overrides))
    assert type(provider).__name__ == classe
    assert provider.name == nome
    assert isinstance(provider.model_id, str) and provider.model_id
    assert callable(provider.generate)


@pytest.mark.asyncio
async def test_get_provider_aceita_apelido_local_cpu() -> None:
    """O apelido IMAGE_PROVIDER=local_cpu tem de cair em local_comfy."""
    provider = get_provider(_settings(image_provider="local_cpu", comfy_url="http://comfy:8188"))
    assert provider.name == "local_comfy"
    assert type(provider).__name__ == "LocalComfyProvider"


@pytest.mark.parametrize(
    ("nome", "overrides", "esperado_no_erro"),
    [
        ("fal", {"fal_key": ""}, "FAL_KEY"),
        ("remote_gpu", {"remote_gpu_url": ""}, "REMOTE_GPU_URL"),
        ("local_comfy", {"comfy_url": ""}, "COMFY_URL"),
    ],
)
@pytest.mark.asyncio
async def test_get_provider_fail_fast_sem_configuracao(
    nome: str, overrides: dict[str, Any], esperado_no_erro: str
) -> None:
    settings = _settings(image_provider=nome, **overrides)

    disponivel, motivo = is_provider_available(nome, settings)
    assert disponivel is False
    assert esperado_no_erro in motivo

    with pytest.raises(ProviderError) as erro:
        get_provider(settings)
    assert esperado_no_erro in str(erro.value)


@pytest.mark.asyncio
async def test_get_provider_nome_invalido_levanta_provider_error() -> None:
    with pytest.raises(ProviderError):
        get_provider(_settings(image_provider="qwen-image-99.9"))


@pytest.mark.asyncio
async def test_get_provider_com_require_available_false_ignora_a_chave() -> None:
    """Escape hatch usado por /api/v1/config para listar metadados sem credencial."""
    provider = get_provider(_settings(image_provider="fal", fal_key=""), require_available=False)
    assert provider.name == "fal"


@pytest.mark.asyncio
async def test_providers_available_lista_so_o_que_esta_configurado() -> None:
    from app.providers import providers_available

    # fake e offline; local_comfy so exige COMFY_URL preenchida (nao checa se responde).
    so_offline = providers_available(_settings(image_provider="fake", comfy_url=""))
    assert so_offline == ["fake"]

    com_fal = providers_available(_settings(image_provider="fake", fal_key="k", comfy_url=""))
    assert com_fal == ["fake", "fal"]  # ordem canonica do contrato
    assert "remote_gpu" not in com_fal  # REMOTE_GPU_URL vazia
    assert "local_comfy" not in com_fal  # COMFY_URL vazia


# ---------------------------------------------------------------------------
# helpers de base compartilhados
# ---------------------------------------------------------------------------


def test_decode_base64_image_aceita_data_uri_e_base64_sem_padding() -> None:
    png = _png()
    encoded = base64.b64encode(png).decode()
    assert decode_base64_image(f"data:image/png;base64,{encoded}") == png
    assert decode_base64_image(encoded.rstrip("=")) == png


def test_decode_base64_image_rejeita_payload_vazio() -> None:
    with pytest.raises(ProviderError):
        decode_base64_image("")


def test_image_meta_rejeita_binario_que_nao_e_imagem() -> None:
    with pytest.raises(ProviderError):
        image_meta(b"isto nao e uma imagem")
    with pytest.raises(ProviderError):
        image_meta(b"")


def test_image_meta_le_dimensoes_e_formato() -> None:
    largura, altura, formato = image_meta(_png(48, 32))
    assert (largura, altura) == (48, 32)
    assert formato == "png"


def json_payload(request: httpx.Request) -> dict[str, Any]:
    """JSON do corpo de um httpx.Request (com assert de Content-Type)."""
    import json

    assert request.headers["Content-Type"].startswith("application/json")
    data = json.loads(request.content.decode("utf-8"))
    assert isinstance(data, dict)
    return data
