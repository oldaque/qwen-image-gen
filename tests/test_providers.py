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
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image, ImageFont

from app import comfy_workflow
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
# contrato de nodes do ComfyUI (app/comfy_workflow.py x dump de /object_info)
# ---------------------------------------------------------------------------

#: Subset cru de GET /object_info do ComfyUI real (2026-09-21). E a fonte de
#: verdade do contrato: nenhum input fora daqui pode ser enviado a um node.
#: ATENCAO: o dump e evidencia do CONTRATO, nao de um job executado — o teste
#: pratico ponta a ponta nao rodou (ver docs/LOCAL-INFERENCE.md).
OBJECT_INFO_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "evidence" / "comfyui-object-info-subset.json"
)


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    with OBJECT_INFO_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _contrato_do_node(spec: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(obrigatorios, permitidos) de um node no /object_info."""
    declarados = spec.get("input") or {}
    obrigatorios = set(declarados.get("required") or {})
    permitidos = obrigatorios | set(declarados.get("optional") or {})
    return obrigatorios, permitidos


def _assert_grafo_respeita_o_contrato(workflow: dict[str, Any], object_info: dict[str, Any]) -> None:
    """Todo node existe no dump, nenhum input sai do que ele declara, e todo link e valido."""
    assert workflow, "workflow vazio"
    for node_id, node in workflow.items():
        spec = object_info[node["class_type"]]  # KeyError = classe inexistente
        obrigatorios, permitidos = _contrato_do_node(spec)
        enviados = set(node["inputs"])
        assert enviados <= permitidos, (
            f"node {node_id} ({node['class_type']}) envia input desconhecido: "
            f"{sorted(enviados - permitidos)}"
        )
        assert obrigatorios <= enviados, (
            f"node {node_id} ({node['class_type']}) sem input obrigatorio: "
            f"{sorted(obrigatorios - enviados)}"
        )
        # Um link e [node_id, slot]: o node tem de existir e o slot tem de estar
        # dentro da aridade de saida que o dump declara para a classe dele.
        for entrada, valor in node["inputs"].items():
            if not (isinstance(valor, list) and len(valor) == 2):
                continue
            destino, slot = str(valor[0]), valor[1]
            assert destino in workflow, (
                f"node {node_id} ({node['class_type']}).{entrada} aponta para node inexistente: {destino}"
            )
            saidas = object_info[workflow[destino]["class_type"]].get("output", [])
            assert isinstance(slot, int) and 0 <= slot < len(saidas), (
                f"node {node_id} ({node['class_type']}).{entrada}: slot {slot} fora da aridade "
                f"de {destino} ({workflow[destino]['class_type']} tem {len(saidas)} saida(s))"
            )


def test_grafo_default_usa_so_inputs_do_contrato_verificado(object_info: dict[str, Any]) -> None:
    workflow = comfy_workflow.build_workflow("uma maca vermelha", width=512, height=512)
    _assert_grafo_respeita_o_contrato(workflow, object_info)


def test_grafo_com_resolution_selector_respeita_o_contrato(object_info: dict[str, Any]) -> None:
    workflow = comfy_workflow.build_workflow(
        "uma maca vermelha", width=1536, height=640, use_resolution_selector=True
    )
    _assert_grafo_respeita_o_contrato(workflow, object_info)


def test_grafo_default_reproduz_os_valores_do_contrato() -> None:
    """Valores exatos que o grafo default emite.

    Os nomes de classe/input vem do dump de /object_info; os valores (arquivos, device,
    dtype, steps, cfg, seed) sao os do grafo que foi aceito por um ComfyUI real em
    2026-09-21 com node_errors {} — ver docs/LOCAL-INFERENCE.md.
    """
    workflow = comfy_workflow.build_workflow(
        "a simple red apple on a wooden table, soft light",
        steps=4,
        seed=12345,
        filename_prefix="qwen21_test",
        width=512,
        height=512,
    )

    assert workflow[comfy_workflow.NODE_ID_UNET]["class_type"] == "UnetLoaderGGUF"
    assert workflow[comfy_workflow.NODE_ID_UNET]["inputs"] == {
        "unet_name": comfy_workflow.DIFFUSION_MODEL_GGUF
    }

    assert workflow[comfy_workflow.NODE_ID_CLIP]["inputs"] == {
        "clip_name": comfy_workflow.TEXT_ENCODER,
        "type": "qwen_image",
        "device": "default",
    }
    assert workflow[comfy_workflow.NODE_ID_VAE]["inputs"] == {"vae_name": comfy_workflow.VAE_MODEL}
    assert workflow[comfy_workflow.NODE_ID_LATENT]["class_type"] == "EmptyLatentImage"
    assert workflow[comfy_workflow.NODE_ID_LATENT]["inputs"] == {
        "width": 512,
        "height": 512,
        "batch_size": 1,
    }

    cache = workflow[comfy_workflow.NODE_ID_CACHE]["inputs"]
    assert cache == {"model": [comfy_workflow.NODE_ID_UNET, 0], "device": "cpu", "dtype": "int8"}

    sampler = workflow[comfy_workflow.DEFAULT_SAMPLER_NODE_ID]["inputs"]
    assert sampler["model"] == [comfy_workflow.NODE_ID_CACHE, 0]
    assert sampler["positive"] == [comfy_workflow.NODE_ID_POSITIVE, 0]
    assert sampler["negative"] == [comfy_workflow.NODE_ID_POSITIVE, 1]
    assert sampler["latent_image"] == [comfy_workflow.NODE_ID_LATENT, 0]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (12345, 4, 1.0)
    assert (sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]) == (
        "euler",
        "simple",
        1.0,
    )

    assert workflow[comfy_workflow.NODE_ID_DECODE]["inputs"] == {
        "samples": [comfy_workflow.NODE_ID_SAMPLER, 0],
        "vae": [comfy_workflow.NODE_ID_VAE, 0],
    }
    assert workflow[comfy_workflow.NODE_ID_SAVE]["inputs"] == {
        "images": [comfy_workflow.NODE_ID_DECODE, 0],
        "filename_prefix": "qwen21_test",
    }


def test_text_encode_leva_os_cinco_inputs_obrigatorios() -> None:
    workflow = comfy_workflow.build_workflow("uma maca", negative="borrado")
    encode = workflow[comfy_workflow.NODE_ID_POSITIVE]

    assert encode["class_type"] == "TextEncodeQwenImage21"
    # So {clip, prompt} faz o ComfyUI recusar o grafo (faltam 3 inputs obrigatorios).
    assert set(encode["inputs"]) == {
        "clip",
        "prompt",
        "negative_prompt",
        "resolution",
        "images",
    }
    assert encode["inputs"]["prompt"] == "uma maca"
    assert encode["inputs"]["negative_prompt"] == "borrado"
    assert encode["inputs"]["clip"] == [comfy_workflow.NODE_ID_CLIP, 0]
    assert encode["inputs"]["resolution"] == comfy_workflow.DEFAULT_ENCODE_RESOLUTION
    assert encode["inputs"]["images"] == {}  # COMFY_AUTOGROW_V3 vazio = t2i puro


def test_um_unico_encode_alimenta_positive_e_negative() -> None:
    workflow = comfy_workflow.build_workflow("uma maca", negative="borrado")

    encodes = [n for n in workflow.values() if n["class_type"] == "TextEncodeQwenImage21"]
    assert len(encodes) == 1  # o grafo validado usa UM node de encode
    assert comfy_workflow.NODE_ID_NEGATIVE not in workflow

    sampler = workflow[comfy_workflow.DEFAULT_SAMPLER_NODE_ID]["inputs"]
    assert sampler["positive"] == [comfy_workflow.NODE_ID_POSITIVE, 0]
    assert sampler["negative"] == [comfy_workflow.NODE_ID_POSITIVE, 1]  # saida 1 do mesmo node


def test_cache_off_liga_o_modelo_direto_no_sampler() -> None:
    workflow = comfy_workflow.build_workflow("uma maca", use_cache_node=False)
    assert comfy_workflow.NODE_ID_CACHE not in workflow
    sampler = workflow[comfy_workflow.DEFAULT_SAMPLER_NODE_ID]["inputs"]
    assert sampler["model"] == [comfy_workflow.NODE_ID_UNET, 0]


def test_contagem_de_nodes_do_grafo_default_e_nove() -> None:
    """9 ids ativos (1,2,3,4,5,7,8,9,10) para as 9 classes do contrato.

    8 nodes NAO existe com as 9 classes do contrato: a unica configuracao que da
    8 e `use_cache_node=False` (o id 7, QwenImage21Cache, sai). O id '6' e
    reservado e nunca entra na conta.
    """
    default = comfy_workflow.build_workflow("uma maca")
    assert comfy_workflow.node_count(default) == 9
    assert set(default) == {"1", "2", "3", "4", "5", "7", "8", "9", "10"}
    assert comfy_workflow.NODE_ID_NEGATIVE not in default
    assert len({n["class_type"] for n in default.values()}) == 9  # 1 classe por node

    sem_cache = comfy_workflow.build_workflow("uma maca", use_cache_node=False)
    assert comfy_workflow.node_count(sem_cache) == 8
    assert set(sem_cache) == {"1", "2", "3", "4", "5", "8", "9", "10"}

    com_selector = comfy_workflow.build_workflow("uma maca", use_resolution_selector=True)
    assert comfy_workflow.node_count(com_selector) == 10


@pytest.mark.parametrize(("width", "height", "esperado"), [
    (1024, 1024, "1:1 (Square)"),
    (1920, 1080, "16:9 (Widescreen)"),
    (1080, 1920, "9:16 (Portrait Widescreen)"),
])
def test_resolution_selector_nao_manda_width_height(
    width: int, height: int, esperado: str
) -> None:
    """O node so aceita aspect_ratio/megapixels/multiple; W/H vem do EmptyLatentImage."""
    workflow = comfy_workflow.build_workflow(
        "uma maca", width=width, height=height, use_resolution_selector=True
    )
    selector = workflow[comfy_workflow.NODE_ID_RESOLUTION]["inputs"]

    assert selector["aspect_ratio"] == esperado
    assert set(selector) == {"aspect_ratio", "megapixels", "multiple"}
    assert selector["multiple"] == comfy_workflow.DEFAULT_RESOLUTION_MULTIPLE

    latent = workflow[comfy_workflow.NODE_ID_LATENT]["inputs"]
    assert latent["width"] == [comfy_workflow.NODE_ID_RESOLUTION, 0]
    assert latent["height"] == [comfy_workflow.NODE_ID_RESOLUTION, 1]
    assert latent["batch_size"] == 1


def test_resolution_selector_desligado_nao_monta_o_node() -> None:
    workflow = comfy_workflow.build_workflow("uma maca", width=512, height=512)
    assert comfy_workflow.NODE_ID_RESOLUTION not in workflow
    assert workflow[comfy_workflow.NODE_ID_LATENT]["class_type"] == "EmptyLatentImage"


def test_save_image_e_a_saida_default() -> None:
    workflow = comfy_workflow.build_workflow("uma maca")
    assert workflow[comfy_workflow.NODE_ID_SAVE]["class_type"] == "SaveImage"
    assert comfy_workflow.DEFAULT_OUTPUT_NODE_IDS == (comfy_workflow.NODE_ID_SAVE,)
    assert comfy_workflow.output_node_ids(workflow) == (comfy_workflow.NODE_ID_SAVE,)
    assert comfy_workflow.DEFAULT_SAMPLER_NODE_ID == "8"


def test_save_image_advanced_so_entra_via_kwarg() -> None:
    workflow = comfy_workflow.build_workflow("uma maca", save_node=comfy_workflow.NODE_SAVE_ADVANCED)
    assert workflow[comfy_workflow.NODE_ID_SAVE]["class_type"] == "SaveImageAdvanced"
    assert comfy_workflow.output_node_ids(workflow) == (comfy_workflow.NODE_ID_SAVE,)


def test_loader_gguf_e_o_default_e_o_int8_usa_o_loader_core() -> None:
    gguf = comfy_workflow.build_workflow("uma maca")
    assert gguf[comfy_workflow.NODE_ID_UNET]["class_type"] == comfy_workflow.NODE_UNET_GGUF

    safetensors = comfy_workflow.build_workflow(
        "uma maca", unet_name=comfy_workflow.DIFFUSION_MODEL_INT8
    )
    assert safetensors[comfy_workflow.NODE_ID_UNET]["class_type"] == comfy_workflow.NODE_UNET_STD

    forcado = comfy_workflow.build_workflow("uma maca", unet_loader="UnetLoaderGGUF")
    assert forcado[comfy_workflow.NODE_ID_UNET]["class_type"] == "UnetLoaderGGUF"


def test_build_workflow_aceita_a_assinatura_publica_completa() -> None:
    """O provider local_comfy chama por kwargs: nada aqui pode virar obrigatorio."""
    workflow = comfy_workflow.build_workflow(
        "prompt",
        "negative",
        512,
        512,
        4,
        1.0,
        7,
        "pref",
        2,
        unet_name=comfy_workflow.DIFFUSION_MODEL_GGUF,
        clip_name=comfy_workflow.TEXT_ENCODER,
        vae_name=comfy_workflow.VAE_MODEL,
        unet_loader=None,
        use_cache_node=True,
        use_resolution_selector=False,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
        clip_type=comfy_workflow.CLIP_TYPE,
        cache_device="cpu",
        cache_dtype="int4",
        resolution=1024,
        save_node=comfy_workflow.NODE_SAVE,
    )
    assert workflow[comfy_workflow.NODE_ID_CACHE]["inputs"]["dtype"] == "int4"
    assert workflow[comfy_workflow.NODE_ID_POSITIVE]["inputs"]["resolution"] == 1024
    assert workflow[comfy_workflow.NODE_ID_LATENT]["inputs"]["batch_size"] == 2
    assert comfy_workflow.total_steps(workflow) == 4
    assert comfy_workflow.node_count(workflow) == 9


def test_verify_notes_e_cabecalho_registram_o_teste_pratico() -> None:
    """O registro diz o que veio do dump E o que o teste pratico mediu de verdade."""
    notas = "\n".join(comfy_workflow.VERIFY_NOTES)
    assert "2026-09-21" in notas
    assert "docs/LOCAL-INFERENCE.md" in notas
    assert "infraestrutura" in notas.lower()

    cabecalho = comfy_workflow.__doc__ or ""
    assert "docs/LOCAL-INFERENCE.md" in cabecalho
    assert "infraestrutura" in cabecalho.lower()

    # O teste ponta a ponta FOI executado: o grafo foi aceito por um servidor real
    # (node_errors {}) e o tempo por step foi medido. Esses numeros sao registro, nao
    # estimativa — o modulo precisa carregar-los para nao virar "nunca testado".
    for medido, fonte in (
        ("129,74", "s/step a 512² medido"),
        ("node_errors", "grafo aceito pelo servidor"),
        ("5 GB", "pico de RAM"),
    ):
        assert medido in cabecalho, f"cabecalho nao registra {fonte}"
        assert medido in notas, f"VERIFY_NOTES nao registra {fonte}"


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
