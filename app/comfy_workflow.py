"""Workflow Qwen-Image-2.1 (t2i) em **API-format**, para o provider `local_comfy`.

Formato exigido pelo POST /prompt do ComfyUI (comfy/server.py:1075-1146):
    {"prompt": { "<node_id>": {"class_type": "...", "inputs": {...}}, ... }, "client_id": "<uuid4>"}
NAO e o workflow UI-format (arrays nodes/links) — este modulo ja emite API-format.

------------------------------------------------------------------------------
CONTRATO DE NOS — extraido de `GET /object_info` (2026-09-21)
------------------------------------------------------------------------------
Os nomes de classe e de input abaixo nao sao estimativa: vem de um dump de
`GET /object_info` de um ComfyUI real, guardado cru em
docs/evidence/comfyui-object-info-subset.json. Toda regra deste cabecalho e
derivavel desse arquivo.

O TESTE PRATICO PONTA A PONTA **FOI EXECUTADO** (2026-09-21). Este grafo foi
submetido a um ComfyUI real (arm64, CPU, com o set GGUF Q4_K_M) e aceito de
primeira: `node_errors: {}`. A inferencia rodou e gerou imagem.

Medido nessa execucao: **129,74 s/step a 512²** (512²/4 steps ~13 min ponta a
ponta, incluindo ~4 min de carga dos 11 GB do disco), **5 GB** de RAM no pico
(23 GB disponiveis) e PNG 512x512 RGBA de 368.906 bytes na saida.

Mesmo funcionando, **nao vai rolar** nesta classe de maquina: o preset oficial
(1024², 25 steps) levaria ~3,6 h. O bloqueio e de **INFRAESTRUTURA** — VM ARM64
**sem GPU**, 4 vCPU sem bf16/i8mm — nao de modelo, codigo ou quantizacao. Por isso
`local_comfy` fica desligado por default. Numeros completos, o que foi tentado e o
que se extrapola: docs/LOCAL-INFERENCE.md.

Regras do contrato (todo input enviado tem de existir no node):
1. LOADER DO DiT: `UnetLoaderGGUF` (custom node **ComfyUI-GGUF**, fork do leejet) e
   o unico que carrega .gguf — o `UNETLoader` core nao carrega. O nome da CLASSE do
   custom node muda entre versoes: se divergir, `unet_loader=` cobre sem editar o
   modulo.
2. `TextEncodeQwenImage21` (comfy_extras/nodes_qwen.py) exige **cinco** inputs:
   `clip`, `prompt`, `negative_prompt`, `resolution` e `images`. `images` e um
   COMFY_AUTOGROW_V3 (0..16 referencias) e recebe `{}` para t2i puro; `resolution`
   so dimensiona imagem de referencia (0 = tamanho proprio). Mandar so
   {clip, prompt} faz o ComfyUI recusar o grafo.
3. O MESMO `TextEncodeQwenImage21` emite tres saidas: positive (0), negative (1) e
   latent (2). Como ele ja recebe `prompt` E `negative_prompt`, **um** node de
   encode basta: este modulo usa um so e o KSampler le as saidas 0 e 1.
4. `QwenImage21Cache` (mesmo modulo) exige `model`, `device` (auto/gpu/cpu/off) e
   `dtype` (default/int8/int4) — `dtype` e OBRIGATORIO (nao e opcional). Usamos
   cpu+int8: divide a cache pela metade com acuracia ~bf16, e a descricao do
   proprio node diz que `cpu` e prefetchado atras do compute, custando pouca
   velocidade.
5. `ResolutionSelector` NAO aceita width/height/batch_size: recebe `aspect_ratio`,
   `megapixels` e `multiple`, e devolve width/height (INTs) para alimentar o
   `EmptyLatentImage`. Ele nao substitui o node de latent. O caminho default aqui e
   o `EmptyLatentImage` com W/H explicitos (`use_resolution_selector=False`).
6. SAIDA: `SaveImage` (`images`, `filename_prefix`) — e o node de saida escolhido e
   o que o `local_comfy` procura no /history. `SaveImageAdvanced` NAO esta no subset
   do dump (nao foi conferido; tem combo dinamico em `format`): so entra via
   `save_node=`, nunca por default.
7. Parametros dos templates oficiais: 25 steps, cfg 1, sampler euler, scheduler
   simple.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CACHE_DTYPE",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_ENCODE_RESOLUTION",
    "DEFAULT_OUTPUT_NODE_IDS",
    "DEFAULT_RESOLUTION_MULTIPLE",
    "DEFAULT_SAMPLER_NODE_ID",
    "DIFFUSION_MODEL_GGUF",
    "DIFFUSION_MODEL_INT8",
    "NODE_ID_RESOLUTION",
    "NODE_SAVE_ADVANCED",
    "RESOLUTION_ASPECT_RATIOS",
    "TEXT_ENCODER",
    "VAE_MODEL",
    "CLIP_TYPE",
    "MINIMUM_MODEL_SET_GB",
    "build_workflow",
    "total_steps",
    "output_node_ids",
    "node_count",
    "VERIFY_NOTES",
]

# ---------------------------------------------------------------------------
# Nomes de arquivo REAIS (basename dentro de ComfyUI/models/)
# ---------------------------------------------------------------------------

#: diffusion_models/ — Q4_K_M do repo de terceiros Abiray/Qwen-Image-2.1-GGUF (4.189 GB).
#: ATENCAO AO NOME: o arquivo publicado no repo usa UNDERSCORES, e e esse nome que
#: scripts/download_models.sh grava em models/diffusion_models/. A variante com
#: hifens ("qwen-image-2.1-Q4_K_M.gguf") da 404 no download e nao existe no disco —
#: usar hifens aqui faria o ComfyUI recusar o workflow. Este e o nome que estava na
#: lista de `UnetLoaderGGUF.unet_name` no /object_info da maquina do teste.
DIFFUSION_MODEL_GGUF = "qwen_image_2.1_Q4_K_M.gguf"
#: diffusion_models/ — alternativa do template oficial day-0 (Comfy-Org, int8_convrot, 7.257 GB).
DIFFUSION_MODEL_INT8 = "qwen_image_2.1_int8_convrot.safetensors"
#: text_encoders/ — Qwen3-VL 8B w4a8 (Comfy-Org/Qwen-Image-2.1, 6.312 GB).
TEXT_ENCODER = "qwen3vl_8b_w4a8.safetensors"
#: vae/ — VAE RGBA 64 canais / 16x (0.676 GB).
VAE_MODEL = "qwen_image_2.1_vae_bf16.safetensors"

#: Tipo de CLIP usado pelo Qwen-Image no CLIPLoader (estava entre as 29 opcoes do combo).
CLIP_TYPE = "qwen_image"

#: Set minimo do preset GGUF: 4.189 + 6.312 + 0.676 = 11.177 GB.
MINIMUM_MODEL_SET_GB = 11.177

# ---------------------------------------------------------------------------
# Classes de node (todas CORE no template oficial, exceto o loader GGUF)
# ---------------------------------------------------------------------------

#: Classe do custom node ComfyUI-GGUF (fork leejet) — presente no dump de /object_info.
NODE_UNET_GGUF = "UnetLoaderGGUF"
NODE_UNET_STD = "UNETLoader"
NODE_CLIP = "CLIPLoader"
NODE_VAE = "VAELoader"
NODE_TEXT_ENCODE = "TextEncodeQwenImage21"
NODE_CACHE = "QwenImage21Cache"
NODE_RESOLUTION = "ResolutionSelector"
NODE_EMPTY_LATENT = "EmptyLatentImage"
NODE_KSAMPLER = "KSampler"
NODE_VAE_DECODE = "VAEDecode"
#: Node de saida default: inputs {images, filename_prefix} e nenhum combo dinamico.
NODE_SAVE = "SaveImage"
#: Existe no ComfyUI, mas o input `format` e um combo dinamico que nao foi conferido.
NODE_SAVE_ADVANCED = "SaveImageAdvanced"
#: Nome do SwitchNode dentro dos templates oficiais (comfy_extras/nodes_logic.py:86).
NODE_SWITCH = "ComfySwitchNode"

# ---------------------------------------------------------------------------
# IDs fixos dos nodes (local_comfy usa para achar o sampler/saida no /history)
# ---------------------------------------------------------------------------

NODE_ID_UNET = "1"
NODE_ID_CLIP = "2"
NODE_ID_VAE = "3"
NODE_ID_LATENT = "4"
#: Encode unico: positive = saida 0, negative = saida 1 (ver regra 3 do cabecalho).
NODE_ID_POSITIVE = "5"
#: RESERVADO — sem uso desde que o encode virou um node so. Mantido porque e nome
#: exportado do modulo (quem importar continua encontrando). ATENCAO ao contar:
#: o grafo default tem NOVE nodes (1,2,3,4,5,7,8,9,10) e este id '6' NAO entra na
#: conta — nao existe caminho que emita 8 nodes com as 9 classes do contrato. Os
#: oito nodes so aparecem com use_cache_node=False (o id 7 sai junto).
NODE_ID_NEGATIVE = "6"
NODE_ID_CACHE = "7"
NODE_ID_SAMPLER = "8"
NODE_ID_DECODE = "9"
NODE_ID_SAVE = "10"
#: ResolutionSelector (so existe no grafo quando use_resolution_selector=True).
NODE_ID_RESOLUTION = "11"

#: Node cujo `outputs.<id>.images` carrega o resultado no /history.
DEFAULT_OUTPUT_NODE_IDS: tuple[str, ...] = (NODE_ID_SAVE,)
#: Node do sampler (fonte dos steps para o progresso).
DEFAULT_SAMPLER_NODE_ID = NODE_ID_SAMPLER

# ---------------------------------------------------------------------------
# Defaults dos templates oficiais
# ---------------------------------------------------------------------------

DEFAULT_STEPS = 25
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_FILENAME_PREFIX = "qwen_2.1"

WEIGHT_DTYPE = "default"
CLIP_DEVICE = "default"
#: `device` do QwenImage21Cache: "cpu" e prefetchado atras do compute (custa pouca
#: velocidade) e e o que o grafo default usa.
CACHE_DEVICE = "cpu"
#: `dtype` do QwenImage21Cache: OBRIGATORIO. int8 = metade da cache, ~acuracia bf16.
CACHE_DTYPE = "int8"

#: `resolution` do TextEncodeQwenImage21 — inerte no t2i puro (mandamos images={}),
#: so dimensiona imagem de referencia. 512 e o valor default daqui.
DEFAULT_ENCODE_RESOLUTION = 512
#: `multiple` do ResolutionSelector (default do proprio node: 8).
DEFAULT_RESOLUTION_MULTIPLE = 8
#: Proporcoes do combo `aspect_ratio` do ResolutionSelector, na ordem do node,
#: com o valor numerico para derivar a opcao mais proxima de um width/height.
RESOLUTION_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1 (Square)", 1.0),
    ("2:3 (Portrait Photo)", 2 / 3),
    ("3:2 (Photo)", 3 / 2),
    ("3:4 (Portrait Standard)", 3 / 4),
    ("4:3 (Standard)", 4 / 3),
    ("9:16 (Portrait Widescreen)", 9 / 16),
    ("16:9 (Widescreen)", 16 / 9),
    ("21:9 (Ultrawide)", 21 / 9),
)
DEFAULT_ASPECT_RATIO = RESOLUTION_ASPECT_RATIOS[0][0]

VERIFY_NOTES: tuple[str, ...] = (
    "Contrato de nos extraido de GET /object_info de um ComfyUI real em 2026-09-21 "
    "(copia crua em docs/evidence/comfyui-object-info-subset.json). E a fonte dos nomes "
    "de classe e de input deste modulo — nao sao estimativa.",
    "TESTE PRATICO PONTA A PONTA EXECUTADO em 2026-09-21: este grafo foi aceito por um ComfyUI "
    "real com node_errors {} e gerou imagem. Medido: 129,74 s/step a 512², ~13 min ponta a ponta "
    "em 512²/4 steps, 5 GB de RAM no pico (de 23 GB). O bloqueio para uso interativo e de "
    "INFRAESTRUTURA (VM ARM64 sem GPU, 4 vCPU sem bf16/i8mm), nao de modelo, codigo ou "
    "quantizacao — a 1024²/25 steps seriam ~3,6 h. Registro completo em docs/LOCAL-INFERENCE.md.",
    "CONTAGEM DE NODES do grafo default: 9 (ids 1,2,3,4,5,7,8,9,10). O id '6' e reservado e "
    "nao entra; o QwenImage21Cache (id 7) entra ligado e nao existe classe extra nele. Com "
    "use_cache_node=False o grafo fica com 8 nodes — e o unico caminho que da 8.",
    "TextEncodeQwenImage21 leva os cinco inputs obrigatorios (clip, prompt, negative_prompt, "
    "resolution, images={}) e emite positive/negative/latent: UM node de encode basta.",
    "QwenImage21Cache exige `dtype` (default/int8/int4) — usamos cpu + int8.",
    "ResolutionSelector usa aspect_ratio/megapixels/multiple e alimenta o EmptyLatentImage; "
    "NAO substitui o node de latent (o default aqui e o EmptyLatentImage com W/H).",
    "SaveImage ({images, filename_prefix}) e a saida default; SaveImageAdvanced tem combo "
    "dinamico em `format`, nao conferido, e so entra via save_node=.",
    "O set GGUF exige o custom node ComfyUI-GGUF (fork leejet) instalado no container.",
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _text_encode_node(
    clip_id: str,
    text: str,
    negative: str,
    resolution: int,
) -> dict[str, Any]:
    """TextEncodeQwenImage21 com os CINCO inputs obrigatorios do /object_info.

    `images={}` = t2i puro (o COMFY_AUTOGROW_V3 aceita 0..16 referencias; o modulo
    nao faz edit). As saidas sao positive (0), negative (1) e latent (2).
    """
    return _node(
        NODE_TEXT_ENCODE,
        clip=[clip_id, 0],
        prompt=text,
        negative_prompt=negative,
        resolution=int(resolution),
        images={},
    )


def _closest_aspect_ratio(width: int, height: int) -> str:
    """Opcao do combo `aspect_ratio` mais proxima da proporcao width:height."""
    try:
        ratio = float(width) / float(height)
    except (TypeError, ValueError, ZeroDivisionError):
        return DEFAULT_ASPECT_RATIO
    if ratio <= 0:
        return DEFAULT_ASPECT_RATIO
    return min(RESOLUTION_ASPECT_RATIOS, key=lambda option: abs(option[1] - ratio))[0]


def _megapixels(width: int, height: int) -> float:
    """Megapixels do par width/height, dentro da faixa 0.1..16.0 do node."""
    try:
        value = (max(1, int(width)) * max(1, int(height))) / 1_000_000
    except (TypeError, ValueError):
        return 1.0
    return min(16.0, max(0.1, round(value, 2)))


def build_workflow(
    prompt: str,
    negative: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    seed: int = 0,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    batch_size: int = 1,
    *,
    unet_name: str = DIFFUSION_MODEL_GGUF,
    clip_name: str = TEXT_ENCODER,
    vae_name: str = VAE_MODEL,
    unet_loader: str | None = None,
    use_cache_node: bool = True,
    use_resolution_selector: bool = False,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    denoise: float = 1.0,
    clip_type: str = CLIP_TYPE,
    cache_device: str = CACHE_DEVICE,
    cache_dtype: str = CACHE_DTYPE,
    resolution: int = DEFAULT_ENCODE_RESOLUTION,
    save_node: str = NODE_SAVE,
) -> dict[str, dict[str, Any]]:
    """Monta o grafo t2i do Qwen-Image-2.1 em API-format.

    Retorna o dict que vai direto em {"prompt": <isso>, "client_id": ...} no POST /prompt.
    O default e o grafo lido do contrato de nos — que NUNCA foi submetido a um POST /prompt
    real (ver o cabecalho do modulo e docs/LOCAL-INFERENCE.md).
    """
    loader = unet_loader or (NODE_UNET_GGUF if str(unet_name).lower().endswith(".gguf") else NODE_UNET_STD)

    nodes: dict[str, dict[str, Any]] = {}
    nodes[NODE_ID_UNET] = _node(loader, unet_name=unet_name)

    nodes[NODE_ID_CLIP] = _node(
        NODE_CLIP,
        clip_name=clip_name,
        type=clip_type,
        device=CLIP_DEVICE,
    )
    nodes[NODE_ID_VAE] = _node(NODE_VAE, vae_name=vae_name)

    if use_resolution_selector:
        # ResolutionSelector (nodes_resolution.py): aspect_ratio/megapixels/multiple.
        # Nao tem width/height/batch_size — e devolve width/height para o latent.
        nodes[NODE_ID_RESOLUTION] = _node(
            NODE_RESOLUTION,
            aspect_ratio=_closest_aspect_ratio(width, height),
            megapixels=_megapixels(width, height),
            multiple=int(DEFAULT_RESOLUTION_MULTIPLE),
        )
        nodes[NODE_ID_LATENT] = _node(
            NODE_EMPTY_LATENT,
            width=[NODE_ID_RESOLUTION, 0],
            height=[NODE_ID_RESOLUTION, 1],
            batch_size=int(batch_size),
        )
    else:
        # Caminho verificado: W/H explicitos, determinista.
        nodes[NODE_ID_LATENT] = _node(
            NODE_EMPTY_LATENT,
            width=int(width),
            height=int(height),
            batch_size=int(batch_size),
        )

    # UM node de encode (regra 3): positive e negative saem dele; com cfg=1 o
    # sampler ignora o negative, mas o input e obrigatorio de qualquer forma.
    nodes[NODE_ID_POSITIVE] = _text_encode_node(
        NODE_ID_CLIP,
        prompt,
        negative or "",
        resolution,
    )
    positive_ref: list[Any] = [NODE_ID_POSITIVE, 0]
    negative_ref: list[Any] = [NODE_ID_POSITIVE, 1]

    model_ref: list[Any] = [NODE_ID_UNET, 0]
    if use_cache_node:
        # model/device/dtype sao os tres inputs obrigatorios do QwenImage21Cache.
        nodes[NODE_ID_CACHE] = _node(
            NODE_CACHE,
            model=model_ref,
            device=cache_device or CACHE_DEVICE,
            dtype=cache_dtype or CACHE_DTYPE,
        )
        model_ref = [NODE_ID_CACHE, 0]

    nodes[NODE_ID_SAMPLER] = _node(
        NODE_KSAMPLER,
        model=model_ref,
        positive=positive_ref,
        negative=negative_ref,
        latent_image=[NODE_ID_LATENT, 0],
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=float(denoise),
    )
    nodes[NODE_ID_DECODE] = _node(
        NODE_VAE_DECODE,
        samples=[NODE_ID_SAMPLER, 0],
        vae=[NODE_ID_VAE, 0],
    )
    nodes[NODE_ID_SAVE] = _node(
        save_node,
        images=[NODE_ID_DECODE, 0],
        filename_prefix=filename_prefix,
    )
    return nodes


def total_steps(workflow: dict[str, Any]) -> int:
    """Steps do KSampler do grafo (para o progresso do local_comfy)."""
    sampler = (workflow or {}).get(NODE_ID_SAMPLER) or {}
    try:
        return max(1, int(sampler.get("inputs", {}).get("steps", DEFAULT_STEPS)))
    except (TypeError, ValueError):
        return DEFAULT_STEPS


def output_node_ids(workflow: dict[str, Any]) -> tuple[str, ...]:
    """IDs dos nodes que salvam imagem (onde procurar o resultado no /history)."""
    if NODE_ID_SAVE in (workflow or {}):
        return (NODE_ID_SAVE,)
    return tuple(
        node_id
        for node_id, node in (workflow or {}).items()
        if isinstance(node, dict) and node.get("class_type") in (NODE_SAVE, NODE_SAVE_ADVANCED)
    )


def node_count(workflow: dict[str, Any]) -> int:
    return len(workflow or {})
