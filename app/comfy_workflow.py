"""Workflow Qwen-Image-2.1 (t2i) em **API-format**, para o provider `local_comfy`.

Formato exigido pelo POST /prompt do ComfyUI (comfy/server.py:1075-1146):
    {"prompt": { "<node_id>": {"class_type": "...", "inputs": {...}}, ... }, "client_id": "<uuid4>"}
NAO e o workflow UI-format (arrays nodes/links) — este modulo ja emite API-format.

------------------------------------------------------------------------------
NOTA DE COMPATIBILIDADE (o JSON do template varia com a versao instalada)
------------------------------------------------------------------------------
Antes do primeiro job em um ComfyUI recem-instalado, valide nomes de classe e de
input com `GET /object_info` (o servidor e a fonte de verdade) ou copie o template
oficial day-0 e ajuste os IDs deste modulo.
1. LOADER DO DiT: o template oficial day-0 aponta para
   `qwen_image_2.1_int8_convrot.safetensors` carregado por `UNETLoader`. Aqui o default
   e a rota GGUF Q4_K_M, a opcao viavel em CPU ARM — mas o `UNETLoader` padrao NAO
   carrega .gguf: exige o custom node **ComfyUI-GGUF** (fork do leejet, nao o city96)
   instalado no container. O nome da CLASSE desse loader muda entre versoes do custom
   node, por isso `NODE_UNET_GGUF` deve ser lido em `GET /object_info` (procure
   "GGUF") e ajustado se necessario.
2. `TextEncodeQwenImage21` (comfy_extras/nodes_qwen.py:111): aqui usamos
   {"clip", "prompt"} (padrao dos text-encode do ComfyUI). Se o node exigir mais
   campos (ex.: "vae"/"image"), leia `GET /object_info` e ajuste
   `_text_encode_node()`.
3. `QwenImage21Cache` (nodes_qwen.py:185): o input `device` aceita "cpu" e as docs
   dizem que o KV prefix cache custa pouca velocidade — em CPU, com muitos steps, o
   cache importa muito, por isso ele fica LIGADO por default (`use_cache_node=True`).
   Se a classe nao existir no ComfyUI instalado, chame build_workflow(use_cache_node=False).
4. `SaveImageAdvanced` (nodes_images.py:1725): usamos {"images", "filename_prefix"}.
   Confira se a classe pede tambem "format"/"extension"/"quality" em `GET /object_info`.
5. `ResolutionSelector` (nodes_resolution.py:30) existe no template oficial, mas aqui
   usamos `EmptyLatentImage` (mais simples e determinista para W/H explicitos).
   Ligue com `use_resolution_selector=True` se preferir seguir o template.
6. Parametros dos templates oficiais: 25 steps, cfg 1, sampler euler, scheduler simple.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_OUTPUT_NODE_IDS",
    "DEFAULT_SAMPLER_NODE_ID",
    "DIFFUSION_MODEL_GGUF",
    "DIFFUSION_MODEL_INT8",
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
#: usar hifens aqui faria o ComfyUI recusar o workflow.
DIFFUSION_MODEL_GGUF = "qwen_image_2.1_Q4_K_M.gguf"
#: diffusion_models/ — alternativa do template oficial day-0 (Comfy-Org, int8_convrot, 7.257 GB).
DIFFUSION_MODEL_INT8 = "qwen_image_2.1_int8_convrot.safetensors"
#: text_encoders/ — Qwen3-VL 8B w4a8 (Comfy-Org/Qwen-Image-2.1, 6.312 GB).
TEXT_ENCODER = "qwen3vl_8b_w4a8.safetensors"
#: vae/ — VAE RGBA 64 canais / 16x (0.676 GB).
VAE_MODEL = "qwen_image_2.1_vae_bf16.safetensors"

#: Tipo de CLIP usado pelo Qwen-Image no CLIPLoader.
CLIP_TYPE = "qwen_image"

#: Set minimo do preset GGUF: 4.189 + 6.312 + 0.676 = 11.177 GB.
MINIMUM_MODEL_SET_GB = 11.177

# ---------------------------------------------------------------------------
# Classes de node (todas CORE no template oficial, exceto o loader GGUF)
# ---------------------------------------------------------------------------

#: Classe do custom node ComfyUI-GGUF (fork leejet). O nome varia com a versao
#: instalada: confirme com `GET /object_info` (procure "GGUF") antes do 1o job.
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
NODE_SAVE = "SaveImageAdvanced"
#: Nome do SwitchNode dentro dos templates oficiais (comfy_extras/nodes_logic.py:86).
NODE_SWITCH = "ComfySwitchNode"

# ---------------------------------------------------------------------------
# IDs fixos dos nodes (local_comfy usa para achar o sampler/saida no /history)
# ---------------------------------------------------------------------------

NODE_ID_UNET = "1"
NODE_ID_CLIP = "2"
NODE_ID_VAE = "3"
NODE_ID_LATENT = "4"
NODE_ID_POSITIVE = "5"
NODE_ID_NEGATIVE = "6"
NODE_ID_CACHE = "7"
NODE_ID_SAMPLER = "8"
NODE_ID_DECODE = "9"
NODE_ID_SAVE = "10"

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
CACHE_DEVICE = "cpu"

VERIFY_NOTES: tuple[str, ...] = (
    "Confirme a classe do loader GGUF (NODE_UNET_GGUF) com GET /object_info (procure 'GGUF').",
    "Confirme os inputs de TextEncodeQwenImage21 com GET /object_info — usamos {clip, prompt}.",
    "Confirme os inputs de QwenImage21Cache com GET /object_info — usamos {model} (+ device='cpu').",
    "Confirme se SaveImageAdvanced exige campos extra de formato com GET /object_info.",
    "O set GGUF exige o custom node ComfyUI-GGUF (fork leejet) instalado no container.",
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _text_encode_node(clip_id: str, text: str) -> dict[str, Any]:
    """TextEncodeQwenImage21 — inputs {clip, prompt} (validar em GET /object_info)."""
    return _node(NODE_TEXT_ENCODE, clip=[clip_id, 0], prompt=text)


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
) -> dict[str, dict[str, Any]]:
    """Monta o grafo t2i do Qwen-Image-2.1 em API-format.

    Retorna o dict que vai direto em {"prompt": <isso>, "client_id": ...} no POST /prompt.
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
        # inputs do ResolutionSelector (nodes_resolution.py:30) — validar em /object_info.
        nodes[NODE_ID_LATENT] = _node(
            NODE_RESOLUTION,
            width=int(width),
            height=int(height),
            batch_size=int(batch_size),
        )
    else:
        nodes[NODE_ID_LATENT] = _node(
            NODE_EMPTY_LATENT,
            width=int(width),
            height=int(height),
            batch_size=int(batch_size),
        )

    nodes[NODE_ID_POSITIVE] = _text_encode_node(NODE_ID_CLIP, prompt)
    # cfg=1 ignora o unconditional; ainda assim montamos o node negativo (o template
    # oficial segue o mesmo padrao) e, se nao houver negative, reaproveitamos o positivo.
    if (negative or "").strip():
        nodes[NODE_ID_NEGATIVE] = _text_encode_node(NODE_ID_CLIP, negative)
        negative_ref: list[Any] = [NODE_ID_NEGATIVE, 0]
    else:
        negative_ref = [NODE_ID_POSITIVE, 0]

    model_ref: list[Any] = [NODE_ID_UNET, 0]
    if use_cache_node:
        cache_inputs: dict[str, Any] = {"model": model_ref}
        if cache_device:
            cache_inputs["device"] = cache_device
        nodes[NODE_ID_CACHE] = _node(NODE_CACHE, **cache_inputs)
        model_ref = [NODE_ID_CACHE, 0]

    nodes[NODE_ID_SAMPLER] = _node(
        NODE_KSAMPLER,
        model=model_ref,
        positive=[NODE_ID_POSITIVE, 0],
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
        NODE_SAVE,
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
        if isinstance(node, dict) and node.get("class_type") == NODE_SAVE
    )


def node_count(workflow: dict[str, Any]) -> int:
    return len(workflow or {})
