# Inferência local do Qwen-Image-2.1: teste prático e veredito

**Data do teste:** 2026-09-21
**Veredito curto:** **funcionou, e não vai rolar — por infraestrutura.**

O caminho `local_comfy` foi executado de ponta a ponta nesta classe de VM e **gerou uma imagem
real**. Mas a **129,74 s/step** medidos, uma imagem de 512² com 4 steps leva ~13 min e o preset
oficial (1024², 25 steps) levaria ~3,6 h. Para uso interativo isso está fora — e o motivo é o
hardware, não o modelo, o código ou a quantização.

Este documento registra o teste **com os números que ele produziu**, para que ninguém (nem nós, no
futuro) repita o download de 11 GB esperando outra coisa. Tudo que aparece como tempo ou consumo
abaixo foi medido nesta máquina; o que é extrapolação está marcado como tal.

---

## 1. Ambiente do teste

| Item | Valor |
|---|---|
| Plataforma | VM ARM64 (aarch64), **sem GPU** — único device de vídeo é um `Virtio 1.0 GPU` (display) |
| CPU | 4 vCPU ARM, flags expostas: `asimd`, `crc32` — **sem `bf16`, sem `i8mm`, sem `dotprod`** |
| RAM | 23 GB (pico da inferência: **5 GB** — ver seção 4) |
| Disco | 45 GB, I/O sequencial medido: **~53 MB/s** |
| Docker | 29.4.1 · Ubuntu 24.04 |
| ComfyUI | branch `master` (clonada no dia), `--cpu --listen 0.0.0.0 --port 8188` |
| Custom node | `ComfyUI-GGUF`, **fork do `leejet`** (o do `city96` está pouco mantido) |

Não existe imagem Docker arm64 pronta de ComfyUI utilizável — o projeto não publica imagem
oficial e as da comunidade são amd64 (a única arm64, `dustynv/comfyui`, é do `jetson-containers` e
depende de iGPU Tegra). A imagem foi construída na mão:

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /ComfyUI
# torch + torchvision + torchaudio NA MESMA origem. Instalar só o torch e deixar o
# requirements trazer o torchvision do PyPI quebra com:
#   RuntimeError: operator torchvision::nms does not exist
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /ComfyUI/requirements.txt
RUN git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF /ComfyUI/custom_nodes/ComfyUI-GGUF \
 && pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
RUN rm -rf /ComfyUI/models && ln -s /models /ComfyUI/models
EXPOSE 8188
CMD ["python","/ComfyUI/main.py","--cpu","--listen","0.0.0.0","--port","8188","--output-directory","/output"]
```

Imagem final: 2,61 GB.

---

## 2. Pesos usados (11,177 GB)

| Destino | Repo | Arquivo | Bytes |
|---|---|---|---|
| `diffusion_models/` | `Abiray/Qwen-Image-2.1-GGUF` | `qwen_image_2.1_Q4_K_M.gguf` | 4.189.343.904 |
| `text_encoders/` | `Comfy-Org/Qwen-Image-2.1` | `qwen3vl_8b_w4a8.safetensors` | 6.312.105.364 |
| `vae/` | `Comfy-Org/Qwen-Image-2.1` | `qwen_image_2.1_vae_bf16.safetensors` | 675.509.688 |

Download completo: **5 min 43 s** (~34 MB/s efetivos), sem token (repos públicos, `gated=false`).

**Header do GGUF (lido do arquivo baixado):** magic `GGUF` v3, 265 tensores,
`general.architecture = qwen_image`, `general.file_type = 15` (Q4_K_M).

Esse detalhe decide o caminho: `qwen_image` **está** na `IMG_ARCH_LIST` do fork, então ele aceita.
Se declarasse `qwen_image21` — o arch que o próprio `tools/convert.py` do fork gera — seria
recusado com *"Unexpected architecture type in GGUF file"*, porque `qwen_image21` não está na
lista de carregamento do loader. Vale conferir o header antes de baixar 4 GB (snippet na seção 5).

---

## 3. Contrato de nós verificado (`/object_info` da máquina real)

Levantado consultando `GET /object_info` do ComfyUI em execução. Substitui os nomes estimados que o
código trazia antes. Cópia crua em [`docs/evidence/comfyui-object-info-subset.json`](evidence/comfyui-object-info-subset.json).

| Nó | Entradas relevantes | Saídas |
|---|---|---|
| `UnetLoaderGGUF` | `unet_name` (required) | `MODEL` |
| `CLIPLoader` | `clip_name`, `type` (required); `device` (optional) — usar **`qwen_image`** | `CLIP` |
| `VAELoader` | `vae_name` | `VAE` |
| `QwenImage21Cache` | `model`, `device` (`auto`/`gpu`/`cpu`/`off`), `dtype` (`default`/`int8`/`int4`) — **os três são required** | `MODEL` |
| `TextEncodeQwenImage21` | `clip`, `prompt`, `negative_prompt`, `resolution`, `images` (COMFY_AUTOGROW_V3, 0..16 refs); `vae` (optional) | `positive`, `negative`, `latent` |
| `EmptyLatentImage` | `width`, `height`, `batch_size` | `LATENT` |
| `KSampler` | `model`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `positive`, `negative`, `latent_image`, `denoise` | `LATENT` |
| `VAEDecode` | `samples`, `vae` | `IMAGE` |
| `SaveImage` | `images`, `filename_prefix` | `IMAGE` |
| `ResolutionSelector` | `aspect_ratio`, `megapixels`, `multiple` (**não tem width/height/batch_size**); `preview` (optional) | `INT`, `INT` |

Usamos `QwenImage21Cache` com `device=cpu` e `dtype=int8` — a descrição do próprio nó diz que `cpu`
é prefetchado atrás do compute e custa pouca velocidade, que `int8` divide a cache pela metade com
acurácia próxima de bf16, e que `int4` divide por quatro ao custo de ~dobrar o erro por step.

O grafo montado com esse contrato foi aceito pelo servidor de primeira:
`{"prompt_id": "...", "node_errors": {}}`.

Os testes do repo travam o grafo contra o dump: todo `required` presente, nenhum input fora de
`required`+`optional`, toda referência `[node_id, slot]` apontando para node existente com índice
dentro da aridade de saída declarada, e os valores exatos do grafo que foi aceito (arquivos, device,
dtype, steps, cfg, seed). O que os testes **não** cobrem: a execução em si — isso é o que este
documento registra.

---

## 4. Resultados medidos

| Medida | Valor |
|---|---|
| Carga do text encoder | 6.019,50 MB, `dtype torch.float16`, `MixedPrecisionOps` |
| Carga do DiT (GGUF) | 4.187,25 MB — `loaded completely; full load: True` |
| qtypes do GGUF | BF16 (3), Q4_K (165), F32 (65), Q6_K (32) |
| Carga dos 11 GB a partir do disco | ~4 min |
| **Tempo por step (512²)** | **129,74 s** |
| Sampling 512² / 4 steps | 8 min 38 s |
| **Total ponta a ponta (512² / 4 steps)** | **~13 min** |
| RAM no pico | **5 GB de 23 GB** |
| Saída | PNG **512×512 RGBA**, 368.906 bytes |

Prompt do teste: `a simple red apple on a wooden table, soft light`. A imagem é coerente e
reconhecível — o caminho local não tem problema de qualidade, tem problema de tempo.

### Extrapolação (mesma máquina, não medido)

O custo por step escala com o número de tokens de imagem (4× de 512² para 1024²), e a atenção
cresce mais rápido que isso — então estes valores são **piso**:

| Configuração | Estimativa |
|---|---|
| 512² / 4 steps | **~13 min** (medido) |
| 512² / 8 steps | ~21 min |
| 1024² / 4 steps | ~39 min |
| 1024² / 25 steps (preset oficial) | **~3,6 h** |
| 2048² (resolução nativa do modelo) | ~35 min **por step** |

---

## 5. Roteiro de reprodução

```bash
# 1. imagem arm64 (não existe pronta) — Dockerfile na seção 1
docker build -t comfyui-arm64:cpu .

# 2. pesos (11,177 GB) — baixe direto na máquina de destino, não no laptop
./scripts/download_models.sh /caminho/para/models

# 3. confira o header do GGUF antes de tudo
python3 - <<'EOF'
import struct
f = open("models/diffusion_models/qwen_image_2.1_Q4_K_M.gguf", "rb")
f.read(8); struct.unpack("<QQ", f.read(16))
n = struct.unpack("<Q", f.read(8))[0]; k = f.read(n).decode()
t = struct.unpack("<I", f.read(4))[0]
n = struct.unpack("<Q", f.read(8))[0]; print(k, "=", f.read(n).decode())
EOF
# esperado: general.architecture = qwen_image   (se for qwen_image21, o fork GGUF recusa)

# 4. suba o ComfyUI (bind em loopback; o servidor não tem auth)
docker run -d --name comfy -p 127.0.0.1:8188:8188 \
  -v /caminho/para/models:/models -v /caminho/para/output:/output comfyui-arm64:cpu
curl -s localhost:8188/object_info | python3 -m json.tool | grep -A2 UnetLoaderGGUF

# 5. primeiro job: 512² com 4 steps, e espere ~13 min
```

O grafo validado está em `app/comfy_workflow.py`; o provider que o consome, em
`app/providers/local_comfy.py`.

---

## 6. Por que não vai rolar nesta classe de hardware

- **Sem GPU e sem matmul acelerado na CPU.** Com `asimd`/`crc32` apenas, o produto de matrizes cai
  em fp32 puro: ~0,1-0,2 TFLOP/s efetivos nestes 4 vCPU, contra ~80 TFLOP/s fp16 de uma GPU de
  consumo — **~800× de diferença**.
- **Difusão não tem cache de KV.** Cada step reprocessa **todos** os tokens de imagem por todas as
  camadas; não existe o caminho "decode" barato de um LLM. Uma imagem de 1024²/25 steps equivale a
  25 prefills completos.
- **Quantização reduz memória, não FLOPs.** Sem `i8mm`, matmul int8 não é mais rápido que fp32 —
  pode ser mais lento. O default do template oficial (`int8_convrot`) cai no `torch._int_mm` a
  ~4,2 GFLOPS, o pior caso nesta arquitetura. A rota GGUF Q4 existe para **caber**, não para
  acelerar.
- **RAM nunca foi o gargalo.** O pico medido foi **5 GB de 23 GB**. As estimativas de 36-44 GB para
  bf16 estavam corretas, mas o set quantizado + cache `int8` derruba o consumo residente para perto
  de 10 GB, e a medição veio abaixo disso. Quem for dimensionar isso de novo: o problema é compute.

Nenhuma escolha de software contorna isso — nenhuma quantização cria unidades de tensor que não
existem no silício.

---

## 7. Caminhos que servem

| Caminho | Tempo por imagem | Observação |
|---|---|---|
| API hospedada (`fal`, provider `fal`) | ~10 s | exige chave; ~$0,021/imagem @1024² |
| GPU alugada/dedicada (`remote_gpu`, vLLM-Omni) | segundos | pesos próprios; ~$0,34-0,80/h alugando |

O provider `local_comfy` permanece no código, **desligado por padrão**, com o grafo idêntico ao que
foi validado aqui. Quem tiver GPU (ou muita paciência) liga com uma variável de ambiente.
