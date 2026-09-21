# image-gen — gerador de imagens com Qwen-Image 2.1

Webapp de geracao de imagens: FastAPI + fila assincrona (asyncio) + SQLite, com UI
propria servida pelo proprio app. Quatro backends de geracao, todos atras da mesma
interface de provider, escolhidos por **uma** variavel de ambiente.

- **API**: `POST /api/v1/jobs` cria o job e responde `202` na hora; o progresso chega
  por SSE (`/api/v1/jobs/{id}/events`) ou por polling. SQLite e a fonte de verdade do
  status (se o processo cair, job `running` volta para `queued` no boot).
- **UI**: SPA em `web/` (tema escuro, texto pt-BR) servida em `/` — sem build step,
  sem Node, sem framework.
- **Providers**: `fake` (Pillow, offline), `local_comfy` (ComfyUI na propria VM),
  `fal` (fal.ai, pago, segundos), `remote_gpu` (vLLM-Omni/SGLang em GPU alugada).

```
navegador ──> FastAPI (/, /api/v1/*) ──> fila asyncio ──> provider ──> IMAGES_DIR + SQLite
```

---

## Leia isto antes de tudo: a realidade de hardware

O alvo deste projeto e uma **VM ARM64 sem GPU**, com poucos vCPU (~4) e ~23 GB de RAM,
alem de ~40 GB de disco livre. Nessa maquina, **gerar localmente e viavel, mas lento**.
Os numeros abaixo sao estimativas para essa classe de hardware: as medicoes de GEMM
desta secao sao diretas, mas os tempos de geracao sao extrapolacao de ancoras x86/ARM
(nao existe benchmark publicado do Qwen-Image-2.1 em CPU ARM), com incerteza de **±2-3x**.

### Tempo por imagem no caminho local (`local_comfy` em CPU arm64)

| Resolucao | Estimativa | Observacao |
|---|---|---|
| 512² / 8 steps | ~3-15 min | melhor caso defensavel; ancoras x86 extrapoladas |
| **1024²** | **~30 min a ~4,5 h** | faixa estimada: 30-90 min (set quantizado), 2h20-3h40 (GGUF Q4_K, 40 steps), ~4,5 h (ComfyUI fp32, 25 steps) |
| 2048² | ~11-18 h | atencao de 16.384 tokens domina |

Custo fixo por imagem, mesmo com o DiT instantaneo: text encoder Qwen3-VL-8B
~20-60 s/prompt (cacheavel entre jobs) + decode do VAE 64 canais/16x ~20-60 s.

### RAM: o bf16 nao cabe, o quantizado cabe

| Configuracao | Pesos residentes | Cabe na RAM disponivel (~23 GB)? |
|---|---|---|
| bf16 completo (como o template oficial sugere) | 32,4 GB (+3-10 GB de ativacao = **36-44 GB**) | **nao** — estoura ~2x |
| set quantizado deste repo (GGUF Q4_K_M + TE w4a8) | ~10-12 GB; pico ~13-16 GB @1024² e ~16-21 GB @2048² | **sim** |
| com PE-T2I/PE-I2I co-residentes | +18,8 GB | nao (pular esses pesos) |

Ou seja: **o gargalo aqui e compute dos poucos cores ARM, nao RAM** — desde que voce use o set
quantizado. Se subir bf16, o gargalo passa a ser memoria e nada termina.

### Disco: 11,2 GB, nao 33 GB

| Item | Tamanho |
|---|---|
| **Set minimo escolhido** (GGUF Q4_K_M 4,189 + TE w4a8 6,312 + VAE 0,676) | **11,177 GB** |
| Set int8_convrot (default do template oficial) | 17,284 GB |
| bf16 completo | 32,44 GB |
| Repo oficial `Qwen/Qwen-Image-2.1` inteiro | 33,135 GB (so passa sem o cache do HF; com ele o pico excede os ~40 GB livres) |
| Repo `Comfy-Org/Qwen-Image-2.1` inteiro | 74,302 GB (nao cabe) |

Com cache do HuggingFace durante o download, reserve 40-80 GB livres; depois de baixar,
`rm -rf models/.hf-cache models/.cache`.

### Por que quantizado, e por que GGUF

No CPU arm64 o PyTorch **nao tem** caminho rapido para int8/bf16: o `torch._int_mm`
(usado pelo template oficial `qwen_image_2.1_int8_convrot.safetensors`) mede ~4,2 GFLOPS,
e fp16/bf16 sao emulados em software (~250-550x mais lentos que fp32). Nesta classe de
hardware (aarch64, PyTorch 2.13, 4 threads) o GEMM medido e: fp32 ~1012 GFLOPS,
fp16 ~1,85, bf16 ~1,83, int8 ~4,2. O default do template oficial e, literalmente, o
pior caso nesta arquitetura.
A rota escolhida e GGUF Q4_K_M (190-310 GFLOPS efetivos via SDOT), com o custom node
**ComfyUI-GGUF**. `.gguf` **nao** carrega no `UNETLoader` padrao.

**Confirme o loader GGUF no seu ComfyUI antes de montar o workflow.** A classe do node
varia com a build do ComfyUI e com a versao do custom node instalado, entao interrogue o
proprio servidor em vez de assumir o nome:

```bash
curl -s http://comfy:8188/object_info | jq -r 'keys[] | select(test("gguf"; "i"))'
```

A classe retornada (tipicamente `UnetLoaderGGUF`) e o nome que o grafo em
`app/comfy_workflow.py` precisa referenciar. Se o `object_info` nao listar nenhuma classe
GGUF, o custom node nao esta instalado e o workflow nao carrega o `.gguf`.

### E a imagem Docker do ComfyUI?

**Nao existe imagem arm64 utilizavel.** O time do ComfyUI nao publica imagem oficial, e
as imagens da comunidade publicadas para ComfyUI sao todas amd64 (`comfyui/base`,
`yanwk/comfyui-boot`, `ghcr.io/ai-dock/comfyui`, `obeliks/comfyui`...). A unica arm64
real, `dustynv/comfyui`, e do projeto jetson-containers (L4T/JetPack 6 + iGPU Tegra) e
nao serve numa VM ARM64 sem GPU. Por isso o servico `comfy` do compose usa
`COMFY_IMAGE` (um **placeholder**): voce builda a sua (~15 linhas de Dockerfile, o
esboco esta comentado dentro do `docker-compose.dokploy.yml`).

---

## Providers

| Provider | O que e | Custo | Latencia 1024² | Precisa de |
|---|---|---|---|---|
| `fake` | PNG sintetico com Pillow (gradiente + prompt + seed desenhados). **Nunca** acessa a rede | R$ 0 | ~3 s (simulado) | nada |
| `local_comfy` | ComfyUI na propria VM, set GGUF Q4_K_M | custo da VM | **30 min a ~4,5 h** | pesos baixados + imagem arm64 + ComfyUI no ar |
| `fal` | fal.ai, endpoint sincrono `fal.run` (o qwen-image-2.1 esta disponivel na plataforma) | **$0,02/megapixel de saida** = $0,021 @1024², $0,047 @1536², $0,084 @2048²; 4 imagens @1024² ≈ $0,08. Edit: ~$0,0367/MP de entrada + de saida (~$0,0733 p/ 1 MP in + 1 MP out) | segundos | `FAL_KEY` |
| `remote_gpu` | GPU alugada com vLLM-Omni (`vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091`), rota OpenAI-compat `/v1/images/generations` | $0,5-2/h de GPU 48 GB | segundos a ~1 min | `REMOTE_GPU_URL` (+ `REMOTE_GPU_KEY`) |

Regras que valem para todos:

- Falta de configuracao (chave/URL) e **fail fast**: `POST /api/v1/jobs` responde `400`
  na hora, com a mensagem dizendo exatamente qual variavel falta. (`local_comfy` e a
  excecao pratica: `COMFY_URL` tem default e o compose sempre a define, entao o `400` so
  aparece se ela for esvaziada a mao — com a URL presente mas o servidor fora, o job e
  aceito e falha depois.)
- `GET /api/v1/health` responde `200` mesmo com o provider quebrado (e o healthcheck do
  container); `GET /api/v1/config` lista em `providers_available` so o que esta
  configurado.
- `remote_gpu` **nao** aceita `reference_images` (a rota e text-to-image); `fal` usa
  `FAL_EDIT_MODEL` com `image_urls` quando ha referencias; `local_comfy` tambem recusa
  referencias (edicao com 2+ imagens da OOM em qualquer tamanho).

### Trocar de provider = 1 variavel de ambiente

```bash
IMAGE_PROVIDER=fake          # default: sobe e funciona sem chave e sem download
IMAGE_PROVIDER=fal           # + FAL_KEY=...
IMAGE_PROVIDER=remote_gpu    # + REMOTE_GPU_URL=https://gpu.exemplo.com/v1 (+ REMOTE_GPU_KEY)
IMAGE_PROVIDER=local_comfy   # + COMFY_URL=http://comfy:8188 (perfil "local" do compose)
```

Apelidos tambem sao aceitos: `local_cpu`, `comfyui`, `vllm`, `fal_ai`, `mock`.

---

## Quickstart local (sem Docker)

Requer Python 3.12+.

```bash
git clone <este-repo> && cd image-gen
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# opcional: copiar o modelo de configuracao (o default ja e fake + /data)
cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

Abra <http://127.0.0.1:8000>. Como `DATA_DIR=/data` nao existe na sua maquina, rode com
um diretorio local:

```bash
DATA_DIR=./data uvicorn app.main:app --reload --port 8000
```

Teste de fumaca ponta-a-ponta (sobe um uvicorn de verdade com `IMAGE_PROVIDER=fake` numa
porta livre, cria um job, espera o status terminal, baixa a imagem e valida o PNG —
sem tocar no seu `.env`, no banco do volume nem em `/data`):

```bash
scripts/smoke_test.sh              # cria venv temporario e instala as deps
scripts/smoke_test.sh --reuse      # reusa <repo>/.venv (pula o pip install)
scripts/smoke_test.sh -p 8123 -t 300 -k    # porta fixa, timeout maior, guarda o tmp
```

### Testes

```bash
pytest -q        # ~60 testes, sem rede e sem Docker (provider fake + httpx.MockTransport)
```

- `tests/test_api.py` — fluxo completo (criar -> aguardar `succeeded` -> baixar PNG ->
  historico -> delete), validacao (prompt vazio, `width` fora de faixa, `num_images=5`),
  404, `/health`, `/config`, cancelamento e SSE.
- `tests/test_providers.py` — `fake` gera PNG valido (assinatura + dimensoes via Pillow);
  `fal` monta payload/headers e parseia a resposta com `httpx.MockTransport`;
  `remote_gpu` decodifica `b64_json`; `get_provider` escolhe a classe por env e levanta
  `ProviderError` quando falta chave.

---

## Baixar os modelos na VM (so para `local_comfy`)

```bash
scripts/download_models.sh --dir ./models --install-accel --volume <projeto>_models
```

- **Set minimo: 11,177 GB** (GGUF Q4_K_M 4,189 + TE w4a8 6,312 + VAE 0,676), dos repos
  `Abiray/Qwen-Image-2.1-GGUF` e `Comfy-Org/Qwen-Image-2.1`. Ja o repo oficial completo
  (33,1 GB) e o Comfy-Org completo (74,3 GB) **nao cabem** no espaco livre (~40 GB) se o
  cache do HuggingFace duplicar o download: os 33,1 GB so passam sem cache, os 74,3 GB
  nunca.
- **Downloads anonimos**: `gated=false` nos quatro repos, sem token HF. Com disco lento
  (I/O sequencial na casa das dezenas de MB/s) o set leva ~4 min no melhor caso; conte
  5-15 min com verificacao de tamanho.
- O script e retomavel (`hf_transfer`/`curl -C -`), valida o tamanho de cada arquivo e
  imprime o tamanho exato de cada um no resumo final.
- Opcoes: `-d DIR` (destino, default `./models`), `-o dit,te,vae` (subset),
  `--volume NAME` (copia para o volume docker), `--no-verify`, `--force` (ignora disco
  apertado).
- Ao final, limpe o cache: `rm -rf models/.hf-cache models/.cache`.

Ordem de grandeza em disco: 11,2 GB de pesos + ~40 GB livres = folga; nao tente o set
bf16 (32,4 GB) nem o TE bf16 (17,5 GB) neste hardware.

---

## Deploy no Dokploy (v0.30.7)

Pre-requisitos no servidor: Docker + Dokploy instalados; a rede `dokploy-network` ja
existe (o Dokploy a cria na instalacao).

1. **Crie o projeto**: `Projects` -> `Create Project` (ex.: `image-gen`).
2. **Crie o servico**: dentro do projeto, `Create Service` -> **Compose**.
3. **Aponte o repositorio**: aba `General` -> `Provider` = GitHub/Git, escolha o repo e a
   branch publicada (`master`) e defina `Compose Path` = `docker-compose.dokploy.yml`.
4. **Variaveis de ambiente**: aba `Environment` -> cole o conteudo do seu `.env`
   (comece de `.env.example`). O compose usa `env_file: .env`, entao **o arquivo e
   obrigatorio** — sem ele o stack nao sobe. Minimo para o primeiro deploy:

   ```env
   IMAGE_PROVIDER=fake
   DATA_DIR=/data
   DB_PATH=/data/jobs.db
   IMAGES_DIR=/data/images
   APP_VERSION=0.1.0
   ```

   Para gerar de verdade, troque para `fal` + `FAL_KEY=...` ou `remote_gpu` +
   `REMOTE_GPU_URL=...`.
5. **Porta**: o servico `web` publica `8080:8000` no host. Em `Advanced` -> `Ports`,
   mapeie a porta do host que quiser (ex.: `8080`) para o container `8000`; o Dokploy
   mostra o mapeamento do compose na aba `General`.
6. **Deploy**: `Deploy`. Acompanhe o build (a imagem `python:3.12-slim` tem manifest
   arm64, entao builda nativo na VM). O healthcheck
   (`curl -f http://localhost:8000/api/v1/health`) precisa passar para o container ficar
   `healthy`.
7. **Dominio (depois)**: aba `Domains` -> `Add Domain`, aponte o DNS para o servidor e
   deixe o Traefik do Dokploy cuidar do TLS. Nao ha labels de Traefik no compose de
   proposito: o Dokploy injeta as dele quando um dominio e atribuido.
8. **ComfyUI local (opcional)**: so se `IMAGE_PROVIDER=local_comfy`. Build sua imagem
   arm64 (esboco comentado no `docker-compose.dokploy.yml`), publique-a num registry,
   defina `COMFY_IMAGE=<sua-imagem>` e suba com `--profile local`. Sem isso o servico
   `comfy` fica no placeholder: o `POST /api/v1/jobs` ainda responde `202` (a checagem de
   disponibilidade so olha se `COMFY_URL` esta preenchida, e o compose sempre a define) e
   o job falha depois, quando o provider nao alcanca `http://comfy:8188`.

Pelo CLI, o equivalente ao passo 1-3 e o padrao do Dokploy:

```bash
docker compose -f docker-compose.dokploy.yml up -d --build                    # so o webapp
docker compose -f docker-compose.dokploy.yml --profile local up -d --build    # + ComfyUI
```

Volume: `appdata` montado em `/data` (SQLite + imagens). O usuario do container e nao-root
(uid/gid 10001) e o diretorio e criado com o dono certo na imagem.

---

## API

| Metodo | Caminho | Resposta |
|---|---|---|
| `POST` | `/api/v1/jobs` | `202` `JobView` (body `JobCreate`) — `400` se o provider estiver indisponivel, `422` se o payload for invalido |
| `GET` | `/api/v1/jobs/{job_id}` | `200` `JobView` \| `404` |
| `GET` | `/api/v1/jobs/{job_id}/events` | `text/event-stream`; evento `status` com o `JobView` em JSON, 1x/s ate o status ser terminal, entao fecha |
| `GET` | `/api/v1/jobs?limit=20&offset=0` | `200 {items: [JobView], total: int}` (mais recentes primeiro) |
| `DELETE` | `/api/v1/jobs/{job_id}` | `204` — cancela o job (se `queued`/`running`) e apaga as imagens dele |
| `GET` | `/api/v1/images/{image_id}` | binario, `Content-Type` do formato, `Content-Disposition: attachment` |
| `GET` | `/api/v1/health` | `200 {ok, provider, model_id, queue_depth, version}` |
| `GET` | `/api/v1/config` | `200 {provider, model_id, providers_available, min_size, max_size, job_timeout_seconds}` |
| `GET` | `/` | `web/index.html` (StaticFiles `html=True`) |

Exemplo:

```bash
curl -s -X POST http://localhost:8080/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"gato astronauta em aquarela","width":1024,"height":1024,"num_images":1}' | jq

# acompanhar por SSE
curl -N http://localhost:8080/api/v1/jobs/<job_id>/events

# baixar a imagem
curl -OJ http://localhost:8080/api/v1/images/<image_id>
```

`JobCreate`: `prompt` (1..4000, obrigatorio), `negative_prompt`, `width`/`height`
(512..2048, default 1024), `num_images` (1..4), `seed`, `output_format`
(`png`|`jpeg`|`webp`), `prompt_expander` (`none`|`quality`), `reference_images`
(URLs http(s) ou data URI base64).

### Variaveis de ambiente

| Variavel | Default | Para que serve |
|---|---|---|
| `IMAGE_PROVIDER` | `fake` | `fake` \| `local_comfy` \| `fal` \| `remote_gpu` |
| `DATA_DIR` | `/data` | raiz dos dados (volume `appdata`) |
| `DB_PATH` | `/data/jobs.db` | SQLite (fonte de verdade do status) |
| `IMAGES_DIR` | `/data/images` | bytes das imagens |
| `FAL_KEY` / `FAL_MODEL` / `FAL_EDIT_MODEL` | vazio / `alibaba/qwen-image-2.1/text-to-image` / `.../edit` | provider `fal` |
| `COMFY_URL` | `http://comfy:8188` | provider `local_comfy` |
| `REMOTE_GPU_URL` / `REMOTE_GPU_KEY` / `REMOTE_GPU_MODEL` | vazio / vazio / `Qwen/Qwen-Image-2.1` | provider `remote_gpu` |
| `JOB_TIMEOUT_SECONDS` | `7200` | teto por job; **suba para 86400 se usar `local_comfy`** (o default de 2 h nao cobre nem 1024², e o pior caso de 2048² passa de 18 h) |
| `MAX_PARALLEL_JOBS` | `1` | jobs em paralelo (em ARM CPU, deixe 1) |
| `DEFAULT_WIDTH` / `DEFAULT_HEIGHT` | `1024` / `1024` | defaults dos campos |
| `MIN_SIZE` / `MAX_SIZE` | `512` / `2048` | validacao de `width`/`height` |
| `APP_VERSION` | `0.1.0` | aparece em `/health` e na tag da imagem |

---

## Aviso de licenca — Qwen Research License (uso comercial restrito)

Os pesos do Qwen-Image-2.1 nao sao Apache/MIT. O texto (identico no GitHub e no
HuggingFace, release de 2026-09-20) determina:

> §1.i: "'Non-Commercial' shall mean for research or evaluation purposes only."

> §2.a: "You are granted a non-exclusive, worldwide, non-transferable and royalty-free
> limited license under our intellectual property or other rights owned by us embodied in
> the Materials to use, reproduce, distribute, copy, create derivative works of, and make
> modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY."

> §2.b: "You shall not use the Materials for any commercial purpose without obtaining a
> separate commercial license from us. If you wish to use the Materials commercially, you
> shall request a license from us at model-business@notice.qwencloud.com."

> §3.c: "You shall retain in all copies of the Materials that you distribute the following
> attribution notices within a 'Notice' text file distributed as a part of such copies:
> 'Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou
> Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.'"

> §4.b: "If you use the Materials or any outputs or results therefrom to create, train,
> fine-tune, or improve an AI model that is distributed or made available, you shall
> prominently display 'Built with Qwen' or 'Improved using Qwen' in the related product
> documentation."

> §4.c: "You shall not use 'Qwen' as the primary name or identifier of any derivative works
> or products; reasonable descriptive use (e.g., 'fine-tuned from Qwen Image') is permitted."

> Aceite: "By clicking to agree or by using or distributing any portion or element of the
> Qwen Materials, you will be deemed to have recognized and accepted the content of this
> Agreement, which is effective immediately." — **baixar ja vincula**.

> §8: lei da China, foro exclusivo dos Tribunais Populares de Hangzhou.

Pontos praticos:

- Rodar o modelo pelo `fal` ou por uma GPU remota **nao muda** a licenca dos pesos que
  voce baixa com `scripts/download_models.sh` (a licenca e dos Materials).
- A licenca restringe pesos/codigo, nao as imagens geradas — **juridicamente incerto**
  para revenda. Nao ha clausula explicita de ownership dos outputs.
- Nao e proibicao absoluta: §2.b da caminho de licenca comercial paga.
- Se o uso for comercial, existem alternativas: modelos Qwen-Image com licenca permissiva
  em outros provedores (ex.: Replicate) e APIs comerciais (ex.: DashScope). Confirme a
  licenca e a versao exata do modelo no provedor escolhido antes de usar.

---

## Troubleshooting

**`POST /api/v1/jobs` responde 400 "provider indisponivel ..."**
Fail fast intencional. A mensagem diz qual variavel falta: `FAL_KEY` (provider `fal`),
`REMOTE_GPU_URL` (provider `remote_gpu`), `COMFY_URL` (provider `local_comfy`). Confira
tambem `GET /api/v1/config` -> `providers_available`. Nao ha "fallback silencioso" para o
`fake`: se voce pediu `fal` sem chave, o job nao roda.

**ComfyUI offline (`local_comfy`)**
Sintoma: job falha com erro de rede apontando `http://comfy:8188`. Cheque, nesta ordem:
(1) o servico `comfy` so sobe com `--profile local` e a imagem `COMFY_IMAGE` precisa ser a
sua build arm64 (o placeholder nao existe no registry);
(2) o ComfyUI precisa estar em `--listen 0.0.0.0` (senao o container nao responde de fora);
(3) `COMFY_URL` tem de ser o DNS do compose (`http://comfy:8188`), **nao** `localhost`;
(4) valide direto: `docker compose exec web curl -s http://comfy:8188/system_stats`.
O ComfyUI nao tem auth embutida — mantenha-o na rede interna.

**Disco cheio**
`/data` (volume `appdata`) guarda SQLite **e** as imagens. Nada e apagado
automaticamente: use `DELETE /api/v1/jobs/{id}` (ou o botao de apagar na UI) e verifique
com `df -h` e `du -sh /data/images`. Na VM, o `models/` com o set quantizado ocupa
11,2 GB e o cache do HuggingFace pode duplicar isso durante o download — depois de
baixar, `rm -rf models/.hf-cache models/.cache`.

**SSE bloqueado por proxy (a barra de progresso nao anda)**
O app ja manda `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no` e
`Connection: keep-alive`. Se ainda assim o stream chega em blocos: em Nginx
`proxy_buffering off;` e `proxy_read_timeout` >= o tempo do job; em Cloudflare, o
timeout de 100 s corta conexoes longas (use um subdominio sem proxy laranja ou aceite o
polling). A UI ja tem fallback: se o `EventSource` cair, ela passa a fazer polling de
1,5 s em `GET /api/v1/jobs/{id}`.

**Job preso em `queued`**
`GET /api/v1/health` -> `queue_depth` mostra quantos jobs ativos existem. Se ha um job
muito lento na frente, os outros esperam (FIFO, `MAX_PARALLEL_JOBS=1`). Job que passa de
`JOB_TIMEOUT_SECONDS` e marcado como `failed` com mensagem de timeout — no caminho local
o default de 7200 s (2 h) e curto para 1024²+; use `JOB_TIMEOUT_SECONDS=86400` (24 h, com
folga sobre o pior caso de ~18 h em 2048²).

**Imagens "somem" do job mas o job continua na lista**
E o comportamento do `DELETE`: ele apaga as imagens (arquivo + registro) e o job permanece
no historico com `images=[]`. O status depende do estado do job no momento do DELETE:
se estava `queued`/`running` ele e cancelado (`status=cancelled`); se ja era terminal
(`succeeded`/`failed`) o status e mantido como estava (o DELETE nao reescreve um resultado
ja concluido). Para `GET /api/v1/images/{id}` depois disso, a resposta e `404`.

**A UI fica sem estilo**
`web/index.html` usa o Tailwind via CDN: se o navegador nao tiver acesso a internet, o
layout perde o utilitario — o CSS proprio (`web/styles.css`) e o app em si continuam
funcionando. Sem build step, sem Node.

---

## Estrutura

```
app/
  main.py            rotas HTTP, SSE, lifespan        queue.py     fila asyncio + workers
  config.py          Settings (env)                   db.py        SQLite (aiosqlite)
  schemas.py         JobCreate / JobView              comfy_workflow.py  grafo do ComfyUI
  providers/         base.py (contrato) + fake.py, fal.py, remote_gpu.py, local_comfy.py
web/                 index.html + app.js + styles.css (SPA servida em /)
scripts/             download_models.sh (pesos) e smoke_test.sh (fumaca ponta-a-ponta)
tests/               test_api.py, test_providers.py, conftest.py
Dockerfile           python:3.12-slim arm64, usuario nao-root, HEALTHCHECK em /api/v1/health
docker-compose.dokploy.yml   web (8080:8000, volume appdata:/data) + comfy (profile local)
```
