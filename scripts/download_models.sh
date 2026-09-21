#!/usr/bin/env bash
#
# =============================================================================
# scripts/download_models.sh — pesos do Qwen-Image-2.1 para o ComfyUI local
# =============================================================================
# Baixa o "set minimo" (11.177 GB no total) para uma VM ARM64 sem GPU.
#
# NAO ha placeholder neste script. Repo, CAMINHO DENTRO DO REPO e tamanho exato
# (bytes) de cada arquivo estao na tabela abaixo — os nomes e valores sao os do
# download real:
#
#   destino              repo                         caminho no repo                                bytes
#   -------------------  ---------------------------  ---------------------------------------------  -------------
#   diffusion_models/    Abiray/Qwen-Image-2.1-GGUF   qwen_image_2.1_Q4_K_M.gguf                      4189343904
#   text_encoders/       Comfy-Org/Qwen-Image-2.1     text_encoders/qwen3vl_8b_w4a8.safetensors       6312105364
#   vae/                 Comfy-Org/Qwen-Image-2.1     vae/qwen_image_2.1_vae_bf16.safetensors          675509688
#   ------------------------------------------------------------------ total: 11176958956 (11.177 GB)
#
# ATENCAO AOS NOMES (importante): listagens e notas antigas costumam citar os
# nomes sem o prefixo de pasta ("qwen3vl_8b_w4a8.safetensors",
# "qwen_image_2.1_vae_bf16.safetensors") e o GGUF com hifens
# ("qwen-image-2.1-Q4_K_M.gguf"). No repo real o caminho e "text_encoders/..." e
# "vae/..." e o GGUF usa UNDERSCORES ("qwen_image_2.1_Q4_K_M.gguf"). Montar a URL
# com o nome sem prefixo da HTTP 404. Este script usa os caminhos reais — e
# app/comfy_workflow.py (DIFFUSION_MODEL_GGUF) aponta para o mesmo nome de arquivo.
#
# AVISOS TECNICOS (leia antes de rodar):
#  * O DiT em GGUF so existe em repos de TERCEIROS: nem o oficial
#    (Qwen/Qwen-Image-2.1) nem o Comfy-Org publicam GGUF (so bf16/int8_convrot/
#    w4a8). Escolhido: Abiray Q4_K_M (4.189 GB). Alternativa direta:
#    leejet/Qwen-Image-2.1-GGUF -> qwen_image_2.1-Q4_K.gguf (4197494816 bytes,
#    hifens + underlines misturados; repo voltado ao stable-diffusion.cpp).
#  * .gguf NAO carrega no UNETLoader padrao: exige o custom node ComfyUI-GGUF
#    (fork do leejet — o README dele diz que o do city96 esta pouco mantido).
#  * O template oficial aponta para qwen_image_2.1_int8_convrot.safetensors
#    (7256783064 bytes). No CPU arm64 esse e o PIOR caso: int8 cai no kernel
#    torch._int_mm (~4.2 GFLOPS medidos nesta classe de hardware) e bf16/fp16
#    sao emulados em software (~250-550x mais lento que fp32). Por isso o set
#    GGUF Q4_K_M.
#  * Alternativas oficiais no MESMO repo Comfy-Org (nao baixadas aqui):
#    text_encoders/qwen3vl_8b_int8_convrot.safetensors (9350798360) e
#    text_encoders/qwen3vl_8b_bf16.safetensors (17534334616 — nao cabe na RAM
#    junto do resto). PE-T2I/I2I opcionais: 9471072252 bytes cada (pular).
#  * RAM com este set: ~10-12 GB de pesos; pico estimado 13-16 GB @1024² e
#    16-21 GB @2048² -> cabe na RAM disponivel nesse tipo de VM (~23 GB). O
#    gargalo real e o compute dos poucos vCPUs disponiveis, nao a RAM.
#  * Sem token/gate: gated=false nos repos e o download anonimo funciona
#    (o curl retoma por Range). Licenca: Qwen Research License
#    Agreement (NAO comercial sem licenca paga).
#
# ESPACO EM DISCO: o set ocupa ~11.2 GB. O huggingface_hub >= 1.0 escreve direto
# no destino (sem duplicar blob), mas versoes antigas duplicavam no cache. Este
# script aponta HF_HOME para "$MODELS_DIR/.hf-cache" (mesmo volume, visivel) e o
# proprio CLI cria "$MODELS_DIR/.cache"; os dois podem ser apagados depois:
#     rm -rf models/.hf-cache models/.cache      # libera o cache residual
#
# USO
#     scripts/download_models.sh [opcoes]
#       -d, --dir DIR        destino dos pesos (default: ./models ou $MODELS_DIR)
#       -o, --only LISTA     baixa so "dit", "te", "vae" (separados por virgula)
#           --volume NAME    apos baixar, copia tudo para o volume docker NAME
#                            (o do compose: "<nome-do-projeto>_models")
#           --install-accel  cria .venv-hf e instala huggingface_hub + hf_transfer
#                            (o acelerador usado depende da versao: Xet no hub
#                            >= 1.0, hf_transfer no hub antigo)
#           --no-verify      nao valida o tamanho dos arquivos
#           --force          segue mesmo com espaco em disco apertado
#       -h, --help
#
# RETOMAVEL: tanto o CLI "hf" (hf_transfer) quanto o curl (-C -) retomam de onde
# pararam; se cair a conexao, rode o script de novo.
# =============================================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"

# ---- set minimo: bytes exatos dos arquivos publicados -----------------------
DIT_REPO="Abiray/Qwen-Image-2.1-GGUF"
DIT_PATH="qwen_image_2.1_Q4_K_M.gguf"
DIT_NAME="qwen_image_2.1_Q4_K_M.gguf"
DIT_DIR="diffusion_models"
DIT_BYTES=4189343904

TE_REPO="Comfy-Org/Qwen-Image-2.1"
TE_PATH="text_encoders/qwen3vl_8b_w4a8.safetensors"
TE_NAME="qwen3vl_8b_w4a8.safetensors"
TE_DIR="text_encoders"
TE_BYTES=6312105364

VAE_REPO="Comfy-Org/Qwen-Image-2.1"
VAE_PATH="vae/qwen_image_2.1_vae_bf16.safetensors"
VAE_NAME="qwen_image_2.1_vae_bf16.safetensors"
VAE_DIR="vae"
VAE_BYTES=675509688

ONLY=""
VOLUME=""
INSTALL_ACCEL=0
VERIFY=1
FORCE=0
HF_CLI=""
HF_TRANSFER="0"
TRANSFER_ENV=""
ACTUAL_TOTAL=0
FAILED=0

log()  { printf '[download_models] %s\n' "$*"; }
warn() { printf '[download_models][AVISO] %s\n' "$*" >&2; }
die()  { printf '[download_models][ERRO] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
scripts/download_models.sh — baixa o set Qwen-Image-2.1 (11.177 GB) para o ComfyUI local

  -d, --dir DIR        destino dos pesos (default: ./models ou $MODELS_DIR)
  -o, --only LISTA     baixa so "dit", "te", "vae" (separados por virgula)
      --volume NAME    apos baixar, copia tudo para o volume docker NAME
      --install-accel  cria .venv-hf e instala huggingface_hub + hf_transfer
      --no-verify      nao valida o tamanho dos arquivos
      --force          segue mesmo com espaco em disco apertado
  -h, --help

Detalhes, avisos tecnicos e a tabela de arquivos estao no cabecalho do script.
EOF
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dir)        [ $# -ge 2 ] || die "--dir exige um valor"; MODELS_DIR="$2"; shift 2 ;;
    -o|--only)       [ $# -ge 2 ] || die "--only exige um valor"; ONLY="$2"; shift 2 ;;
    --volume)        [ $# -ge 2 ] || die "--volume exige um valor"; VOLUME="$2"; shift 2 ;;
    --install-accel) INSTALL_ACCEL=1; shift ;;
    --no-verify)     VERIFY=0; shift ;;
    --force)         FORCE=1; shift ;;
    -h|--help)       usage ;;
    *)               die "opcao desconhecida: $1 (use --help)" ;;
  esac
done

mkdir -p "$MODELS_DIR" || die "nao consegui criar o diretorio $MODELS_DIR"
MODELS_DIR=$(cd "$MODELS_DIR" && pwd) || die "nao consegui resolver --dir $MODELS_DIR"
ONLY=$(printf '%s' "$ONLY" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "comando '$1' nao encontrado no PATH"; }

# stat portatil (GNU/Linux e BSD/macOS)
file_size() {
  if stat -c%s "$1" >/dev/null 2>&1; then stat -c%s "$1"; else stat -f%z "$1"; fi
}
bytes_h() {
  awk -v b="$1" 'BEGIN{ split("B KiB MiB GiB TiB",u," "); i=1; while (b>=1024 && i<5){ b/=1024; i++ } printf "%.2f %s", b, u[i] }'
}
gb3() { awk -v b="$1" 'BEGIN{ printf "%.3f GB", b/1e9 }'; }

want() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

# ---------------------------------------------------------------------------
# backend de download: hf CLI (rapido, com hf_transfer) ou curl (sempre existe)
# ---------------------------------------------------------------------------
resolve_backend() {
  local candidates=() c py hubver
  if [ "$INSTALL_ACCEL" = "1" ]; then
    require_cmd python3
    if [ ! -x "$ROOT_DIR/.venv-hf/bin/python" ]; then
      log "criando $ROOT_DIR/.venv-hf"
      python3 -m venv "$ROOT_DIR/.venv-hf" || die "falha ao criar o venv"
    fi
    log "instalando huggingface_hub + hf_transfer no .venv-hf ..."
    "$ROOT_DIR/.venv-hf/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
    # hub >= 1.0 nao tem mais o extra "cli" (o comando "hf" ja vem no pacote);
    # versoes antigas precisam do extra. Tentamos os dois caminhos.
    "$ROOT_DIR/.venv-hf/bin/python" -m pip install --quiet huggingface_hub hf_transfer \
      || "$ROOT_DIR/.venv-hf/bin/python" -m pip install --quiet "huggingface_hub[cli]" hf_transfer \
      || warn "nao consegui instalar huggingface_hub/hf_transfer; sigo com o que existir"
    candidates+=("$ROOT_DIR/.venv-hf/bin/hf" "$ROOT_DIR/.venv-hf/bin/huggingface-cli")
  fi
  candidates+=("$(command -v hf 2>/dev/null || true)" "$(command -v huggingface-cli 2>/dev/null || true)")

  for c in "${candidates[@]}"; do
    if [ -n "$c" ] && [ -x "$c" ]; then HF_CLI="$c"; break; fi
  done

  if [ -n "$HF_CLI" ]; then
    py="$(dirname "$HF_CLI")/python"
    [ -x "$py" ] || py="$(command -v python3 2>/dev/null || true)"
    hubver=""
    [ -n "$py" ] && hubver=$("$py" -c 'import huggingface_hub as h; print(h.__version__)' 2>/dev/null || true)
    if [ -n "$hubver" ] && [ -n "$py" ] && \
       "$py" -c 'import huggingface_hub as h; p=[int(x) for x in h.__version__.split(".")[:2]]; raise SystemExit(0 if p >= [1, 0] else 1)' 2>/dev/null; then
      # hub >= 1.0: o transporte acelerado e o Xet; HF_HUB_ENABLE_HF_TRANSFER
      # virou deprecated e so emite FutureWarning.
      TRANSFER_ENV="HF_XET_HIGH_PERFORMANCE"
      HF_TRANSFER=1
      log "backend: $HF_CLI (huggingface_hub $hubver -> Xet high performance)"
    elif [ -n "$py" ] && "$py" -c 'import hf_transfer' >/dev/null 2>&1; then
      TRANSFER_ENV="HF_HUB_ENABLE_HF_TRANSFER"
      HF_TRANSFER=1
      log "backend: $HF_CLI (huggingface_hub ${hubver:-?} -> hf_transfer ATIVO)"
    else
      log "backend: $HF_CLI (sem acelerador instalado; use --install-accel)"
    fi
  else
    require_cmd curl
    HF_CLI=""
    log "backend: curl (retomavel com -C -; mais rapido com --install-accel)"
  fi
}

# fetch_file <repo> <caminho-no-repo> <dir-destino> <nome-no-destino> <bytes>
fetch_file() {
  local repo="$1" rpath="$2" dest_dir="$3" name="$4" expected="$5"
  local target="$dest_dir/$name" downloaded url
  mkdir -p "$dest_dir"
  if [ -s "$target" ]; then
    log "ja existe: $target ($(bytes_h "$(file_size "$target")")) — retomando/verificando"
  fi
  if [ -n "$HF_CLI" ]; then
    # --local-dir na raiz reproduz a arvore do repo; o CLI pula o que ja esta
    # completo e usa o cache para retomar.
    "$HF_CLI" download "$repo" "$rpath" --local-dir "$MODELS_DIR" \
      || die "falha ao baixar $repo/$rpath com $HF_CLI"
    downloaded="$MODELS_DIR/$rpath"
    if [ -e "$downloaded" ] && [ "$downloaded" != "$target" ]; then
      log "movendo $(basename "$downloaded") -> $dest_dir/"
      mv -f "$downloaded" "$target"
    fi
  else
    url="https://huggingface.co/$repo/resolve/main/$rpath"
    if ! curl -fL -C - --retry 5 --retry-delay 5 --connect-timeout 30 \
              --speed-limit 1024 --speed-time 120 \
              -o "$target" "$url"; then
      # curl -C - num arquivo ja completo pode levar 416; se o tamanho esta
      # certo, o arquivo esta bom e seguimos.
      if [ -s "$target" ] && [ "$(file_size "$target")" -ge "$expected" ]; then
        warn "curl saiu com erro em $url, mas o arquivo local ja tem o tamanho esperado; seguindo"
      else
        die "falha ao baixar $url (rode de novo: o curl retoma com -C -)"
      fi
    fi
  fi
  [ -s "$target" ] || die "download nao gerou arquivo: $target"
}

# verify_size <chave> <arquivo> <bytes-esperados>
verify_size() {
  local key="$1" path="$2" expected="$3" size
  size=$(file_size "$path")
  if [ "$VERIFY" = "0" ]; then
    log "tamanho de $key: $(bytes_h "$size") (--no-verify)"
    return 0
  fi
  if [ "$size" -lt "$expected" ]; then
    warn "$key INCOMPLETO: $(bytes_h "$size") < $(bytes_h "$expected") esperados — rode o script de novo (e retomavel)"
    return 1
  fi
  if [ "$size" -gt "$expected" ]; then
    warn "$key: $(bytes_h "$size") > $(bytes_h "$expected") esperados (arquivo maior que o publicado?)"
  fi
  return 0
}

expected_total() {
  local total=0
  want dit && total=$((total + DIT_BYTES))
  want te  && total=$((total + TE_BYTES))
  want vae && total=$((total + VAE_BYTES))
  printf '%s' "$total"
}

check_space() {
  local need avail
  # hf >= 1.0 escreve direto no destino (sem duplicar blob no cache); hf antigo
  # materializava o cache + o destino. curl nao duplica. Estimativas folgadas.
  need=$(awk -v g="$(expected_total)" -v hf="$HF_CLI" 'BEGIN{ printf "%d", (hf=="") ? g*1.15 : g*1.6 }')
  avail=$(df -Pk "$1" 2>/dev/null | awk 'NR==2{ printf "%d", $4*1024 }')
  if [ -z "$avail" ] || [ "$avail" = "0" ]; then
    warn "nao consegui medir o espaco livre de $1; seguindo"
    return 0
  fi
  log "espaco livre em $1: $(bytes_h "$avail") | necessario (com cache HF): $(bytes_h "$need")"
  if [ "$avail" -lt "$need" ]; then
    if [ "$FORCE" = "1" ]; then
      warn "espaco apertado, seguindo por causa de --force"
    else
      die "espaco insuficiente em $1 ($(bytes_h "$avail") < $(bytes_h "$need")). Limpe o disco, use -d em outro volume, ou --force para ignorar."
    fi
  fi
}

# process_entry <chave> <dir> <repo> <caminho> <nome> <bytes>
process_entry() {
  local key="$1" dir="$2" repo="$3" rpath="$4" name="$5" expected="$6"
  local target="$MODELS_DIR/$dir/$name" size
  want "$key" || { log "pulando $key (--only)"; return 0; }
  if [ -s "$target" ] && [ "$(file_size "$target")" -eq "$expected" ]; then
    log "== $key ja completo: $dir/$name ($(bytes_h "$expected")) — nada a fazer"
    ACTUAL_TOTAL=$((ACTUAL_TOTAL + expected))
    return 0
  fi
  log "== $key -> $dir/$name ($(gb3 "$expected"), repo $repo)"
  fetch_file "$repo" "$rpath" "$MODELS_DIR/$dir" "$name" "$expected"
  size=$(file_size "$target")
  verify_size "$key" "$target" "$expected" || FAILED=$((FAILED + 1))
  ACTUAL_TOTAL=$((ACTUAL_TOTAL + size))
}

copy_to_volume() {
  [ -n "$VOLUME" ] || return 0
  if ! command -v docker >/dev/null 2>&1; then
    warn "--volume pedido mas 'docker' nao esta no PATH; copie na mao:"
    warn "  docker run --rm -v $VOLUME:/dst -v $MODELS_DIR:/src:ro alpine:3 sh -c 'mkdir -p /dst && cp -a /src/diffusion_models /src/text_encoders /src/vae /dst/'"
    return 0
  fi
  log "copiando os pesos para o volume docker '$VOLUME' ..."
  docker run --rm \
    -v "$VOLUME":/dst \
    -v "$MODELS_DIR":/src:ro \
    alpine:3 sh -c 'mkdir -p /dst && cp -a /src/diffusion_models /src/text_encoders /src/vae /dst/ && du -sh /dst/*' \
    || warn "falha ao copiar para o volume $VOLUME"
}

# ---------------------------------------------------------------------------
resolve_backend
export HF_HOME="${HF_HOME:-$MODELS_DIR/.hf-cache}"
# Acelerador conforme a versao do hub (Xet no >= 1.0, hf_transfer no antigo).
if [ -n "$TRANSFER_ENV" ]; then
  export "${TRANSFER_ENV}=1"
fi

log "destino: $MODELS_DIR"
log "HF_HOME (cache): $HF_HOME"
check_space "$MODELS_DIR"

process_entry dit "$DIT_DIR" "$DIT_REPO" "$DIT_PATH" "$DIT_NAME" "$DIT_BYTES"
process_entry te  "$TE_DIR"  "$TE_REPO"  "$TE_PATH"  "$TE_NAME"  "$TE_BYTES"
process_entry vae "$VAE_DIR" "$VAE_REPO" "$VAE_PATH" "$VAE_NAME" "$VAE_BYTES"

copy_to_volume

printf '\n'
log "===== RESUMO ====="
for rel in "$DIT_DIR/$DIT_NAME" "$TE_DIR/$TE_NAME" "$VAE_DIR/$VAE_NAME"; do
  if [ -f "$MODELS_DIR/$rel" ]; then
    printf '  %-52s %s\n' "$rel" "$(bytes_h "$(file_size "$MODELS_DIR/$rel")")"
  else
    printf '  %-52s AUSENTE\n' "$rel"
  fi
done
log "total baixado: $(bytes_h "$ACTUAL_TOTAL") (esperado: $(gb3 "$(expected_total)"))"

if [ "$FAILED" -gt 0 ]; then
  warn "$FAILED arquivo(s) com tamanho abaixo do esperado — rode o script de novo (e retomavel)"
  exit 1
fi

log "OK"
log "proximos passos:"
log "  1) libere os caches residuais: rm -rf '$HF_HOME' '$MODELS_DIR/.cache'"
log "  2) alimente o volume do compose (se nao usou --volume):"
log "     scripts/download_models.sh --volume <nome-do-projeto>_models"
log "     ou, no docker-compose.dokploy.yml, troque 'models:/models' por './models:/models'"
log "  3) no workflow do ComfyUI troque os loaders para qwen_image_2.1_Q4_K_M.gguf /"
log "     qwen3vl_8b_w4a8.safetensors / qwen_image_2.1_vae_bf16.safetensors"
log "     (o .gguf exige o custom node ComfyUI-GGUF do leejet)"
