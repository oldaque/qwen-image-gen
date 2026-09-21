#!/usr/bin/env bash
#
# =============================================================================
# scripts/smoke_test.sh — teste de fumaca ponta-a-ponta do image-gen
# =============================================================================
# Sobe o app de verdade (uvicorn) com IMAGE_PROVIDER=fake numa porta livre, cria
# um job, acompanha ate o status terminal, baixa a imagem e valida o PNG.
# Nao toca no seu .env, no banco do volume nem em /data: tudo roda num diretorio
# temporario proprio.
#
# Fluxo:
#   1. confere comandos (python3, curl), cria venv e instala requirements.txt
#   2. escolhe uma porta livre e sobe:  uvicorn app.main:app --host 127.0.0.1
#      com IMAGE_PROVIDER=fake, DATA_DIR/DB_PATH/IMAGES_DIR no tmp
#   3. espera GET /api/v1/health responder {ok:true} (valida provider=model_id)
#   4. POST /api/v1/jobs (512x512, 1 imagem, png) -> exige HTTP 202 + job_id
#   5. polling em GET /api/v1/jobs/{id} ate succeeded|failed|cancelled
#   6. baixa GET /api/v1/images/{id} e valida: HTTP 200, Content-Type de imagem,
#      Content-Disposition attachment, assinatura PNG e Pillow abrindo 512x512
#   7. derruba o uvicorn e sai com 0 (PASS) ou 1 (FAIL)
#
# USO
#     scripts/smoke_test.sh [opcoes]
#       -p, --port N       porta fixa (default: uma porta livre sorteada)
#       -t, --timeout N    segundos de espera pelo job (default: 180)
#       -v, --venv DIR     diretorio do venv (default: <tmp>/venv)
#       -r, --reuse        reusa <repo>/.venv se existir (pula o pip install)
#       -k, --keep         nao apaga o diretorio temporario (logs/artefatos)
#           --skip-install nao instala requirements (venv precisa estar pronta)
#       -h, --help
#
# Requer: python3, curl. O servidor de teste sempre usa o provider fake (sem rede).
# =============================================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

PORT=""
JOB_TIMEOUT=180
VENV_DIR=""
REUSE_VENV=0
KEEP=0
SKIP_INSTALL=0

log()   { printf '[smoke] %s\n' "$*"; }
warn()  { printf '[smoke][AVISO] %s\n' "$*" >&2; }
fail()  { printf '\n[smoke][FALHOU] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
scripts/smoke_test.sh — teste de fumaca ponta-a-ponta (IMAGE_PROVIDER=fake)

  -p, --port N       porta fixa (default: porta livre sorteada)
  -t, --timeout N    segundos de espera pelo job (default: 180)
  -v, --venv DIR     diretorio do venv (default: <tmp>/venv)
  -r, --reuse        reusa <repo>/.venv se existir (pula o pip install)
  -k, --keep         nao apaga o diretorio temporario
      --skip-install nao instala requirements
  -h, --help
EOF
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--port)       [ $# -ge 2 ] || fail "--port exige um valor"; PORT="$2"; shift 2 ;;
    -t|--timeout)    [ $# -ge 2 ] || fail "--timeout exige um valor"; JOB_TIMEOUT="$2"; shift 2 ;;
    -v|--venv)       [ $# -ge 2 ] || fail "--venv exige um valor"; VENV_DIR="$2"; shift 2 ;;
    -r|--reuse)      REUSE_VENV=1; shift ;;
    -k|--keep)       KEEP=1; shift ;;
    --skip-install)  SKIP_INSTALL=1; shift ;;
    -h|--help)       usage ;;
    *)               fail "opcao desconhecida: $1 (use --help)" ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || fail "comando '$1' nao encontrado no PATH"; }
need python3
need curl

PY3=$(command -v python3)
PY3_VER=$("$PY3" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
log "python3 $PY3_VER ($PY3)"
"$PY3" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || fail "python3 $PY3_VER e antigo demais (o projeto requer 3.12+)"

[ -f "$ROOT_DIR/requirements.txt" ] || fail "requirements.txt nao existe em $ROOT_DIR"
[ -f "$ROOT_DIR/app/main.py" ]       || fail "app/main.py nao existe em $ROOT_DIR (o app foi escrito?)"

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/image-gen-smoke.XXXXXX")
SERVER_PID=""
LOG_FILE=""
CLEANUP_DONE=0

cleanup() {
  local status=$?
  [ "$CLEANUP_DONE" = "1" ] && return 0
  CLEANUP_DONE=1
  # Em caso de falha, o diretorio temporario e apagado: salva o log do servidor
  # aqui para nao perder a causa raiz.
  if [ "$status" != "0" ] && [ -n "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    warn "ultimas linhas do uvicorn (exit $status):"
    tail -n 20 "$LOG_FILE" >&2 || true
  fi
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    log "uvicorn (pid $SERVER_PID) derrubado"
  fi
  if [ "$KEEP" = "1" ]; then
    warn "--keep: artefatos em $WORK_DIR"
  else
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT INT TERM

DATA_DIR="$WORK_DIR/data"
DB_PATH="$DATA_DIR/jobs.db"
IMAGES_DIR="$DATA_DIR/images"
mkdir -p "$IMAGES_DIR"

# ---------------------------------------------------------------------------
# 1. venv + dependencias
# ---------------------------------------------------------------------------
if [ -z "$VENV_DIR" ]; then
  if [ "$REUSE_VENV" = "1" ] && [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    VENV_DIR="$ROOT_DIR/.venv"
  else
    VENV_DIR="$WORK_DIR/venv"
  fi
fi

# o servidor e iniciado a partir de $ROOT_DIR: o caminho do venv precisa ser
# absoluto antes disso.
case "$VENV_DIR" in /*) : ;; *) VENV_DIR="$PWD/$VENV_DIR" ;; esac

if [ -x "$VENV_DIR/bin/python" ]; then
  log "venv existente: $VENV_DIR"
else
  log "criando venv em $VENV_DIR"
  "$PY3" -m venv "$VENV_DIR" || fail "falha ao criar o venv (falta python3-venv?)"
fi
PY="$VENV_DIR/bin/python"

if [ "$SKIP_INSTALL" = "1" ]; then
  log "--skip-install: usando o venv como esta"
else
  log "instalando requirements.txt (pode demorar no primeiro run) ..."
  "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  "$PY" -m pip install --quiet -r "$ROOT_DIR/requirements.txt" \
    || fail "pip install -r requirements.txt falhou"
fi

"$PY" -c 'import fastapi, uvicorn, httpx, aiosqlite, PIL' \
  || fail "dependencias essenciais ausentes no venv (fastapi/uvicorn/httpx/aiosqlite/PIL)"

# ---------------------------------------------------------------------------
# 2. porta livre + subir o uvicorn
# ---------------------------------------------------------------------------
if [ -z "$PORT" ]; then
  PORT=$("$PY" -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
fi
case "$PORT" in ''|*[!0-9]*) fail "porta invalida: $PORT" ;; esac
BASE="http://127.0.0.1:$PORT"
LOG_FILE="$WORK_DIR/uvicorn.log"
log "subindo uvicorn com IMAGE_PROVIDER=fake em $BASE"
(
  cd "$ROOT_DIR"
  exec env \
    IMAGE_PROVIDER=fake \
    APP_VERSION=smoke \
    DATA_DIR="$DATA_DIR" \
    DB_PATH="$DB_PATH" \
    IMAGES_DIR="$IMAGES_DIR" \
    MAX_PARALLEL_JOBS=1 \
    "$VENV_DIR/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$PORT" --log-level info
) >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
log "uvicorn pid=$SERVER_PID (log: $LOG_FILE)"

json_get() { # json_get <arquivo> <expressao python sobre o dict d>
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as fh:
    d = json.load(fh)

safe = {"len": len, "str": str, "int": int, "list": list, "dict": dict, "sum": sum}
print(eval(sys.argv[2], {"__builtins__": {}, "d": d}, safe))
PYEOF
}

# ---------------------------------------------------------------------------
# 3. esperar /api/v1/health
# ---------------------------------------------------------------------------
HEALTH="$WORK_DIR/health.json"
READY=0
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    warn "uvicorn morreu durante o boot; log:"
    tail -n 40 "$LOG_FILE" >&2 || true
    fail "servidor nao subiu"
  fi
  if curl -fsS --max-time 2 "$BASE/api/v1/health" -o "$HEALTH" 2>/dev/null; then
    READY=1; break
  fi
  sleep 1
done
[ "$READY" = "1" ] || { tail -n 40 "$LOG_FILE" >&2 || true; fail "GET /api/v1/health nao respondeu ok em 60 s"; }

ok=$(json_get "$HEALTH" 'd["ok"]')
provider=$(json_get "$HEALTH" 'd["provider"]')
version=$(json_get "$HEALTH" 'd["version"]')
[ "$ok" = "True" ] || fail "health retornou ok=$ok"
[ "$provider" = "fake" ] || fail "health retornou provider=$provider (esperado fake)"
log "health ok: provider=$provider version=$version queue_depth=$(json_get "$HEALTH" 'd["queue_depth"]')"

CONFIG="$WORK_DIR/config.json"
curl -fsS --max-time 5 "$BASE/api/v1/config" -o "$CONFIG" || fail "GET /api/v1/config falhou"
[ "$(json_get "$CONFIG" 'd["provider"]')" = "fake" ] || fail "/api/v1/config nao anuncia o provider fake"
log "/api/v1/config ok: providers_available=$(json_get "$CONFIG" 'd["providers_available"]')"

# ---------------------------------------------------------------------------
# 4. POST /api/v1/jobs
# ---------------------------------------------------------------------------
JOB_BODY='{"prompt":"smoke test: gato azul em aquarela","negative_prompt":"borrado","width":512,"height":512,"num_images":1,"output_format":"png","prompt_expander":"none"}'
JOB_FILE="$WORK_DIR/job-create.json"
HTTP_CODE=$(curl -sS --max-time 30 -o "$JOB_FILE" -w '%{http_code}' \
  -X POST -H 'Content-Type: application/json' -d "$JOB_BODY" "$BASE/api/v1/jobs") \
  || fail "POST /api/v1/jobs falhou na conexao"
[ "$HTTP_CODE" = "202" ] || { cat "$JOB_FILE" >&2; fail "POST /api/v1/jobs devolveu $HTTP_CODE (esperado 202)"; }
JOB_ID=$(json_get "$JOB_FILE" 'd["job_id"]')
[ -n "$JOB_ID" ] || fail "resposta sem job_id"
log "job criado: $JOB_ID (HTTP $HTTP_CODE)"

# ---------------------------------------------------------------------------
# 5. polling ate status terminal
# ---------------------------------------------------------------------------
JOB="$WORK_DIR/job.json"
STATUS=""
STARTED=$SECONDS
DEADLINE=$((SECONDS + JOB_TIMEOUT))
while :; do
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    fail "timeout de ${JOB_TIMEOUT}s esperando o job (ultimo status: ${STATUS:-indefinido})"
  fi
  curl -fsS --max-time 5 "$BASE/api/v1/jobs/$JOB_ID" -o "$JOB" \
    || { sleep 1.5; continue; }
  STATUS=$(json_get "$JOB" 'd["status"]')
  case "$STATUS" in
    succeeded) break ;;
    failed|cancelled)
      err=$(json_get "$JOB" 'd["error"]')
      fail "job terminou como '$STATUS': $err"
      ;;
  esac
  sleep 1.5
done
ELAPSED=$((SECONDS - STARTED))
N_IMAGES=$(json_get "$JOB" 'len(d["images"])')
[ "$N_IMAGES" -ge 1 ] || fail "job succeeded mas images[] veio vazio"
log "job $JOB_ID -> succeeded em ${ELAPSED}s (${N_IMAGES} imagem(ns))"

# ---------------------------------------------------------------------------
# 6. baixar e validar a imagem
# ---------------------------------------------------------------------------
IMG_URL=$(json_get "$JOB" 'd["images"][0]["url"]')
IMG_ID=$(json_get "$JOB" 'd["images"][0]["image_id"]')
[ -n "$IMG_URL" ] || fail "job succeeded mas sem imagens[]"
IMG_FILE="$WORK_DIR/image.png"
HEADERS="$WORK_DIR/image.headers"
HTTP_CODE=$(curl -sS --max-time 30 -D "$HEADERS" -o "$IMG_FILE" -w '%{http_code}' "$BASE$IMG_URL") \
  || fail "download de $IMG_URL falhou na conexao"
[ "$HTTP_CODE" = "200" ] || fail "GET $IMG_URL devolveu $HTTP_CODE (esperado 200)"
HEADERS_LC="$WORK_DIR/image.headers.lc"
tr 'A-Z' 'a-z' <"$HEADERS" >"$HEADERS_LC"
grep -q -E '^content-disposition: *attachment' "$HEADERS_LC" \
  || fail "GET /api/v1/images/{id} sem 'Content-Disposition: attachment'"
CTYPE=$(awk -F': *' '/^content-type:/{gsub(/\r/,"",$2); print $2; exit}' "$HEADERS_LC")
case "$CTYPE" in
  image/png*) : ;;
  *) fail "Content-Type inesperado para PNG: '$CTYPE'" ;;
esac
log "imagem $IMG_ID baixada: $IMG_URL (HTTP 200, $CTYPE, $(wc -c <"$IMG_FILE" | tr -d ' ') bytes)"

"$PY" - "$IMG_FILE" <<'PYEOF' || fail "validacao do PNG falhou (Pillow)"
import sys
from PIL import Image

path = sys.argv[1]
with open(path, "rb") as fh:
    head = fh.read(8)
assert head == b"\x89PNG\r\n\x1a\n", f"assinatura PNG invalida: {head!r}"

with Image.open(path) as im:
    im.verify()                      # checa integridade do arquivo
with Image.open(path) as im:         # verify() invalida a instancia: reabrir
    w, h, fmt, mode = im.width, im.height, im.format, im.mode

assert fmt == "PNG", f"formato {fmt} != PNG"
assert (w, h) == (512, 512), f"tamanho {w}x{h} != 512x512"
print(f"[smoke] PNG valido: {w}x{h} {fmt} {mode}")
PYEOF

# ---------------------------------------------------------------------------
# 7. fim
# ---------------------------------------------------------------------------
IMG_BYTES=$(wc -c <"$IMG_FILE" | tr -d ' ')
cleanup || true
trap - EXIT INT TERM
printf '\n[smoke] PASS — job %s, %s imagem(ns) em %ss, imagem %s (%s bytes) validada\n' \
  "$JOB_ID" "$N_IMAGES" "$ELAPSED" "$IMG_ID" "$IMG_BYTES"
exit 0
