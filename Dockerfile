# =============================================================================
# image-gen — imagem de runtime do webapp (FastAPI + uvicorn + SQLite)
# Alvo: linux/arm64 (VM ARM64 sem GPU, Ubuntu 24.04).
#
# Decisoes de projeto:
#  * python:3.12-slim publica manifest arm64 -> base correta e pequena.
#  * NADA de torch/ComfyUI aqui: a geracao local vive no servico "comfy"
#    (profile "local" do docker-compose.dokploy.yml). Este container so roda a
#    API + a fila + a UI estatica.
#  * Todas as deps do requirements.txt tem wheel manylinux aarch64 publicado
#    (Pillow inclusive) -> o build nao compila nada exotico. --no-cache-dir
#    evita inchar a camada de pip.
#  * curl entra apenas para o HEALTHCHECK (exigido pelo contrato).
#  * Usuario nao-root (uid/gid 10001). /data e criado E cedido ao usuario app
#    na imagem: quando o volume nomeado "appdata" esta vazio, o Docker copia
#    este diretorio com dono/modo, e o app escreve sem precisar de root.
# =============================================================================
FROM python:3.12-slim

ARG APP_VERSION=0.1.0
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    DB_PATH=/data/jobs.db \
    IMAGES_DIR=/data/images \
    APP_VERSION=${APP_VERSION}

# curl = HEALTHCHECK; ca-certificates = TLS das chamadas httpx (fal/remote_gpu).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" app \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
      --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

# Camada de dependencias separada do codigo: mexer em app/ ou web/ nao reinstala.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web

RUN mkdir -p /data/images \
 && chown -R app:app /data /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
