"""Configuracao da aplicacao via pydantic-settings.

Todas as variaveis de ambiente do contrato sao lidas aqui, com os defaults
exatos. Uso:

    from app.config import get_settings
    settings = get_settings()

O objeto e imutavel (``model_config`` sem ``frozen`` para permitir
``model_copy(update=...)``, usado pela resolucao de providers).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuracao imutavel da aplicacao (env vars do contrato)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- provider ---------------------------------------------------------
    image_provider: str = "fake"

    # --- caminhos ---------------------------------------------------------
    data_dir: str = "/data"
    db_path: str = "/data/jobs.db"
    images_dir: str = "/data/images"

    # --- fal --------------------------------------------------------------
    fal_key: str = ""
    fal_model: str = "alibaba/qwen-image-2.1/text-to-image"
    fal_edit_model: str = "alibaba/qwen-image-2.1/edit"

    # --- comfyui local ----------------------------------------------------
    comfy_url: str = "http://comfy:8188"

    # --- gpu remota (OpenAI-compat) ---------------------------------------
    remote_gpu_url: str = ""
    remote_gpu_key: str = ""
    remote_gpu_model: str = "Qwen/Qwen-Image-2.1"

    # --- fila / limites ---------------------------------------------------
    job_timeout_seconds: int = 7200
    max_parallel_jobs: int = 1
    default_width: int = 1024
    default_height: int = 1024
    min_size: int = 512
    max_size: int = 2048

    # --- app --------------------------------------------------------------
    app_version: str = "0.1.0"

    # ------------------------------------------------------------------ #
    # validacao
    # ------------------------------------------------------------------ #
    @model_validator(mode="before")
    @classmethod
    def _rebase_paths_on_data_dir(cls, data: object) -> object:
        """Se DATA_DIR foi informado e DB_PATH/IMAGES_DIR nao, deriva os dois dela.

        Sem isso, subir com ``DATA_DIR=/tmp/x`` deixaria o banco em ``/data``
        (default do contrato) e a config pareceria ignorada. Roda em mode
        "before" porque no dict cru so aparecem os campos realmente informados
        (via env/kwargs) -- dentro de um validator "after" o
        ``model_fields_set`` ainda esta vazio.
        """
        if not isinstance(data, dict):
            return data
        data_dir = data.get("data_dir")
        if not data_dir:
            return data
        base = Path(str(data_dir)).expanduser()
        updates: dict[str, str] = {}
        if not data.get("db_path"):
            updates["db_path"] = str(base / "jobs.db")
        if not data.get("images_dir"):
            updates["images_dir"] = str(base / "images")
        if not updates:
            return data
        logger.debug(
            "Derivando caminhos a partir de DATA_DIR=%s: %s", data_dir, updates
        )
        return {**data, **updates}

    @model_validator(mode="after")
    def _check_ranges(self) -> Settings:
        if self.min_size > self.max_size:
            raise ValueError(
                f"MIN_SIZE ({self.min_size}) nao pode ser maior que MAX_SIZE ({self.max_size})."
            )
        if self.max_size < 1:
            raise ValueError(f"MAX_SIZE precisa ser >= 1 (recebido {self.max_size}).")
        if self.job_timeout_seconds < 1:
            raise ValueError(
                f"JOB_TIMEOUT_SECONDS precisa ser >= 1 (recebido {self.job_timeout_seconds})."
            )
        if self.max_parallel_jobs < 1:
            raise ValueError(
                f"MAX_PARALLEL_JOBS precisa ser >= 1 (recebido {self.max_parallel_jobs})."
            )
        if not self.data_dir:
            raise ValueError("DATA_DIR nao pode ser vazio.")
        return self

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def db_file(self) -> Path:
        """Caminho absoluto do arquivo SQLite."""
        return Path(self.db_path).expanduser()

    @property
    def images_path(self) -> Path:
        """Diretorio onde os bytes das imagens sao gravados."""
        return Path(self.images_dir).expanduser()

    @property
    def data_path(self) -> Path:
        """Diretorio raiz de dados."""
        return Path(self.data_dir).expanduser()

    def ensure_dirs(self) -> None:
        """Cria DATA_DIR, o diretorio do banco e IMAGES_DIR (idempotente)."""
        for directory in (self.data_path, self.db_file.parent, self.images_path):
            directory.mkdir(parents=True, exist_ok=True)

    def model_id_for(self, provider_name: str) -> str:
        """Model id "estatico" de um provider, sem instancia-lo.

        Fallback usado quando o provider configurado nao pode ser construido
        (chave/URL ausente ou nome invalido) mas /health e /config ainda
        precisam responder com uma string.
        """
        if provider_name == "fal":
            return self.fal_model
        if provider_name == "remote_gpu":
            return self.remote_gpu_model
        return provider_name or "desconhecido"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings singleton (cacheado). Sobrescreva limpando o cache nos testes."""
    return Settings()


def reset_settings_cache() -> None:
    """Limpa o cache do singleton (util em testes)."""
    get_settings.cache_clear()
