"""Servico de geracao de imagens (FastAPI + fila asyncio + SQLite).

Pacote principal da aplicacao. O entrypoint ASGI e ``app.main:app``.

Modulos:
    config    -- Settings (pydantic-settings) lido de variaveis de ambiente.
    schemas   -- modelos pydantic da API (JobCreate / JobView / ...).
    db        -- persistencia SQLite via aiosqlite (fonte de verdade do status).
    queue     -- worker asyncio FIFO que executa os jobs.
    main      -- aplicacao FastAPI com as rotas HTTP e o lifespan.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
