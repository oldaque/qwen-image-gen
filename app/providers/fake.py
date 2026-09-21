"""Provider `fake`: gera imagens localmente com Pillow. NUNCA acessa a rede.

Usado como default (IMAGE_PROVIDER=fake) e nos testes/smoke test, porque em um
servidor ARM64 sem GPU qualquer provider real e caro/lento.

Desenha: fundo em gradiente + prompt quebrado em linhas + job id + seed.
Simula progresso 0 -> 100 em ~3 s (divididos entre as N imagens pedidas).

As fontes sao carregadas de forma PORTATIL: a imagem de runtime (python:3.12-slim)
nao instala nenhum pacote de fontes, entao procuramos os caminhos usuais do
fontconfig e, se nenhum existir, usamos a fonte embutida do Pillow
(`ImageFont.load_default`). A geracao nunca falha por falta de fonte.
"""

from __future__ import annotations

import asyncio
import io
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.providers.base import GeneratedImage, ProgressFn, normalize_format
from app.schemas import JobCreate

__all__ = ["FakeProvider", "Provider", "is_available", "SIMULATED_SECONDS"]

#: Tempo total aproximado da simulacao de progresso (contrato: ~3 s).
SIMULATED_SECONDS = 3.0
_PROGRESS_TICKS = 10

#: Pares de cores (inicio, fim) do gradiente vertical, escolhidos pelo seed.
_GRADIENTS: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((24, 18, 61), (12, 148, 136)),
    ((46, 12, 60), (216, 92, 74)),
    ((9, 33, 71), (86, 176, 230)),
    ((33, 26, 12), (198, 154, 61)),
    ((12, 40, 30), (126, 208, 120)),
    ((52, 20, 44), (170, 96, 190)),
)

#: Caminhos procurados em ordem (Linux primero — e o alvo do container — e
#: caminhos do macOS apenas como conveniencia para rodar local sem setup).
#: DejaVuSans vem do pacote `fonts-dejavu-core`, Liberation de
#: `fonts-liberation`/`fonts-liberation2` e Noto de `fonts-noto-core`; em
#: imagens slim nenhum deles esta instalado, e o fallback do Pillow assume.
_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def is_available(settings: Settings) -> tuple[bool, str]:
    """`fake` esta sempre disponivel: e 100% offline (so depende do Pillow)."""
    return True, "sempre disponivel (offline, Pillow)"


def _load_font(size: int) -> ImageFont.ImageFont:
    """Devolve uma fonte utilizavel de tamanho `size`. NUNCA levanta excecao.

    Estrategia, em ordem: (1) primeiro caminho existente e legivel da lista de
    candidatos; (2) fonte embutida do Pillow (`load_default(size=...)`, que
    funciona sem nenhum arquivo do sistema). Qualquer falha de leitura apenas
    passa para o proximo candidato — a renderizacao nao depende de fontes do
    sistema, so do Pillow.
    """
    for path in _FONT_CANDIDATES:
        try:
            if not Path(path).is_file():
                continue
            return ImageFont.truetype(path, size=size)
        except (OSError, ValueError):  # pragma: no cover - ilegivel/corrompida
            continue
    try:  # Pillow >= 10.1 aceita size no default (FreeType embutido)
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001 - ultimo recurso: o bitmap embutido sempre existe
        return ImageFont.load_default()


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Quebra o texto em linhas que caibam em max_width (quebra palavras longas)."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            while draw.textlength(word, font=font) > max_width and len(word) > 1:
                cut = max(1, int(len(word) * max_width / max(1.0, draw.textlength(word, font=font))))
                piece, word = word[:cut], word[cut:]
                if current:
                    lines.append(current)
                    current = ""
                lines.append(piece)
            candidate = f"{current} {word}".strip()
            if not current or draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


class FakeProvider:
    """Provider offline: gradiente + prompt + job id + seed, via Pillow."""

    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = "fake/pillow-gradient"

    # ------------------------------------------------------------------
    # API do protocolo ImageProvider
    # ------------------------------------------------------------------
    async def generate(
        self,
        req: JobCreate,
        progress: ProgressFn,
        job_id: str | None = None,
    ) -> list[GeneratedImage]:
        """Gera `req.num_images` imagens. `job_id` e opcional (so para desenhar no canvas)."""
        total = max(1, int(req.num_images or 1))
        base_seed = req.seed if req.seed is not None else random.randrange(0, 2**31 - 1)
        label_job = job_id or f"fake-{base_seed:08x}"

        per_image_seconds = SIMULATED_SECONDS / total
        tick_seconds = per_image_seconds / _PROGRESS_TICKS

        images: list[GeneratedImage] = []
        for index in range(total):
            seed = base_seed + index
            for tick in range(_PROGRESS_TICKS):
                percent = int(round((index * _PROGRESS_TICKS + tick + 1) * 100 / (total * _PROGRESS_TICKS)))
                await progress(
                    percent,
                    f"gerando (fake) imagem {index + 1}/{total}: passo {tick + 1}/{_PROGRESS_TICKS}",
                )
                await asyncio.sleep(tick_seconds)

            data, width, height, fmt = await asyncio.to_thread(
                self._render,
                prompt=req.prompt,
                negative=req.negative_prompt or "",
                job_id=label_job,
                seed=seed,
                width=req.width,
                height=req.height,
                output_format=req.output_format,
                index=index,
                total=total,
            )
            images.append(
                GeneratedImage(data=data, width=width, height=height, seed=seed, format=fmt)
            )

        await progress(100, f"concluido (fake): {total} imagem(ns)")
        return images

    # ------------------------------------------------------------------
    # Renderizacao
    # ------------------------------------------------------------------
    def _render(
        self,
        *,
        prompt: str,
        negative: str,
        job_id: str,
        seed: int,
        width: int,
        height: int,
        output_format: str,
        index: int,
        total: int,
    ) -> tuple[bytes, int, int, str]:
        width = max(64, int(width))
        height = max(64, int(height))
        fmt = normalize_format(output_format, default="png")

        canvas = self._gradient(width, height, seed)

        title_font = _load_font(max(16, height // 30))
        body_font = _load_font(max(12, height // 52))
        small_font = _load_font(max(11, height // 72))

        # Painel escuro semitransparente para o texto ficar legivel no gradiente.
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        pad = max(14, width // 32)
        panel_box = (pad, pad, width - pad, height - pad)
        odraw.rounded_rectangle(panel_box, radius=max(8, width // 64), fill=(8, 10, 16, 150))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(canvas)
        max_text_width = int(width * 0.80)
        lines = _wrap_text(draw, prompt or "(sem prompt)", body_font, max_text_width)

        line_height = int(body_font.size * 1.35) if hasattr(body_font, "size") else body_font.size + 6
        header = "FAKE / imagem gerada offline (sem rede)"
        header_height = int(title_font.size * 1.6) if hasattr(title_font, "size") else title_font.size + 8
        footer_lines = [
            f"job: {job_id}",
            f"seed: {seed}",
            f"imagem {index + 1}/{total} - {width}x{height} - {fmt}",
        ]
        if negative:
            footer_lines.append(f"negative: {negative[:120]}")
        footer_height = len(footer_lines) * (
            int(small_font.size * 1.45) if hasattr(small_font, "size") else small_font.size + 6
        )

        available = height - 2 * pad - header_height - footer_height
        max_lines = max(1, available // max(1, line_height))
        truncated = len(lines) > max_lines
        shown = lines[:max_lines]
        if truncated:
            shown[-1] = shown[-1][: max(0, len(shown[-1]) - 1)] + "…"

        x = pad + max(8, width // 48)
        y = pad + max(8, height // 48)

        draw.text((x + 2, y + 2), header, font=title_font, fill=(0, 0, 0))
        draw.text((x, y), header, font=title_font, fill=(226, 232, 240))
        y += header_height

        for line in shown:
            draw.text((x + 1, y + 1), line, font=body_font, fill=(0, 0, 0))
            draw.text((x, y), line, font=body_font, fill=(248, 250, 252))
            y += line_height

        y = height - pad - footer_height + max(4, height // 96)
        for line in footer_lines:
            draw.text((x + 1, y + 1), line, font=small_font, fill=(0, 0, 0))
            draw.text((x, y), line, font=small_font, fill=(203, 213, 225))
            y += int(small_font.size * 1.45) if hasattr(small_font, "size") else small_font.size + 6

        buffer = io.BytesIO()
        save_kwargs: dict[str, object] = {}
        if fmt in ("jpeg", "webp"):
            save_kwargs["quality"] = 92
        if fmt == "jpeg":
            canvas = canvas.convert("RGB")
        canvas.save(buffer, format=fmt.upper() if fmt != "jpeg" else "JPEG", **save_kwargs)
        return buffer.getvalue(), width, height, fmt

    def _gradient(self, width: int, height: int, seed: int) -> Image.Image:
        rnd = random.Random(seed)
        top, bottom = _GRADIENTS[rnd.randrange(len(_GRADIENTS))]
        band_phase = rnd.random() * math.tau
        band_width = max(24, width // 12)
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        for y in range(height):
            t = y / max(1, height - 1)
            # banda diagonal senoidal, para o gradiente nao ficar chapado
            offset = int(0.5 * width * math.sin(band_phase + t * math.pi))
            base = (_lerp(top[0], bottom[0], t), _lerp(top[1], bottom[1], t), _lerp(top[2], bottom[2], t))
            draw.line([(0, y), (width, y)], fill=base)
            lighter = tuple(min(255, c + 26) for c in base)
            draw.line(
                [(offset, y), (min(width, offset + band_width), y)],
                fill=lighter,
            )
        return img


#: Alias de compatibilidade (nome curto).
Provider = FakeProvider
