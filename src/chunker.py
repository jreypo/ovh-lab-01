from __future__ import annotations

from dataclasses import dataclass

from .loader import Post


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    post_slug: str
    post_title: str
    post_date: str | None
    source_url: str
    chunk_index: int
    strategy: str
    section_heading: str | None = None

    def metadata(self) -> dict[str, str | int | None]:
        return {
            "post_slug": self.post_slug,
            "post_title": self.post_title,
            "post_date": self.post_date,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "section_heading": self.section_heading,
        }


def chunk_post_naive(post: Post, *, chunk_chars: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Fase 1: chunking naïve por caracteres con overlap fijo.

    Nota: usamos caracteres como proxy de tokens para evitar depender del tokenizer
    del modelo. Aprox. 4 chars ≈ 1 token en inglés/español, así que 1800 chars ≈ 450 tokens.
    """
    raise NotImplementedError("Implementar en notebook 01")


def chunk_post_semantic_md(post: Post, *, max_chars: int = 2400) -> list[Chunk]:
    """Fase 2: chunking semántico respetando jerarquía Markdown (#, ##, ###)."""
    raise NotImplementedError("Implementar en notebook 02")
