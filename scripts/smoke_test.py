"""Smoke test: verifica conectividad con OVH AI Endpoints y ChromaDB.

Uso:
    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys

import chromadb
from rich.console import Console

from src.config import load_settings
from src.loader import load_all_posts
from src.ovh_client import OVHClient

console = Console()


def check_blog_repo(settings) -> None:
    console.rule("[bold]1. Blog repo[/bold]")
    if not settings.posts_dir.exists():
        console.print(f"[red]✗[/red] Posts dir no existe: {settings.posts_dir}")
        sys.exit(1)
    posts = load_all_posts(settings.posts_dir)
    console.print(f"[green]✓[/green] {len(posts)} posts cargados desde {settings.posts_dir}")
    if posts:
        sample = posts[0]
        console.print(f"   primer post: {sample.date} · {sample.title!r} · tags={list(sample.tags)}")


def check_chroma(settings) -> None:
    console.rule("[bold]2. ChromaDB[/bold]")
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    heartbeat = client.heartbeat()
    console.print(f"[green]✓[/green] Chroma heartbeat OK (ns={heartbeat})")
    collections = client.list_collections()
    console.print(f"   colecciones existentes: {[c.name for c in collections]}")


def check_ovh_embeddings(settings) -> None:
    console.rule("[bold]3. OVH embeddings[/bold]")
    ovh = OVHClient(settings)
    vectors = ovh.embed(["hola, esto es un test de embeddings"])
    dim = len(vectors[0])
    console.print(f"[green]✓[/green] Embedding generado, dim={dim} (modelo={settings.embedding_model})")


def check_ovh_chat(settings) -> None:
    console.rule("[bold]4. OVH chat completion[/bold]")
    ovh = OVHClient(settings)
    reply = ovh.chat(
        [
            {"role": "system", "content": "Responde en una sola palabra."},
            {"role": "user", "content": "Capital de Francia?"},
        ],
        max_tokens=20,
    )
    console.print(f"[green]✓[/green] Respuesta de {settings.chat_model}: {reply.strip()!r}")


def main() -> None:
    settings = load_settings()
    check_blog_repo(settings)
    check_chroma(settings)
    check_ovh_embeddings(settings)
    check_ovh_chat(settings)
    console.rule("[bold green]Todo OK[/bold green]")


if __name__ == "__main__":
    main()
