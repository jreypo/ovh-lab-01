from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ovh_api_key: str
    ovh_base_url: str
    embedding_model: str
    chat_model: str
    chroma_host: str
    chroma_port: int
    blog_repo_path: Path
    blog_posts_subdir: str

    @property
    def posts_dir(self) -> Path:
        return self.blog_repo_path / self.blog_posts_subdir


def load_settings() -> Settings:
    def _required(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"Missing required env var: {name}")
        return value

    return Settings(
        ovh_api_key=_required("OVH_AI_API_KEY"),
        ovh_base_url=_required("OVH_AI_BASE_URL"),
        embedding_model=os.environ.get("OVH_EMBEDDING_MODEL", "bge-multilingual-gemma2"),
        chat_model=os.environ.get("OVH_CHAT_MODEL", "gpt-oss-120b"),
        chroma_host=os.environ.get("CHROMA_HOST", "localhost"),
        chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
        blog_repo_path=Path(_required("BLOG_REPO_PATH")).expanduser(),
        blog_posts_subdir=os.environ.get("BLOG_POSTS_SUBDIR", "content/posts"),
    )
