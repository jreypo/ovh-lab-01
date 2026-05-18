from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import frontmatter

SLUG_FROM_FILENAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.md$")


@dataclass(frozen=True)
class Post:
    slug: str
    title: str
    date: date | None
    tags: tuple[str, ...]
    body: str
    source_path: Path
    extra: dict = field(default_factory=dict)

    @property
    def url_path(self) -> str:
        if self.date is None:
            return f"/{self.slug}/"
        return f"/{self.date:%Y/%m/%d}/{self.slug}/"


def _coerce_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _slug_from_filename(path: Path) -> tuple[str, date | None]:
    match = SLUG_FROM_FILENAME.match(path.name)
    if not match:
        return path.stem, None
    return match.group("slug"), _coerce_date(match.group("date"))


def load_post(path: Path) -> Post:
    parsed = frontmatter.load(path)
    fm = parsed.metadata or {}

    file_slug, file_date = _slug_from_filename(path)
    title = str(fm.get("title") or file_slug.replace("-", " ").title())
    post_date = _coerce_date(fm.get("date")) or file_date

    raw_tags = fm.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = tuple(str(t).strip() for t in raw_tags if str(t).strip())

    body = (parsed.content or "").strip()

    return Post(
        slug=file_slug,
        title=title,
        date=post_date,
        tags=tags,
        body=body,
        source_path=path,
        extra={k: v for k, v in fm.items() if k not in {"title", "date", "tags"}},
    )


def iter_posts(posts_dir: Path) -> Iterator[Post]:
    for path in sorted(posts_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        post = load_post(path)
        if not post.body:
            continue
        yield post


def load_all_posts(posts_dir: Path) -> list[Post]:
    return list(iter_posts(posts_dir))
