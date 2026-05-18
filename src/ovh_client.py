from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI

from .config import Settings


class OVHClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.ovh_api_key,
            base_url=settings.ovh_base_url,
        )

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        response = self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=batch,
        )
        return [item.embedding for item in response.data]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
