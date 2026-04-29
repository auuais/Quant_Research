from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from algoding.settings import Settings


@dataclass(frozen=True)
class NewsArticle:
    article_id: int
    created_at: str
    headline: str
    summary: str
    symbols: tuple[str, ...]
    source: str
    url: str


class AlpacaNewsHistoricalClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for historical news.")
        self._client = NewsClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            raw_data=False,
        )
        self._cache_dir = settings.llm_news_cache_dir / "alpaca_news"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_news_articles(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        include_content: bool = False,
        limit: int = 50,
    ) -> list[NewsArticle]:
        cache_path = self._cache_path(symbol=symbol, start=start, end=end, include_content=include_content)
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return [NewsArticle(**item) for item in raw]

        articles_by_id: dict[int, NewsArticle] = {}
        cursor_end = end
        while True:
            request = NewsRequest(
                symbols=symbol,
                start=start,
                end=cursor_end,
                sort="desc",
                limit=limit,
                include_content=include_content,
                exclude_contentless=False,
            )
            response = self._client.get_news(request)
            items = list(response.data["news"])
            if not items:
                break
            for item in items:
                article = NewsArticle(
                    article_id=int(item.id),
                    created_at=item.created_at.astimezone(timezone.utc).isoformat(),
                    headline=str(item.headline or ""),
                    summary=str(item.summary or ""),
                    symbols=tuple(str(value).upper() for value in item.symbols),
                    source=str(getattr(item.source, "name", getattr(item, "source", ""))),
                    url=str(item.url or ""),
                )
                articles_by_id[article.article_id] = article

            oldest_created_at = min(datetime.fromisoformat(article.created_at) for article in items_to_articles(items))
            if len(items) < limit:
                break
            next_cursor_end = oldest_created_at - timedelta(seconds=1)
            if next_cursor_end <= start or next_cursor_end >= cursor_end:
                break
            cursor_end = next_cursor_end

        articles = sorted(articles_by_id.values(), key=lambda item: item.created_at)
        cache_path.write_text(
            json.dumps([article.__dict__ for article in articles], indent=2),
            encoding="utf-8",
        )
        return articles

    def _cache_path(self, *, symbol: str, start: datetime, end: datetime, include_content: bool) -> Path:
        symbol_key = symbol.replace("/", "_").lower()
        content_key = "content" if include_content else "meta"
        return self._cache_dir / (
            f"{symbol_key}_{start.date().isoformat()}_{end.date().isoformat()}_{content_key}_v2.json"
        )


def items_to_articles(items: list[object]) -> list[NewsArticle]:
    return [
        NewsArticle(
            article_id=int(item.id),
            created_at=item.created_at.astimezone(timezone.utc).isoformat(),
            headline=str(item.headline or ""),
            summary=str(item.summary or ""),
            symbols=tuple(str(value).upper() for value in item.symbols),
            source=str(getattr(item.source, "name", getattr(item, "source", ""))),
            url=str(item.url or ""),
        )
        for item in items
    ]
