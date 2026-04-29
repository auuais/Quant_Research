from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _load_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


@dataclass
class Settings:
    app_env: str = "local"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "algoding"
    postgres_user: str = "algoding"
    postgres_password: str = "algoding"

    questdb_http_host: str = "localhost"
    questdb_http_port: int = 9000
    questdb_pg_port: int = 8812

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_base_url: str = "https://data.alpaca.markets"
    alpaca_data_feed: str = "iex"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_news_model_name: str = "google/flan-t5-base"
    llm_news_cache_dir: Path = Path("cache/llm_news")
    llm_news_quantization: str = ""
    llm_news_force_gpu: bool = False

    default_universe: str = "SPY,QQQ,IWM,TLT,GLD,XLF,XLE,XLV,SMH"
    heartbeat_timeout_seconds: int = 30
    shadow_sink_path: Path = Path("logs/shadow_orders.jsonl")
    shadow_order_quantity: float = 1.0
    shadow_price_move_threshold_bps: int = 10
    paper_max_positions: int = 3
    paper_lookback_bars: int = 20
    paper_position_notional: float = 1000.0

    def __post_init__(self) -> None:
        env_file_values = _load_dotenv_file(Path(".env"))

        def env_value(key: str, default: str) -> str:
            return os.getenv(key, env_file_values.get(key, default))

        self.app_env = env_value("APP_ENV", self.app_env)
        self.log_level = env_value("LOG_LEVEL", self.log_level)
        self.postgres_host = env_value("POSTGRES_HOST", self.postgres_host)
        self.postgres_port = int(env_value("POSTGRES_PORT", str(self.postgres_port)))
        self.postgres_db = env_value("POSTGRES_DB", self.postgres_db)
        self.postgres_user = env_value("POSTGRES_USER", self.postgres_user)
        self.postgres_password = env_value("POSTGRES_PASSWORD", self.postgres_password)
        self.questdb_http_host = env_value("QUESTDB_HTTP_HOST", self.questdb_http_host)
        self.questdb_http_port = int(env_value("QUESTDB_HTTP_PORT", str(self.questdb_http_port)))
        self.questdb_pg_port = int(env_value("QUESTDB_PG_PORT", str(self.questdb_pg_port)))
        self.alpaca_api_key = env_value("ALPACA_API_KEY", self.alpaca_api_key)
        self.alpaca_secret_key = env_value("ALPACA_SECRET_KEY", self.alpaca_secret_key)
        self.alpaca_paper_base_url = env_value(
            "ALPACA_PAPER_BASE_URL", self.alpaca_paper_base_url
        )
        self.alpaca_data_base_url = env_value("ALPACA_DATA_BASE_URL", self.alpaca_data_base_url)
        self.alpaca_data_feed = env_value("ALPACA_DATA_FEED", self.alpaca_data_feed)
        self.openai_api_key = env_value("OPENAI_API_KEY", self.openai_api_key)
        self.openai_base_url = env_value("OPENAI_BASE_URL", self.openai_base_url)
        self.llm_news_model_name = env_value("LLM_NEWS_MODEL_NAME", self.llm_news_model_name)
        self.llm_news_cache_dir = Path(env_value("LLM_NEWS_CACHE_DIR", str(self.llm_news_cache_dir)))
        self.llm_news_quantization = env_value("LLM_NEWS_QUANTIZATION", self.llm_news_quantization)
        self.llm_news_force_gpu = env_value("LLM_NEWS_FORCE_GPU", str(self.llm_news_force_gpu)).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.default_universe = env_value("DEFAULT_UNIVERSE", self.default_universe)
        self.heartbeat_timeout_seconds = int(
            env_value("HEARTBEAT_TIMEOUT_SECONDS", str(self.heartbeat_timeout_seconds))
        )
        self.shadow_sink_path = Path(env_value("SHADOW_SINK_PATH", str(self.shadow_sink_path)))
        self.shadow_order_quantity = float(
            env_value("SHADOW_ORDER_QUANTITY", str(self.shadow_order_quantity))
        )
        self.shadow_price_move_threshold_bps = int(
            env_value(
                "SHADOW_PRICE_MOVE_THRESHOLD_BPS", str(self.shadow_price_move_threshold_bps)
            )
        )
        self.paper_max_positions = int(env_value("PAPER_MAX_POSITIONS", str(self.paper_max_positions)))
        self.paper_lookback_bars = int(env_value("PAPER_LOOKBACK_BARS", str(self.paper_lookback_bars)))
        self.paper_position_notional = float(
            env_value("PAPER_POSITION_NOTIONAL", str(self.paper_position_notional))
        )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def questdb_http_url(self) -> str:
        return f"http://{self.questdb_http_host}:{self.questdb_http_port}"

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    def resolve_symbols(self, symbols: str | None = None) -> list[str]:
        raw = symbols if symbols else self.default_universe
        return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if payload["alpaca_api_key"]:
            payload["alpaca_api_key"] = self._redact(payload["alpaca_api_key"])
        if payload["alpaca_secret_key"]:
            payload["alpaca_secret_key"] = self._redact(payload["alpaca_secret_key"])
        if payload["openai_api_key"]:
            payload["openai_api_key"] = self._redact(payload["openai_api_key"])
        payload["postgres_dsn"] = self.postgres_dsn
        payload["questdb_http_url"] = self.questdb_http_url
        payload["shadow_sink_path"] = str(self.shadow_sink_path)
        payload["llm_news_cache_dir"] = str(self.llm_news_cache_dir)
        payload["has_alpaca_credentials"] = self.has_alpaca_credentials
        return payload

    @staticmethod
    def _redact(value: str) -> str:
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"
