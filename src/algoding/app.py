from __future__ import annotations

from dataclasses import dataclass

from algoding.data.questdb import QuestDBClient
from algoding.execution.paper import PaperOrderRouter
from algoding.monitoring.logging import configure_logging
from algoding.settings import Settings
from algoding.shadow.null_sink import NullOrderSink


@dataclass
class AppContext:
    settings: Settings
    questdb: QuestDBClient
    shadow_sink: NullOrderSink
    paper_router: PaperOrderRouter


def build_app_context() -> AppContext:
    settings = Settings()
    configure_logging(settings.log_level)
    return AppContext(
        settings=settings,
        questdb=QuestDBClient(settings.questdb_http_url),
        shadow_sink=NullOrderSink(settings.shadow_sink_path),
        paper_router=PaperOrderRouter(settings),
    )
