from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import psycopg

from algoding.execution.models import OrderIntent
from algoding.settings import Settings

logger = logging.getLogger(__name__)


class OrderStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._paper_fallback_path = Path("logs/paper_orders.jsonl")
        self._strategy_fallback_path = Path("logs/strategy_runs.jsonl")
        self._postgres_available: bool | None = None

    def _dsn(self) -> str:
        return self._settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")

    def _connect(self) -> psycopg.Connection:
        if self._postgres_available is False:
            raise RuntimeError("Postgres unavailable for current store session.")
        try:
            conn = psycopg.connect(self._dsn(), connect_timeout=1)
        except Exception:
            self._postgres_available = False
            raise
        self._postgres_available = True
        return conn

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str))
            handle.write("\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        results: list[dict[str, object]] = []
        decoder = json.JSONDecoder()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cursor = 0
            while cursor < len(line):
                try:
                    payload, offset = decoder.raw_decode(line, cursor)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL payload in %s: %s", path, line[:120])
                    break
                if isinstance(payload, dict):
                    results.append(payload)
                cursor = offset
                while cursor < len(line) and line[cursor].isspace():
                    cursor += 1
        return results

    def ensure_schema(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        create table if not exists paper_orders (
                            client_order_id text primary key,
                            strategy_name text not null,
                            symbol text not null,
                            side text not null,
                            quantity double precision,
                            notional double precision,
                            status text not null,
                            broker_order_id text,
                            broker text not null,
                            created_at timestamptz not null,
                            updated_at timestamptz,
                            filled_at timestamptz,
                            filled_qty double precision,
                            filled_avg_price double precision,
                            payload jsonb not null
                        )
                        """
                    )
                    cursor.execute(
                        """
                        alter table paper_orders
                        add column if not exists updated_at timestamptz,
                        add column if not exists filled_at timestamptz,
                        add column if not exists filled_qty double precision,
                        add column if not exists filled_avg_price double precision
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists strategy_runs (
                            id bigserial primary key,
                            strategy_name text not null,
                            symbol text not null,
                            run_type text not null,
                            total_return double precision not null,
                            max_drawdown double precision not null,
                            trades integer not null,
                            win_rate double precision not null,
                            latest_signal text not null,
                            score double precision not null,
                            created_at timestamptz not null default now(),
                            payload jsonb not null
                        )
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists paper_positions (
                            symbol text primary key,
                            qty double precision not null,
                            avg_entry_price double precision,
                            current_price double precision,
                            market_value double precision,
                            unrealized_pl double precision,
                            change_today double precision,
                            updated_at timestamptz not null,
                            payload jsonb not null
                        )
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists account_snapshots (
                            id bigserial primary key,
                            snapshot_name text not null,
                            equity double precision,
                            cash double precision,
                            buying_power double precision,
                            positions_count integer not null,
                            created_at timestamptz not null,
                            payload jsonb not null
                        )
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists virtual_strategy_books (
                            strategy_name text not null,
                            symbol text not null,
                            target_notional double precision not null,
                            position_qty double precision not null,
                            avg_entry_price double precision,
                            realized_pnl double precision not null,
                            unrealized_pnl double precision not null,
                            total_pnl double precision not null,
                            last_signal text not null,
                            last_price double precision not null,
                            updated_at timestamptz not null,
                            payload jsonb not null,
                            primary key (strategy_name, symbol)
                        )
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists virtual_strategy_trades (
                            id bigserial primary key,
                            strategy_name text not null,
                            symbol text not null,
                            side text not null,
                            notional double precision,
                            quantity double precision,
                            price double precision not null,
                            signal text not null,
                            created_at timestamptz not null,
                            payload jsonb not null
                        )
                        """
                    )
                    cursor.execute(
                        """
                        create table if not exists virtual_strategy_snapshots (
                            id bigserial primary key,
                            strategy_name text not null,
                            symbol text not null,
                            target_notional double precision not null,
                            position_qty double precision not null,
                            avg_entry_price double precision,
                            realized_pnl double precision not null,
                            unrealized_pnl double precision not null,
                            total_pnl double precision not null,
                            last_signal text not null,
                            last_price double precision not null,
                            score double precision,
                            win_rate double precision,
                            historical_total_return double precision,
                            historical_max_drawdown double precision,
                            historical_trades integer,
                            created_at timestamptz not null,
                            payload jsonb not null
                        )
                        """
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres schema setup unavailable, using JSONL fallback: %s", exc)

    def record_order(
        self,
        intent: OrderIntent,
        payload: dict[str, object],
        status: str,
        broker_order_id: str = "",
    ) -> None:
        serialized = asdict(intent)
        serialized["created_at"] = intent.created_at.isoformat()
        record = {
            "client_order_id": intent.client_order_id,
            "strategy_name": intent.strategy_name,
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "notional": intent.notional,
            "status": status,
            "broker_order_id": broker_order_id,
            "broker": "alpaca",
            "created_at": intent.created_at.isoformat(),
            "payload": {"intent": serialized, "result": payload},
        }
        self._append_jsonl(self._paper_fallback_path, record)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into paper_orders (
                            client_order_id,
                            strategy_name,
                            symbol,
                            side,
                            quantity,
                            notional,
                            status,
                            broker_order_id,
                            broker,
                            created_at,
                            updated_at,
                            filled_at,
                            filled_qty,
                            filled_avg_price,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        on conflict (client_order_id) do update
                        set status = excluded.status,
                            broker_order_id = excluded.broker_order_id,
                            updated_at = excluded.updated_at,
                            filled_at = excluded.filled_at,
                            filled_qty = excluded.filled_qty,
                            filled_avg_price = excluded.filled_avg_price,
                            payload = excluded.payload
                        """,
                        (
                            intent.client_order_id,
                            intent.strategy_name,
                            intent.symbol,
                            intent.side,
                            intent.quantity,
                            intent.notional,
                            status,
                            broker_order_id,
                            "alpaca",
                            intent.created_at,
                            intent.created_at,
                            None,
                            None,
                            None,
                            json.dumps(record["payload"], default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres order write unavailable, kept JSONL fallback: %s", exc)

    def sync_broker_order(self, payload: dict[str, object]) -> None:
        self._append_jsonl(self._paper_fallback_path, payload)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into paper_orders (
                            client_order_id,
                            strategy_name,
                            symbol,
                            side,
                            quantity,
                            notional,
                            status,
                            broker_order_id,
                            broker,
                            created_at,
                            updated_at,
                            filled_at,
                            filled_qty,
                            filled_avg_price,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        on conflict (client_order_id) do update
                        set status = excluded.status,
                            broker_order_id = excluded.broker_order_id,
                            updated_at = excluded.updated_at,
                            filled_at = excluded.filled_at,
                            filled_qty = excluded.filled_qty,
                            filled_avg_price = excluded.filled_avg_price,
                            payload = excluded.payload
                        """,
                        (
                            payload["client_order_id"],
                            payload.get("strategy_name", "alpaca_sync"),
                            payload["symbol"],
                            payload["side"],
                            payload.get("quantity"),
                            payload.get("notional"),
                            payload["status"],
                            payload.get("broker_order_id", ""),
                            "alpaca",
                            payload["created_at"],
                            payload.get("updated_at"),
                            payload.get("filled_at"),
                            payload.get("filled_qty"),
                            payload.get("filled_avg_price"),
                            json.dumps(payload, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres order sync unavailable, kept JSONL fallback: %s", exc)

    def sync_position(self, payload: dict[str, object]) -> None:
        positions_path = Path("logs/paper_positions.jsonl")
        self._append_jsonl(positions_path, payload)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into paper_positions (
                            symbol,
                            qty,
                            avg_entry_price,
                            current_price,
                            market_value,
                            unrealized_pl,
                            change_today,
                            updated_at,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        on conflict (symbol) do update
                        set qty = excluded.qty,
                            avg_entry_price = excluded.avg_entry_price,
                            current_price = excluded.current_price,
                            market_value = excluded.market_value,
                            unrealized_pl = excluded.unrealized_pl,
                            change_today = excluded.change_today,
                            updated_at = excluded.updated_at,
                            payload = excluded.payload
                        """,
                        (
                            payload["symbol"],
                            payload["qty"],
                            payload.get("avg_entry_price"),
                            payload.get("current_price"),
                            payload.get("market_value"),
                            payload.get("unrealized_pl"),
                            payload.get("change_today"),
                            payload["updated_at"],
                            json.dumps(payload, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres position sync unavailable, kept JSONL fallback: %s", exc)

    def record_account_snapshot(self, snapshot: dict[str, object]) -> None:
        snapshots_path = Path("logs/account_snapshots.jsonl")
        self._append_jsonl(snapshots_path, snapshot)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into account_snapshots (
                            snapshot_name,
                            equity,
                            cash,
                            buying_power,
                            positions_count,
                            created_at,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            snapshot["snapshot_name"],
                            snapshot.get("equity"),
                            snapshot.get("cash"),
                            snapshot.get("buying_power"),
                            snapshot["positions_count"],
                            snapshot["created_at"],
                            json.dumps(snapshot, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres snapshot write unavailable, kept JSONL fallback: %s", exc)

    def list_account_snapshots(self, limit: int = 10) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        select snapshot_name, equity, cash, buying_power, positions_count, created_at, payload
                        from account_snapshots
                        order by created_at desc
                        limit %s
                        """,
                        (limit,),
                    )
                    rows = cursor.fetchall()
            return [
                {
                    "snapshot_name": row[0],
                    "equity": row[1],
                    "cash": row[2],
                    "buying_power": row[3],
                    "positions_count": row[4],
                    "created_at": row[5].isoformat(),
                    "payload": row[6],
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning("Postgres snapshot read unavailable, using JSONL fallback: %s", exc)
            rows = self._read_jsonl(Path("logs/account_snapshots.jsonl"))
            return list(reversed(rows[-limit:]))

    def upsert_virtual_strategy_book(self, book: dict[str, object]) -> None:
        books_path = Path("logs/virtual_strategy_books.jsonl")
        self._append_jsonl(books_path, book)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into virtual_strategy_books (
                            strategy_name,
                            symbol,
                            target_notional,
                            position_qty,
                            avg_entry_price,
                            realized_pnl,
                            unrealized_pnl,
                            total_pnl,
                            last_signal,
                            last_price,
                            updated_at,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        on conflict (strategy_name, symbol) do update
                        set target_notional = excluded.target_notional,
                            position_qty = excluded.position_qty,
                            avg_entry_price = excluded.avg_entry_price,
                            realized_pnl = excluded.realized_pnl,
                            unrealized_pnl = excluded.unrealized_pnl,
                            total_pnl = excluded.total_pnl,
                            last_signal = excluded.last_signal,
                            last_price = excluded.last_price,
                            updated_at = excluded.updated_at,
                            payload = excluded.payload
                        """,
                        (
                            book["strategy_name"],
                            book["symbol"],
                            book["target_notional"],
                            book["position_qty"],
                            book.get("avg_entry_price"),
                            book["realized_pnl"],
                            book["unrealized_pnl"],
                            book["total_pnl"],
                            book["last_signal"],
                            book["last_price"],
                            book["updated_at"],
                            json.dumps(book, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres virtual book write unavailable, kept JSONL fallback: %s", exc)

    def record_virtual_strategy_trade(self, trade: dict[str, object]) -> None:
        trades_path = Path("logs/virtual_strategy_trades.jsonl")
        self._append_jsonl(trades_path, trade)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into virtual_strategy_trades (
                            strategy_name,
                            symbol,
                            side,
                            notional,
                            quantity,
                            price,
                            signal,
                            created_at,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            trade["strategy_name"],
                            trade["symbol"],
                            trade["side"],
                            trade.get("notional"),
                            trade.get("quantity"),
                            trade["price"],
                            trade["signal"],
                            trade["created_at"],
                            json.dumps(trade, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres virtual trade write unavailable, kept JSONL fallback: %s", exc)

    def get_virtual_strategy_books(self, symbol: str) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        select strategy_name, symbol, target_notional, position_qty, avg_entry_price,
                               realized_pnl, unrealized_pnl, total_pnl, last_signal, last_price,
                               updated_at, payload
                        from virtual_strategy_books
                        where symbol = %s
                        order by strategy_name
                        """,
                        (symbol,),
                    )
                    rows = cursor.fetchall()
            return [
                {
                    "strategy_name": row[0],
                    "symbol": row[1],
                    "target_notional": row[2],
                    "position_qty": row[3],
                    "avg_entry_price": row[4],
                    "realized_pnl": row[5],
                    "unrealized_pnl": row[6],
                    "total_pnl": row[7],
                    "last_signal": row[8],
                    "last_price": row[9],
                    "updated_at": row[10].isoformat(),
                    "payload": row[11],
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning("Postgres virtual book read unavailable, using JSONL fallback: %s", exc)
            rows = [row for row in self._read_jsonl(Path("logs/virtual_strategy_books.jsonl")) if row.get("symbol") == symbol]
            latest: dict[tuple[str, str], dict[str, object]] = {}
            for row in rows:
                latest[(str(row["strategy_name"]), str(row["symbol"]))] = row
            return list(latest.values())

    def list_virtual_strategy_trades(self, symbol: str, limit: int = 20) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        select strategy_name, symbol, side, notional, quantity, price, signal, created_at, payload
                        from virtual_strategy_trades
                        where symbol = %s
                        order by created_at desc
                        limit %s
                        """,
                        (symbol, limit),
                    )
                    rows = cursor.fetchall()
            return [
                {
                    "strategy_name": row[0],
                    "symbol": row[1],
                    "side": row[2],
                    "notional": row[3],
                    "quantity": row[4],
                    "price": row[5],
                    "signal": row[6],
                    "created_at": row[7].isoformat(),
                    "payload": row[8],
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning("Postgres virtual trade read unavailable, using JSONL fallback: %s", exc)
            rows = [row for row in self._read_jsonl(Path("logs/virtual_strategy_trades.jsonl")) if row.get("symbol") == symbol]
            return list(reversed(rows[-limit:]))

    def record_virtual_strategy_snapshot(self, snapshot: dict[str, object]) -> None:
        snapshots_path = Path("logs/virtual_strategy_snapshots.jsonl")
        self._append_jsonl(snapshots_path, snapshot)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into virtual_strategy_snapshots (
                            strategy_name,
                            symbol,
                            target_notional,
                            position_qty,
                            avg_entry_price,
                            realized_pnl,
                            unrealized_pnl,
                            total_pnl,
                            last_signal,
                            last_price,
                            score,
                            win_rate,
                            historical_total_return,
                            historical_max_drawdown,
                            historical_trades,
                            created_at,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            snapshot["strategy_name"],
                            snapshot["symbol"],
                            snapshot["target_notional"],
                            snapshot["position_qty"],
                            snapshot.get("avg_entry_price"),
                            snapshot["realized_pnl"],
                            snapshot["unrealized_pnl"],
                            snapshot["total_pnl"],
                            snapshot["last_signal"],
                            snapshot["last_price"],
                            snapshot.get("score"),
                            snapshot.get("win_rate"),
                            snapshot.get("historical_total_return"),
                            snapshot.get("historical_max_drawdown"),
                            snapshot.get("historical_trades"),
                            snapshot["created_at"],
                            json.dumps(snapshot, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres virtual snapshot write unavailable, kept JSONL fallback: %s", exc)

    def list_virtual_strategy_snapshots(
        self, symbol: str, strategy_name: str | None = None, limit: int = 100
    ) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    if strategy_name:
                        cursor.execute(
                            """
                            select strategy_name, symbol, target_notional, position_qty, avg_entry_price,
                                   realized_pnl, unrealized_pnl, total_pnl, last_signal, last_price,
                                   score, win_rate, historical_total_return, historical_max_drawdown,
                                   historical_trades, created_at, payload
                            from virtual_strategy_snapshots
                            where symbol = %s and strategy_name = %s
                            order by created_at desc
                            limit %s
                            """,
                            (symbol, strategy_name, limit),
                        )
                    else:
                        cursor.execute(
                            """
                            select strategy_name, symbol, target_notional, position_qty, avg_entry_price,
                                   realized_pnl, unrealized_pnl, total_pnl, last_signal, last_price,
                                   score, win_rate, historical_total_return, historical_max_drawdown,
                                   historical_trades, created_at, payload
                            from virtual_strategy_snapshots
                            where symbol = %s
                            order by created_at desc
                            limit %s
                            """,
                            (symbol, limit),
                        )
                    rows = cursor.fetchall()
            return [
                {
                    "strategy_name": row[0],
                    "symbol": row[1],
                    "target_notional": row[2],
                    "position_qty": row[3],
                    "avg_entry_price": row[4],
                    "realized_pnl": row[5],
                    "unrealized_pnl": row[6],
                    "total_pnl": row[7],
                    "last_signal": row[8],
                    "last_price": row[9],
                    "score": row[10],
                    "win_rate": row[11],
                    "historical_total_return": row[12],
                    "historical_max_drawdown": row[13],
                    "historical_trades": row[14],
                    "created_at": row[15].isoformat(),
                    "payload": row[16],
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning("Postgres virtual snapshot read unavailable, using JSONL fallback: %s", exc)
            rows = [row for row in self._read_jsonl(Path("logs/virtual_strategy_snapshots.jsonl")) if row.get("symbol") == symbol]
            if strategy_name:
                rows = [row for row in rows if row.get("strategy_name") == strategy_name]
            return list(reversed(rows[-limit:]))

    def get_virtual_strategy_leaderboard(self, symbol: str) -> list[dict[str, object]]:
        books = self.get_virtual_strategy_books(symbol)
        snapshots = self.list_virtual_strategy_snapshots(symbol, limit=500)
        history_count: dict[str, int] = {}
        first_seen: dict[str, str] = {}
        for snapshot in snapshots:
            name = str(snapshot["strategy_name"])
            history_count[name] = history_count.get(name, 0) + 1
            if name not in first_seen or str(snapshot["created_at"]) < first_seen[name]:
                first_seen[name] = str(snapshot["created_at"])

        leaderboard: list[dict[str, object]] = []
        for book in books:
            name = str(book["strategy_name"])
            payload = dict(book.get("payload", {})) if isinstance(book.get("payload"), dict) else {}
            leaderboard.append(
                {
                    "strategy_name": name,
                    "symbol": book["symbol"],
                    "total_pnl": book["total_pnl"],
                    "realized_pnl": book["realized_pnl"],
                    "unrealized_pnl": book["unrealized_pnl"],
                    "last_signal": book["last_signal"],
                    "score": payload.get("score"),
                    "win_rate": payload.get("win_rate"),
                    "historical_total_return": payload.get("historical_total_return"),
                    "historical_max_drawdown": payload.get("historical_max_drawdown"),
                    "historical_trades": payload.get("historical_trades"),
                    "snapshots_recorded": history_count.get(name, 0),
                    "first_seen": first_seen.get(name),
                    "updated_at": book["updated_at"],
                }
            )
        leaderboard.sort(key=lambda item: (float(item["total_pnl"]), float(item.get("score") or 0.0)), reverse=True)
        return leaderboard

    def record_strategy_run(self, summary: dict[str, object]) -> None:
        self._append_jsonl(self._strategy_fallback_path, summary)
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into strategy_runs (
                            strategy_name,
                            symbol,
                            run_type,
                            total_return,
                            max_drawdown,
                            trades,
                            win_rate,
                            latest_signal,
                            score,
                            payload
                        ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            summary["strategy_name"],
                            summary["symbol"],
                            summary["run_type"],
                            summary["total_return"],
                            summary["max_drawdown"],
                            summary["trades"],
                            summary["win_rate"],
                            summary["latest_signal"],
                            summary["score"],
                            json.dumps(summary, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Postgres strategy write unavailable, kept JSONL fallback: %s", exc)

    def list_latest_strategy_runs(self, symbol: str, limit: int = 10) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        select strategy_name, symbol, run_type, total_return, max_drawdown, trades,
                               win_rate, latest_signal, score, created_at, payload
                        from strategy_runs
                        where symbol = %s
                        order by created_at desc
                        limit %s
                        """,
                        (symbol, limit),
                    )
                    rows = cursor.fetchall()
            results: list[dict[str, object]] = []
            for row in rows:
                results.append(
                    {
                        "strategy_name": row[0],
                        "symbol": row[1],
                        "run_type": row[2],
                        "total_return": row[3],
                        "max_drawdown": row[4],
                        "trades": row[5],
                        "win_rate": row[6],
                        "latest_signal": row[7],
                        "score": row[8],
                        "created_at": row[9].isoformat(),
                        "payload": row[10],
                    }
                )
            return results
        except Exception as exc:
            logger.warning("Postgres strategy read unavailable, using JSONL fallback: %s", exc)
            rows = [row for row in self._read_jsonl(self._strategy_fallback_path) if row.get("symbol") == symbol]
            return list(reversed(rows[-limit:]))

    def list_all_strategy_runs(self, run_type: str | None = None, limit: int = 1000) -> list[dict[str, object]]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    if run_type:
                        cursor.execute(
                            """
                            select strategy_name, symbol, run_type, total_return, max_drawdown, trades,
                                   win_rate, latest_signal, score, created_at, payload
                            from strategy_runs
                            where run_type = %s
                            order by created_at desc
                            limit %s
                            """,
                            (run_type, limit),
                        )
                    else:
                        cursor.execute(
                            """
                            select strategy_name, symbol, run_type, total_return, max_drawdown, trades,
                                   win_rate, latest_signal, score, created_at, payload
                            from strategy_runs
                            order by created_at desc
                            limit %s
                            """,
                            (limit,),
                        )
                    rows = cursor.fetchall()
            results: list[dict[str, object]] = []
            for row in rows:
                payload = row[10] if isinstance(row[10], dict) else {}
                item = {
                    "strategy_name": row[0],
                    "symbol": row[1],
                    "run_type": row[2],
                    "total_return": row[3],
                    "max_drawdown": row[4],
                    "trades": row[5],
                    "win_rate": row[6],
                    "latest_signal": row[7],
                    "score": row[8],
                    "created_at": row[9].isoformat(),
                    "payload": row[10],
                }
                if isinstance(payload, dict):
                    item.update(payload)
                results.append(item)
            return results
        except Exception as exc:
            logger.warning("Postgres strategy list unavailable, using JSONL fallback: %s", exc)
            rows = self._read_jsonl(self._strategy_fallback_path)
            if run_type:
                rows = [row for row in rows if row.get("run_type") == run_type]
            return list(reversed(rows[-limit:]))
