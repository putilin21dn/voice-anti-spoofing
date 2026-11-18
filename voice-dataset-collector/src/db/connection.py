from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import sql

from config.settings import Settings


class DatabaseError(Exception):
    """Ошибка подключения или чтения PostgreSQL без секретов в тексте."""


def _public_db_error(exc: BaseException, password: str) -> str:
    text = " ".join(str(exc).split())
    if password:
        text = text.replace(password, "***")
    return text or exc.__class__.__name__


def _require_db(settings: Settings) -> None:
    missing = [
        name
        for name, value in (
            ("DB_HOST", settings.db_host),
            ("DB_NAME", settings.db_name),
            ("DB_USER", settings.db_user),
        )
        if not value
    ]
    if missing:
        raise DatabaseError("Не заданы параметры подключения: " + ", ".join(missing))


@contextmanager
def connect(settings: Settings) -> Iterator[psycopg.Connection]:
    """Открывает сессию только для чтения и закрывает её в любом случае."""

    _require_db(settings)
    try:
        connection = psycopg.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=10,
            autocommit=True,
            # libpq по умолчанию предлагает GSSAPI. Этот сервер закрывает такое
            # соединение до ответа. Шифрование GSS здесь не используется.
            gssencmode="disable",
        )
    except psycopg.Error as exc:
        raise DatabaseError(
            "Не удалось подключиться к PostgreSQL: " + _public_db_error(exc, settings.db_password)
        ) from exc

    try:
        connection.execute("SET default_transaction_read_only = on")
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(settings.db_schema)))
        yield connection
    except psycopg.Error as exc:
        raise DatabaseError(f"Ошибка чтения PostgreSQL: {exc.__class__.__name__}: {exc.diag.message_primary or ''}".strip()) from exc
    finally:
        connection.close()


def check_connection(settings: Settings) -> dict[str, str]:
    with connect(settings) as connection:
        row = connection.execute(
            """
            SELECT current_database() AS database_name,
                   current_user AS user_name,
                   current_schema() AS schema_name,
                   to_regclass('calls') IS NOT NULL AS calls_table
            """
        ).fetchone()
    if row is None:
        raise DatabaseError("Проверка подключения не вернула строку")
    return {
        "database_name": str(row[0]),
        "user_name": str(row[1]),
        "schema_name": str(row[2]),
        "calls_table": "yes" if row[3] else "no",
    }
