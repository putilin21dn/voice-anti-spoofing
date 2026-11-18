from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _raw(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(name: str) -> int | None:
    raw = _raw(name)
    if not raw:
        return None
    return int(raw)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_schema: str
    s3_endpoint_url: str
    s3_host: str
    s3_port: int | None
    s3_secure: bool
    s3_region: str
    s3_bucket: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_session_token: str

    @property
    def manifest_path(self) -> Path:
        return self.project_root / "data" / "manifests" / "calls_manifest.csv"

    @property
    def audio_dir(self) -> Path:
        return self.project_root / "data" / "audio"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "data" / "logs"

    def s3_endpoint(self) -> str | None:
        """Тот же выбор endpoint, что в worker/backend.

        Явный S3_ENDPOINT_URL имеет приоритет.
        Host с amazonaws.com и пустой host означают обычный AWS endpoint.
        Иначе собирается http(s)://host[:port], как в S3Client worker и backend.
        """

        explicit = self.s3_endpoint_url.strip()
        if explicit:
            return explicit
        host = self.s3_host.strip()
        if not host or "amazonaws.com" in host:
            return None
        protocol = "https" if self.s3_secure else "http"
        if self.s3_port:
            return f"{protocol}://{host}:{self.s3_port}"
        return f"{protocol}://{host}"

    def public_summary(self) -> str:
        endpoint = self.s3_endpoint() or f"aws:{self.s3_region}"
        return (
            f"db={self.db_user}@{self.db_host}:{self.db_port}/{self.db_name} "
            f"schema={self.db_schema} s3_bucket={self.s3_bucket} s3_endpoint={endpoint}"
        )


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    schema = _raw("DB_SCHEMA", "public") or "public"
    if not IDENTIFIER_RE.fullmatch(schema):
        raise ValueError("DB_SCHEMA должен быть простым SQL-идентификатором")

    port_raw = _raw("DB_PORT", "5432") or "5432"
    return Settings(
        project_root=PROJECT_ROOT,
        db_host=_raw("DB_HOST"),
        db_port=int(port_raw),
        db_name=_raw("DB_NAME"),
        db_user=_raw("DB_USER"),
        db_password=os.environ.get("DB_PASSWORD", ""),
        db_schema=schema,
        s3_endpoint_url=_raw("S3_ENDPOINT_URL"),
        s3_host=_raw("S3_HOST"),
        s3_port=_int_or_none("S3_PORT"),
        s3_secure=_bool("S3_SECURE", False),
        s3_region=_raw("S3_REGION", "eu-central-1") or "eu-central-1",
        s3_bucket=_raw("S3_BUCKET", "genai") or "genai",
        s3_access_key_id=_raw("S3_ACCESS_KEY_ID"),
        s3_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
        s3_session_token=os.environ.get("S3_SESSION_TOKEN", ""),
    )
