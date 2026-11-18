from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, ConnectionError as BotocoreConnectionError

from config.settings import Settings


class S3StorageError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ObjectInfo:
    bucket: str
    key: str
    size: int
    content_type: str


NOT_FOUND = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}
AUTH_ERRORS = {
    "403",
    "AccessDenied",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "ExpiredToken",
    "InvalidToken",
}


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


class ReadOnlyS3:
    """Чтение объектов. Загрузка, удаление и изменение объектов не реализованы."""

    def __init__(self, settings: Settings):
        self._bucket = settings.s3_bucket
        self._endpoint = settings.s3_endpoint()
        client_kwargs: dict = {"region_name": settings.s3_region}
        if settings.s3_access_key_id:
            client_kwargs["aws_access_key_id"] = settings.s3_access_key_id
        if settings.s3_secret_access_key:
            client_kwargs["aws_secret_access_key"] = settings.s3_secret_access_key
        if settings.s3_session_token.strip():
            client_kwargs["aws_session_token"] = settings.s3_session_token.strip()
        if self._endpoint:
            client_kwargs["endpoint_url"] = self._endpoint
            client_kwargs["config"] = Config(s3={"addressing_style": "path"})
        self._client = boto3.client("s3", **client_kwargs)

    @property
    def bucket(self) -> str:
        return self._bucket

    def check_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            self._raise_client(exc, key="")
        except (EndpointConnectionError, BotocoreConnectionError, BotoCoreError) as exc:
            raise S3StorageError("network_error", self._network_message(exc)) from exc

    def head(self, key: str) -> ObjectInfo:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            self._raise_client(exc, key=key)
        except (EndpointConnectionError, BotocoreConnectionError, BotoCoreError) as exc:
            raise S3StorageError("network_error", self._network_message(exc)) from exc
        return ObjectInfo(
            bucket=self._bucket,
            key=key,
            size=int(response.get("ContentLength") or 0),
            content_type=str(response.get("ContentType") or ""),
        )

    def download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            self._client.download_file(self._bucket, key, str(temporary))
            temporary.replace(destination)
        except ClientError as exc:
            temporary.unlink(missing_ok=True)
            self._raise_client(exc, key=key)
        except (EndpointConnectionError, BotocoreConnectionError, BotoCoreError) as exc:
            temporary.unlink(missing_ok=True)
            raise S3StorageError("network_error", self._network_message(exc)) from exc
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise S3StorageError("download_error", f"Не удалось записать файл: {exc.__class__.__name__}") from exc

    def _network_message(self, exc: BaseException) -> str:
        detail = " ".join(str(exc).split())
        target = self._endpoint or self._bucket
        return f"Нет ответа S3 от {target}: {detail}"

    def _raise_client(self, exc: ClientError, *, key: str) -> None:
        code = _error_code(exc)
        if code in NOT_FOUND:
            target = key or self._bucket
            raise S3StorageError("missing", f"Объект не найден: {target}") from exc
        if code in AUTH_ERRORS:
            raise S3StorageError("auth_error", f"Ошибка доступа к S3: {code}") from exc
        raise S3StorageError("download_error", f"Ошибка S3: {code or exc.__class__.__name__}") from exc
