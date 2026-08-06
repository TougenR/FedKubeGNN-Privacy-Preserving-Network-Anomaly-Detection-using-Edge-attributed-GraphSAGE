"""Minimal Elasticsearch sink for privacy-reduced detection events."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.application.alerting.privacy import validate_elasticsearch_document


class ElasticsearchSinkError(RuntimeError):
    """Raised when an event cannot be durably accepted by Elasticsearch."""


@dataclass(frozen=True)
class ElasticsearchSettings:
    endpoint: str
    index: str
    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("Elasticsearch endpoint must use HTTP or HTTPS.")
        if not self.index or any(character in self.index for character in "\\/*?\"<>| ,#:"):
            raise ValueError("Elasticsearch index has unsafe characters.")
        if self.api_key and (self.username or self.password):
            raise ValueError("Choose API-key or basic authentication, not both.")
        if bool(self.username) != bool(self.password):
            raise ValueError("Basic authentication requires username and password.")
        if self.timeout_seconds <= 0:
            raise ValueError("Elasticsearch timeout must be positive.")


class ElasticsearchSink:
    def __init__(
        self,
        settings: ElasticsearchSettings,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.settings = settings
        self._opener = opener

    def index_event(self, document: Mapping[str, Any]) -> str:
        validate_elasticsearch_document(document)
        target = (
            self.settings.endpoint.rstrip("/")
            + "/"
            + quote(self.settings.index, safe="-_.")
            + "/_doc"
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"ApiKey {self.settings.api_key}"
        elif self.settings.username and self.settings.password:
            credential = base64.b64encode(
                f"{self.settings.username}:{self.settings.password}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {credential}"
        request = Request(
            target,
            data=json.dumps(document, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if int(response.status) not in {200, 201}:
                    raise ElasticsearchSinkError(
                        f"Elasticsearch rejected event with HTTP {response.status}."
                    )
        except HTTPError as exc:
            raise ElasticsearchSinkError(
                f"Elasticsearch rejected event with HTTP {exc.code}."
            ) from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise ElasticsearchSinkError("Elasticsearch request failed.") from exc
        document_id = payload.get("_id")
        if not isinstance(document_id, str) or not document_id:
            raise ElasticsearchSinkError("Elasticsearch response has no document ID.")
        return document_id
