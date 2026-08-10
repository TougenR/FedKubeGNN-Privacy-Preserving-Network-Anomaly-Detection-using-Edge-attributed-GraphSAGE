"""Small JSON HTTP client used between Phase 4 services."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ServiceRequestError(RuntimeError):
    pass


def post_json(
    url: str,
    document: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(
        url,
        data=json.dumps(document, separators=(",", ":")).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if int(response.status) < 200 or int(response.status) >= 300:
                raise ServiceRequestError(f"Service returned HTTP {response.status}.")
    except HTTPError as exc:
        raise ServiceRequestError(f"Service returned HTTP {exc.code}.") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ServiceRequestError("Service request failed.") from exc
    if not isinstance(payload, dict):
        raise ServiceRequestError("Service response must be a JSON object.")
    return payload


def get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers, method="GET")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if int(response.status) < 200 or int(response.status) >= 300:
                raise ServiceRequestError(f"Service returned HTTP {response.status}.")
    except HTTPError as exc:
        raise ServiceRequestError(f"Service returned HTTP {exc.code}.") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ServiceRequestError("Service request failed.") from exc
    if not isinstance(payload, dict):
        raise ServiceRequestError("Service response must be a JSON object.")
    return payload


def delete_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers, method="DELETE")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if int(response.status) < 200 or int(response.status) >= 300:
                raise ServiceRequestError(f"Service returned HTTP {response.status}.")
    except HTTPError as exc:
        raise ServiceRequestError(f"Service returned HTTP {exc.code}.") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ServiceRequestError("Service request failed.") from exc
    if not isinstance(payload, dict):
        raise ServiceRequestError("Service response must be a JSON object.")
    return payload
