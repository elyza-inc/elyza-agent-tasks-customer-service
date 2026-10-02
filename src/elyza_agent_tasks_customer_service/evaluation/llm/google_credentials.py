"""Google Application Default Credentials のアクセストークンを返す。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_REFRESH_MARGIN = timedelta(minutes=5)
_credentials = None
_credentials_lock = Lock()


def access_token() -> str:
    """Return an ADC token, refreshing it when it expires within five minutes."""

    global _credentials
    with _credentials_lock:
        if _credentials is None:
            import google.auth

            _credentials, _ = google.auth.default(scopes=_SCOPES)
        expiry = _credentials.expiry
        if (
            not _credentials.valid
            or expiry is None
            or expiry < datetime.now(timezone.utc).replace(tzinfo=None) + _REFRESH_MARGIN
        ):
            from google.auth.transport.requests import Request

            _credentials.refresh(Request())
        return _credentials.token
