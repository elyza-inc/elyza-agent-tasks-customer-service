"""Google credentials and static-token compatibility checks."""

import sys
import types
from datetime import datetime, timedelta, timezone


def test_access_token_uses_adc_and_caches_credentials(monkeypatch):
    import google

    from elyza_agent_tasks_customer_service.evaluation.llm import google_credentials

    class Credentials:
        valid = False
        expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        token = "x" * 254

        def refresh(self, request):
            self.valid = True

    credentials = Credentials()
    default_calls = []
    google_auth = types.ModuleType("google.auth")
    google_auth.default = lambda *, scopes: (default_calls.append(scopes) or credentials, None)
    google_requests = types.ModuleType("google.auth.transport.requests")
    google_requests.Request = object
    monkeypatch.setitem(sys.modules, "google.auth", google_auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", google_requests)
    monkeypatch.setattr(google, "auth", google_auth, raising=False)
    monkeypatch.setattr(google_credentials, "_credentials", None)

    assert google_credentials.access_token() == "x" * 254
    assert google_credentials.access_token() == "x" * 254
    assert default_calls == [["https://www.googleapis.com/auth/cloud-platform"]]


def test_static_operator_token_is_unchanged(monkeypatch):
    from elyza_agent_tasks_customer_service.evaluation.cli.run_eval import (
        build_operator_connection_headers,
    )

    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "dummy")
    assert build_operator_connection_headers({"operator_api_key_env": "GOOGLE_ACCESS_TOKEN"}) == {
        "Authorization": "Bearer dummy"
    }
