from __future__ import annotations

import logging
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.autonomous.context.lark_source as lark_source
from src.autonomous.context import (
    ContextUnavailableError,
    ContextUnavailableReason,
    EmployeeMessageScope,
)
from src.autonomous.domain.employees import BotPrincipal


class _Vault:
    def __init__(self, secrets):
        self.secrets = secrets
        self.calls = []

    def resolve(self, credential_ref, agent_id, app_id):
        self.calls.append((credential_ref, agent_id, app_id))
        return self.secrets[credential_ref]


def _scope(n: int = 1, **overrides: str) -> EmployeeMessageScope:
    values = {
        "tenant_key": "tenant_1",
        "agent_id": f"agt_{n}",
        "bot_principal_id": f"bot_{n}",
        "app_id": f"cli_{n}",
        "chat_id": f"oc_{n}",
        "thread_root_message_id": f"om_root_{n}",
        "current_message_id": f"om_current_{n}",
    }
    values.update(overrides)
    return EmployeeMessageScope(**values)


def _principal(n: int = 1, **overrides: str) -> BotPrincipal:
    values = {
        "bot_principal_id": f"bot_{n}",
        "tenant_key": "tenant_1",
        "agent_id": f"agt_{n}",
        "app_id": f"cli_{n}",
        "credential_ref": f"cred_{n}",
    }
    values.update(overrides)
    return BotPrincipal(**values)


def _factory(*, vault, client_builder):
    with patch.object(lark_source, "_default_client_builder", client_builder):
        return lark_source.LarkEmployeeMessageSourceFactory(
            credential_resolver=vault,
        )


def test_employee_credentials_are_isolated_and_never_rendered(caplog) -> None:
    secrets = {"cred_1": "secret-alpha", "cred_2": "secret-bravo"}
    vault = _Vault(secrets)
    builds = []

    class API:
        def __init__(self, secret: str) -> None:
            self.secret = secret

        def get(self, _request):
            raise RuntimeError(f"upstream accidentally included {self.secret}")

    def builder(*, app_id, app_secret, timeout):
        builds.append((app_id, app_secret, timeout))
        return SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=API(app_secret)))
        )

    factory = _factory(vault=vault, client_builder=builder)
    with caplog.at_level(logging.DEBUG):
        for n in (1, 2):
            source = factory.open(scope=_scope(n), principal=_principal(n))
            with source, pytest.raises(ContextUnavailableError) as raised:
                source.resolve_thread()
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert raised.value.reason is ContextUnavailableReason.SOURCE
            assert all(secret not in rendered for secret in secrets.values())

    assert vault.calls == [
        ("cred_1", "agt_1", "cli_1"),
        ("cred_2", "agt_2", "cli_2"),
    ]
    assert builds == [
        ("cli_1", "secret-alpha", 10.0),
        ("cli_2", "secret-bravo", 10.0),
    ]
    exposed = f"{factory!r}\n{caplog.text}"
    assert all(secret not in exposed for secret in secrets.values())


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("tenant_key", "tenant_other"),
        ("agent_id", "agt_other"),
        ("bot_principal_id", "bot_other"),
        ("app_id", "cli_other"),
    ],
)
def test_scope_and_principal_four_tuple_must_match(
    field: str,
    mismatch: str,
) -> None:
    vault = _Vault({"cred_1": "secret-alpha"})
    factory = _factory(
        vault=vault,
        client_builder=lambda **_: pytest.fail("client must not be built"),
    )

    with pytest.raises(ContextUnavailableError) as raised:
        with factory.open(
            scope=_scope(**{field: mismatch}),
            principal=_principal(),
        ):
            pytest.fail("mismatched source must not open")

    assert raised.value.reason is ContextUnavailableReason.CREDENTIALS
    assert vault.calls == []
