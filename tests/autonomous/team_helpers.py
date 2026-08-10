from __future__ import annotations

from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.team.service import TeamAttemptResult, TeamTarget
from tests.autonomous.workforce_helpers import make_writer

TEAM_KEY = b"t" * 32


def make_team_storage(tmp_path):
    writer = make_writer(tmp_path)
    blobs = BlobStore(
        tmp_path / "team-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: TEAM_KEY),
    )
    return writer, blobs


class ImmediateTeamBackend:
    def __init__(self) -> None:
        self.targets = (
            TeamTarget(
                "agt_coder",
                "Coder",
                "coder",
                ("python", "implementation"),
                "ready_warm",
                2,
            ),
            TeamTarget(
                "agt_reviewer",
                "Reviewer",
                "reviewer",
                ("review", "security"),
                "ready_cold",
                0,
            ),
        )
        self.submissions = []
        self.notifications = []
        self.results = {}

    def list_active(self, tenant_key, chat_id):
        assert tenant_key == "tenant_1"
        assert chat_id == "oc_team"
        return self.targets

    def submit(
        self,
        *,
        run_id,
        step_id,
        target,
        tenant_key,
        chat_id,
        message_id,
        requester_principal_id,
        instruction,
        deadline_at,
    ):
        acceptance_id = f"acc_{len(self.submissions)}"
        self.submissions.append(
            (
                step_id,
                target.agent_id,
                instruction,
                {
                    "run_id": run_id,
                    "tenant_key": tenant_key,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "requester_principal_id": requester_principal_id,
                    "deadline_at": deadline_at,
                },
            )
        )
        self.results[acceptance_id] = TeamAttemptResult(
            "completed",
            output=f"deliverable by {target.agent_id} for {step_id}",
            history_record_id=f"hist_{len(self.submissions)}",
        )
        return acceptance_id

    def result(self, acceptance_id):
        return self.results.get(acceptance_id)

    def cancel(self, acceptance_id, *, run_id, step_id):
        del run_id, step_id
        return TeamAttemptResult("canceled", error_code="canceled")

    def notify(
        self,
        message_id,
        chat_id,
        result,
        *,
        idempotency_key="",
        tenant_key="",
        requester_principal_id="",
    ):
        del idempotency_key, tenant_key, requester_principal_id
        self.notifications.append((message_id, chat_id, result))
