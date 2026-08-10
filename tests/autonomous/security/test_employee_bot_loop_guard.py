from __future__ import annotations

from src.autonomous.team.coordinator import TeamCoordinatorActor
from src.autonomous.team.models import TeamRunPhase
from tests.autonomous.team_helpers import ImmediateTeamBackend, make_team_storage


def test_coordinator_rejects_duplicate_fake_wrong_member_and_terminal_events(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = ImmediateTeamBackend()
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
    )
    original_phase = actor._phase  # noqa: SLF001

    def stop_before_review(run, phase, **kwargs):
        if phase is TeamRunPhase.REVIEWING:
            raise SystemExit
        return original_phase(run, phase, **kwargs)

    actor._phase = stop_before_review  # type: ignore[method-assign] # noqa: SLF001
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_loop",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="Python implementation",
    )
    actor.drain()
    assignment_id = f"{run.run_id}:assignment:1"
    coordinates = dict(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_coder",
        team_run_id=run.run_id,
        assignment_id=assignment_id,
        causal_event_id="cause_once",
    )
    assert actor.record_collaboration_event(**coordinates) is True
    assert actor.record_collaboration_event(**coordinates) is False
    assert actor.record_collaboration_event(
        **{**coordinates, "causal_event_id": "cause_wrong", "agent_id": "agt_reviewer"}
    ) is False
    assert actor.record_collaboration_event(
        **{**coordinates, "causal_event_id": "cause_fake", "assignment_id": "fake"}
    ) is False
    assert actor.record_collaboration_event(
        **{**coordinates, "causal_event_id": "cause_tenant", "tenant_key": "tenant_2"}
    ) is False
    actor._phase = original_phase  # type: ignore[method-assign] # noqa: SLF001
    actor.recover()
    actor.drain()
    assert actor.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert actor.record_collaboration_event(
        **{**coordinates, "causal_event_id": "cause_terminal"}
    ) is False
    actor.close()
    blobs.close()
    writer.close()
