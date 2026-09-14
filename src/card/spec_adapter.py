"""Spec's single task card with capacity-driven history continuation."""

from .events import CardEvent
from .programming_adapter import ProgrammingCardSession
from .render.budget import RenderBudget
from .render.renderer import render_card
from .state.models import CardState
from .state.reducer import reduce_card_state


class SpecCardSession(ProgrammingCardSession):
    def __init__(self, session, *, budget: RenderBudget, **kwargs):
        super().__init__(session, **kwargs)
        self._budget = budget

    @property
    def current(self):
        return self._rotator.current

    @property
    def session_id(self) -> str:
        return self._rotator.session_id

    def _requires_capacity_rotation(self, state: CardState, event: CardEvent) -> bool:
        projected = reduce_card_state(state, event, retain_all_blocks=True)
        return len(render_card(projected, self._budget)) > 1

    def _rotate_for_capacity(self, old_session, crossing_event: CardEvent) -> bool:
        state = old_session.state
        if not super()._rotate_for_capacity(old_session, crossing_event):
            return False
        if state is None or state.engine_ext is None:
            return True
        extension = state.engine_ext
        self.current.dispatch(CardEvent.cycle_started(extension.cycle_num, extension.max_cycles))
        if extension.criteria_section:
            self.current.dispatch(CardEvent.criteria_updated(
                extension.criteria_section,
                satisfied_count=extension.criteria_satisfied,
                total_count=extension.criteria_total,
            ))
        for block in state.blocks:
            if block.kind == "phase" and block.status == "active" and block.cycle_num == extension.cycle_num:
                self.current.dispatch(CardEvent.phase_started(
                    block.cycle_num, block.phase_name,
                    subtitle=state.header.subtitle, content=block.content,
                ))
        return True
