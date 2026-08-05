from src.card.render.live_ticker import (
    DEFAULT_TICKER_FRAMES,
    DEFAULT_TICKER_INTERVAL,
    FROZEN_FRAME,
    LiveTicker,
    frame_for_tick,
)


class FakeHandle:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False


class FakeScheduler:
    def __init__(self):
        self.handles = []

    def schedule(self, delay, callback, *, session_id=""):
        handle = FakeHandle(callback)
        handle.delay = delay
        handle.session_id = session_id
        self.handles.append(handle)
        return handle

    def cancel(self, handle):
        handle.cancelled = True

    def fire_next(self):
        handle = self.handles.pop(0)
        if not handle.cancelled:
            handle.callback()
        return handle


def test_frame_for_tick_cycles_frames():
    assert frame_for_tick(0, ("a", "b")) == "a"
    assert frame_for_tick(1, ("a", "b")) == "b"
    assert frame_for_tick(2, ("a", "b")) == "a"


def test_default_ticker_contract():
    assert DEFAULT_TICKER_FRAMES == ("🟢", "⚪")
    assert DEFAULT_TICKER_INTERVAL == 1.2
    assert FROZEN_FRAME == "⏸"
    assert LiveTicker(session_id="test", on_frame=lambda f: None).interval == 1.2


def test_live_ticker_emits_now_and_reschedules_frames():
    scheduler = FakeScheduler()
    frames = []
    ticker = LiveTicker(
        session_id="sess_1",
        on_frame=frames.append,
        interval=0.25,
        frames=("a", "b"),
        scheduler=scheduler,
    )

    ticker.start()
    scheduler.fire_next()
    scheduler.fire_next()

    assert frames == ["a", "b", "a"]
    assert ticker.running is True
    assert scheduler.handles[-1].session_id == "sess_1"
    assert scheduler.handles[-1].delay == 0.25


def test_ticker_accepts_custom_interval():
    """LiveTicker should accept a custom interval and use it for scheduling."""
    scheduler = FakeScheduler()
    ticker = LiveTicker(
        session_id="test",
        on_frame=lambda f: None,
        interval=2.5,
        scheduler=scheduler,
    )

    ticker.start()
    assert ticker.interval == 2.5
    assert scheduler.handles[-1].delay == 2.5


def test_live_ticker_stop_cancels_pending_handle():
    scheduler = FakeScheduler()
    frames = []
    ticker = LiveTicker(
        session_id="sess_1",
        on_frame=frames.append,
        frames=("a",),
        scheduler=scheduler,
    )

    ticker.start(emit_now=False)
    pending = scheduler.handles[-1]
    ticker.stop()
    scheduler.fire_next()

    assert pending.cancelled is True
    assert frames == []
    assert ticker.running is False
