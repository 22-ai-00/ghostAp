"""Integration test: graceful_shutdown with multiple active CardSessions."""
import gc
import threading
import weakref

from src.card.delivery.engine import CardDelivery
from src.card.delivery.registry import DeliveryRegistry, delivery_registry
from src.card.events import CardEvent
from src.card.session import CardSession
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.models import CardMetadata


class _SlowClient:
    """Mock client that records calls."""

    def __init__(self):
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        idx = len(self.created) + 1
        self.created.append(card_json)
        return (f"msg_{idx}", f"card_{idx}")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.updated.append(card_json)


def _make_session(chat_id: str, client: _SlowClient) -> CardSession:
    delivery = CardDelivery(client)
    delivery_registry.register(delivery)
    meta = CardMetadata(engine_type="deep", mode_name="Test")
    config = SessionConfig(metadata=meta, sync_delivery=True)
    return CardSession(
        chat_id=chat_id,
        config=config,
        delivery=delivery,
        callbacks=SessionCallbacks(notify_callback=lambda _c, _t: None),
    )


class TestMultiSessionGracefulShutdown:
    def test_registry_does_not_keep_unowned_delivery_alive(self):
        client = _SlowClient()
        delivery = CardDelivery(client)
        delivery_ref = weakref.ref(delivery)
        # Stop the owned worker explicitly so this WeakSet-only assertion does
        # not leave a daemon thread behind in the test process.
        delivery._lock_pool.shutdown()

        del delivery
        gc.collect()

        assert delivery_ref() is None

    def test_register_after_shutdown_is_closed_by_next_shutdown_wave(self):
        first = CardDelivery(_SlowClient())
        assert delivery_registry.shutdown_all() is True
        assert first._shutdown_complete is True

        second = CardDelivery(_SlowClient())
        assert second._shutdown_complete is False
        assert delivery_registry.shutdown_all() is True
        assert second._shutdown_complete is True

    def test_instance_registered_during_shutdown_requires_next_wave(self):
        registry = DeliveryRegistry()
        registry.install_atexit = lambda: None

        class _DeliveryStub:
            def __init__(self, on_shutdown=None):
                self.shutdown_calls = 0
                self._on_shutdown = on_shutdown

            def shutdown(self, timeout=5.0):
                self.shutdown_calls += 1
                registry.unregister(self)
                if self._on_shutdown is not None:
                    self._on_shutdown()
                return True

        late = _DeliveryStub()
        first = _DeliveryStub(on_shutdown=lambda: registry.register(late))
        registry.register(first)

        assert registry.shutdown_all() is False
        assert late.shutdown_calls == 0
        assert registry.shutdown_all() is True
        assert late.shutdown_calls == 1

    def test_graceful_shutdown_drains_all_sessions(self):
        """N=3 active sessions dispatching → graceful_shutdown → all closed."""
        client = _SlowClient()
        sessions = [_make_session(f"chat_{i}", client) for i in range(3)]

        # Dispatch STARTED to each — makes them active
        for s in sessions:
            s.dispatch(CardEvent.started())

        # Verify sessions are active
        for s in sessions:
            assert not s.closed

        # Now trigger shutdown sequence (simulating graceful_shutdown internals)
        # We can't call the real graceful_shutdown since it sets a global flag,
        # but we test the relevant delivery shutdown path:
        delivery_registry.drain_in_flight(timeout=5.0)
        delivery_registry.shutdown_all()

        # After shutdown_all, sessions should still have their closed state
        # (shutdown_all stops deliveries, doesn't close sessions).
        # Close sessions explicitly (as ws_client would):
        for s in sessions:
            s.close()

        # Verify all sessions are now closed
        for s in sessions:
            assert s.closed

    def test_concurrent_dispatch_during_shutdown(self):
        """Sessions with in-flight dispatches should drain cleanly on shutdown."""
        client = _SlowClient()
        sessions = [_make_session(f"chat_{i}", client) for i in range(3)]

        # Start sessions
        for s in sessions:
            s.dispatch(CardEvent.started())

        # Dispatch text concurrently
        threads = []
        for i, s in enumerate(sessions):
            t = threading.Thread(target=s.dispatch, args=(CardEvent.text_delta(f"b{i}", "hello"),))
            t.start()
            threads.append(t)

        # Wait for all dispatches
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Dispatch thread did not complete in time"

        # Drain and shutdown
        delivery_registry.drain_in_flight(timeout=5.0)
        delivery_registry.shutdown_all()

        for s in sessions:
            s.close()
            assert s.closed

        # Verify that all sessions delivered something
        assert len(client.created) == 3  # 3 STARTED events created cards
        assert len(client.updated) >= 3  # At least 3 text_delta updates
