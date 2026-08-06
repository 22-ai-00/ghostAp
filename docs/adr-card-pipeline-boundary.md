# ADR: Card Pipeline Boundary — Builder vs Session

- **Status**: Accepted
- **Date**: 2026-05-05
- **Context**: Card refactoring (DirectCardSession → CardSession migration)

## Decision

Card rendering is split into two distinct subsystems with non-overlapping responsibilities:

### CardBuilder (Static One-Shot Cards)

**Responsibility**: Build and deliver a single immutable card payload. No lifecycle, no streaming, no state management.

**Use cases**:
- Help/menu cards (`build_info_card`)
- Lock notification cards
- Diagnostic cards
- Error/status reply cards
- Project overview cards

**Freeze policy**: No new features. Bug fixes only. Any new card type that requires interaction or streaming MUST use CardSession.

### CardSession Pipeline (Streaming Lifecycle Cards)

**Responsibility**: Manage the full lifecycle of an engine execution card via dispatch→reduce→render→deliver.

**Use cases**:
- Deep engine execution cards
- Spec engine structured iteration cards
- Workflow parallel orchestration cards
- Any future engine with real-time progress or interactive buttons

**Characteristics**:
- Event-driven state management via `CardEvent` + reducer
- Thread-safe dispatch with `threading.Lock`
- TTL/idle timeout with pre-warning
- Button interaction routing via `ActionRouter`
- Automatic card rotation for long sessions
- Delivery retry with exponential backoff

## Consequences

1. `CardBuilder.build_engine_card()` has been removed. Engine cards are exclusively driven by CardSession.
2. `CardBuilder.build_info_card()` and other static builders remain frozen — maintenance-only.
3. New interactive card types MUST use the CardSession pipeline.
4. Static builders do NOT gain streaming, state management, or button routing capabilities.

## Related

- [CHANGELOG](../CHANGELOG.md)
