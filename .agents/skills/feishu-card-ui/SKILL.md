---
name: feishu-card-ui
description: >-
  Feishu Card JSON 2.0 authoring, interactive callbacks, strict src/card/ layer boundaries,
  and visual ux/*.html screenshot inspection. Load when modifying cards, COT streams, or reports.
---

# Feishu Card & UI Engineering

## Architecture & Import Boundary

Enforce strict one-way imports across `src/card/` (see [adr-card-pipeline-boundary.md](../../../docs/adr-card-pipeline-boundary.md)):

```text
handler -> session -> render / delivery
```

- `src/card/render/` and `src/card/builders/` **never** import `src/card/delivery`, `src/card/session`, or `src/feishu`.
- `src/card/delivery/` **never** imports `src/card/session`.
- Cross-layer types live in [src/card/protocols.py](../../../src/card/protocols.py) or `src/card/events/`.
- Static cards use `CoreBuilder` / `build_info_card()`; streaming engine & task progress cards use `CardSession` (`CardBuilder.build_engine_card()` is removed).

## Card JSON 2.0 & Callback Contracts

- Build Card JSON 2.0 (`schema: "2.0"`, `config.update_multi: True`), staying within [src/card/thresholds.py](../../../src/card/thresholds.py) (`<= 180` tagged nodes, `<= 27KB`, `summary.content` 8–60 chars).
- Attach `"behaviors": [{"type": "callback", "value": payload}]` alongside `"value": payload` on every interactive `button` and `select_static`.
- Ordinary programming tasks use **one main card** (overall task list, active task, subagent summary; subagent text/tools/images stream inside it). Open a continuation card only when reaching Feishu capacity limits, and omit empty tool/detail blocks.
- In [ws_card_pipeline.py](../../../src/feishu/ws_card_pipeline.py), `select_static` picks arrive as `value["_option"]`. Register new actions in [dispatch.py](../../../src/card/actions/dispatch.py) and [action_registry.py](../../../src/feishu/action_registry.py).
- Fix any Feishu schema errors reported in `logs.log` in the builder/renderer with a regression test.

## Visual Verification (Before Completing UI Changes)

1. Create or update an HTML preview under `ux/` before aligning production card/report builders.
2. Render the preview or output HTML to an image (e.g. headless browser PNG screenshot) and **inspect the image directly** for clipped text, awkward badge wrapping, unrendered Markdown/escaping inside code spans, and dark/light contrast.
