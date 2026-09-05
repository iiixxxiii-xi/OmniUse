# minicua

A long-horizon, browser-first computer-use agent. It drives a real Chromium
browser (via Playwright), perceives the page as linearized DOM, grounds the model's
tool calls back to page elements, and recovers from stale elements / page changes /
loops / crashes — then scores its own runs with a declarative evaluator.

**Architecture (one sentence):** a layered pipeline of *perception → action/grounding →
controller loop → recovery → state → eval*, wrapped by a persistent browser session,
with Playwright as the primary action space, DOM linearization as the primary
perception, and a declarative evaluator (getter → metric → conj) as the judge.

**Two action spaces.** The `browser/` pipeline drives a real Chromium via
Playwright. A parallel `desktop/` pipeline drives a local desktop or an
SSH-connected VM (`pyautogui` + screenshots over a persistent `paramiko` channel)
for OSWorld-style GUI tasks — the same agent loop and recovery machinery run
against both with no other changes.

## Install & test

```bash
# Python 3.12 + uv
uv sync
export PLAYWRIGHT_BROWSERS_PATH=/d/playwright-browsers   # Windows (D: drive); see tests/conftest.py
uv run pytest -v                                        # full suite (TDD, fake model + inline HTML fixtures)
```

## Run

```bash
# one task (FakeModel by default — no API key needed)
uv run minicua run tasks/click_button.json --script script.json

# a whole task set -> report.md / report.csv / results.json
uv run minicua eval tasks/ --output out/

# re-render a report from a saved results.json (no browser)
uv run minicua report out/results.json --output out/
```

`--script` is a JSON list of scripted model responses, e.g.
`[{"name": "click", "params": {"index": 1}}, {"name": "done", "params": {"success": true}}]`.

## Task format (declarative — new tasks are pure JSON)

```json
{
  "id": "click_button",
  "instruction": "Click the 'Click me' button.",
  "html": "<button id=btn onclick=\"document.getElementById('out').textContent='clicked'\">Click me</button><div id=out></div>",
  "evaluator": {
    "func": "exact_match",
    "result": { "getter": "element_text", "selector": "#out" },
    "expected": { "expected": "clicked" }
  }
}
```

* **getter** (`result`) — reads final browser state (`page_url`, `page_text`,
  `element_exists`, `element_text`, `element_attribute`, `element_count`,
  `cookie_exists`, `local_storage`, `screenshot`, `page_title`).
* **metric** (`func`) — compares the getter's value against `expected`
  (`exact_match`, `contains`, `regex_match`, `count_eq`,
  `element_exists_metric`, `match_in_list`, `is_in_list`).
* **conj** — `"and"` (every check must pass) or `"or"` (any check passes).

## Six run metrics (from the event log)

`task_success`, `avg_tool_calls`, `token_cost`, `latency`, `recovery_rate`,
`invalid_action_rate` — each extracted from the typed event log and guarded
against division by zero (see `src/minicua/eval/metrics_aggregate.py`).

## Package layout

```
src/minicua/
  browser/     persistent Playwright session + crash watchdog
  perception/  DOM linearization, selector map, screenshots
  action/      action models, grounding (index -> locator), executor, registry
  controller/  agent loop, budget, retry, ChatModel + FakeModel
  desktop/     desktop action space: local + SSH-driven VM (OSWorld-style)
  recovery/    stale relocalize, page-change, loop, crash rebuild
  state/       append-only event log, checkpoint, trajectory (JSONL)
  eval/        getters, metrics, evaluator, six-metric aggregate, runner, report
  cli/         run / eval / report commands
```
