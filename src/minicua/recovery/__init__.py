"""Recovery layer: graded degradation from transient action failures back to progress.

Four recovery strategies, each a rung on a ladder that prefers cheap, local fixes
over expensive re-planning:

* **stale element** — re-ground an element whose index moved between perceptions.
* **page change** — detect that the page moved under the agent and abort a stale
  multi-action queue.
* **loop detection** — softly nudge the model when it repeats itself or the page
  stagnates.
* **crash recovery** — rebuild a crashed browser session and restore task state.
"""
