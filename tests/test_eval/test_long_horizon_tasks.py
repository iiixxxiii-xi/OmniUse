"""Long-horizon tasks: multi-page flows requiring 20-40+ agent steps.

These tasks are the endurance proof of a computer-use agent — a genuine page
navigation + cross-page state passing + conditional branching, not a 3-step
stub. Each task is graded on its *final* state (an order confirmation, a
fully-populated summary, a completed settings panel), never a single click.

The tests lock three properties in:

* the task set contains 5-8 long tasks, each hard, each self-contained, each
  budgeted >= 40 steps;
* the evaluator's expected values appear ONLY in the final state (they are
  absent from the freshly-served page, so a lazy ``done`` cannot pass);
* each task is actually solvable — a scripted solver drives every required
  action through a reachable selector and lands on score 1.0 (proving every
  step's action target exists and the evaluator's expected values match reality).
"""

from pathlib import Path

import pytest

from minicua.eval.evaluator import evaluate
from minicua.eval.task import load_tasks
from minicua.perception.extract import extract_state

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
_FIXTURE_URL = "http://minicua.local/"

#: Minimum number of real actions a long task must demand (perceive cycles are
#: on top of this — each page read is an extra agent step).
MIN_ACTIONS = 18


def _long_tasks():
    return [t for t in load_tasks(TASKS_DIR) if t.max_steps >= 40]


def _expected_substrings(task):
    out = []
    for expected in task.evaluator.expecteds:
        v = expected.get("expected")
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        elif isinstance(v, str):
            out.append(v)
    return out


async def _serve(session, task):
    html = task.html

    async def handler(route):
        await route.fulfill(status=200, content_type="text/html", body=html)

    await session.context.route(_FIXTURE_URL + "**", handler)
    await session.page.goto(_FIXTURE_URL)


# --------------------------------------------------------------------------- #
# Per-task solver recipes: (op, *args) where op in fill/click/check/uncheck/dd.
# `dd` clicks the custom dropdown trigger then the named option.
# --------------------------------------------------------------------------- #
RECIPES = {
    "account_setup_wizard": [
        ("dd", "title", "Ms"),
        ("fill", "#fullname", "Alice Johnson"),
        ("fill", "#email", "alice@example.com"),
        ("fill", "#username", "alicej"),
        ("fill", "#password", "S3cure!Pass"),
        ("fill", "#confirm", "S3cure!Pass"),
        ("click", "#page1 button:has-text('Next')"),
        ("fill", "#dob", "1992-08-15"),
        ("check", "input[name=gender][value=Female]"),
        ("dd", "country", "Canada"),
        ("fill", "#phone", "+1-555-0100"),
        ("check", "#lang_en"),
        ("check", "#lang_es"),
        ("fill", "#bio", "Designer based in Toronto"),
        ("click", "#page2 button:has-text('Next')"),
        ("fill", "#street", "42 Maple Ave"),
        ("fill", "#apt", "Unit 3"),
        ("fill", "#city", "Toronto"),
        ("dd", "province", "Ontario"),
        ("fill", "#postal", "M5V 2T6"),
        ("fill", "#website", "alice.dev"),
        ("click", "#page3 button:has-text('Next')"),
        ("check", "#int_tech"),
        ("check", "#int_music"),
        ("check", "#int_travel"),
        ("check", "#newsletter"),
        ("check", "input[name=plan][value=Pro]"),
        ("dd", "theme", "Dark"),
        ("dd", "notifications", "Email"),
        ("dd", "timezone", "UTC-5"),
        ("click", "#page4 button:has-text('Next')"),
        ("click", "#page5 button:has-text('Create Account')"),
    ],
    "ecommerce_checkout_flow": [
        ("fill", "#query", "laptop"),
        ("click", "button:has-text('Search')"),
        ("click", "#catalog div:has-text('Laptop Pro') button:has-text('Add')"),
        ("click", "button:has-text('Accessories')"),
        ("click", "#catalog div:has-text('Wireless Mouse') button:has-text('Add')"),
        ("click", "#catalog div:has-text('USB Hub') button:has-text('Add')"),
        ("click", "button:has-text('View Cart')"),
        ("click", "button:has-text('Proceed to Shipping')"),
        ("fill", "#ship_name", "Jordan Lee"),
        ("fill", "#ship_address", "100 Granville St"),
        ("fill", "#ship_city", "Vancouver"),
        ("dd", "ship_country", "Canada"),
        ("fill", "#ship_postal", "V6Z 1L3"),
        ("fill", "#ship_phone", "+1-604-555-0100"),
        ("check", "input[name=ship_method][value=Express]"),
        ("click", "button:has-text('Next: Payment')"),
        ("fill", "#card", "4111111111111111"),
        ("fill", "#expiry", "12/27"),
        ("fill", "#cvv", "123"),
        ("fill", "#card_name", "Jordan Lee"),
        ("uncheck", "#bill_same"),
        ("fill", "#bill_name", "Dana Lee"),
        ("fill", "#bill_address", "5 Water St"),
        ("fill", "#bill_city", "Vancouver"),
        ("click", "button:has-text('Place Order')"),
    ],
    "cross_page_data_collection": [
        ("click", "#dpage1 button:has-text('Next')"),
        ("click", "#dpage2 button:has-text('Next')"),
        ("click", "#dpage3 button:has-text('Next')"),
        ("click", "#dpage4 button:has-text('Next')"),
        ("click", "#dpage5 button:has-text('Next')"),
        ("click", "#dpage6 button:has-text('Go to summary')"),
        ("fill", "#f_p_users", "847"),
        ("fill", "#f_p_revenue", "$12,450"),
        ("fill", "#f_p_orders", "312"),
        ("fill", "#f_p_nps", "42"),
        ("fill", "#f_p_uptime", "99.98%"),
        ("fill", "#f_p_churn", "1.4%"),
        ("fill", "#f_p_sessions", "5,210"),
        ("fill", "#f_p_rating", "4.8"),
        ("fill", "#f_p_tickets", "73"),
        ("fill", "#f_p_refunds", "0.9%"),
        ("fill", "#f_p_views", "98,120"),
        ("fill", "#f_p_signups", "1,240"),
        ("click", "button:has-text('Submit')"),
    ],
    "settings_panel_wizard": [
        ("check", "#sw_email_alerts"),
        ("check", "#sw_push_alerts"),
        ("check", "#sw_sms_alerts"),
        ("check", "#sw_desktop_alerts"),
        ("dd", "freq", "Daily"),
        ("click", "#page1 button:has-text('Next')"),
        ("check", "#sw_profile_public"),
        ("check", "#sw_share_analytics"),
        ("check", "#sw_personalized_ads"),
        ("dd", "visibility", "Friends"),
        ("click", "#page2 button:has-text('Next')"),
        ("check", "#sw_dark_mode"),
        ("check", "#sw_autosave"),
        ("check", "#sw_beta_features"),
        ("check", "#sw_two_factor"),
        ("dd", "language", "French"),
        ("click", "#page3 button:has-text('Save Settings')"),
    ],
    "conditional_registration": [
        ("fill", "#c_name", "Morgan Chen"),
        ("fill", "#c_email", "morgan@example.com"),
        ("fill", "#c_password", "pass123"),
        ("fill", "#c_phone", "+1-555-0199"),
        ("check", "input[name=acct][value=Business]"),
        ("fill", "#c_company", "Acme Labs"),
        ("fill", "#c_vat", "CA123456"),
        ("dd", "c_size", "51-200"),
        ("click", "#page1 button:has-text('Next')"),
        ("fill", "#d_address", "1 Main St"),
        ("fill", "#d_city", "Toronto"),
        ("check", "#c_diff"),
        ("fill", "#d_alt_name", "Riley Chen"),
        ("fill", "#d_alt_address", "8 Pine Rd"),
        ("fill", "#d_alt_city", "Seattle"),
        ("check", "#c_gift"),
        ("fill", "#d_giftmsg", "Happy birthday!"),
        ("click", "#page2 button:has-text('Complete')"),
    ],
    "project_setup_crud": [
        ("fill", "#p_name", "Atlas Dashboard"),
        ("fill", "#p_desc", "Internal metrics"),
        ("dd", "p_type", "Web"),
        ("fill", "#p_start", "2024-03-01"),
        ("fill", "#p_end", "2024-09-30"),
        ("click", "#page1 button:has-text('Next')"),
        ("fill", "#m1_name", "Sam Wu"),
        ("fill", "#m1_email", "sam@example.com"),
        ("dd", "m1_role", "Admin"),
        ("click", "#add1"),
        ("fill", "#m2_name", "Priya Nair"),
        ("fill", "#m2_email", "priya@example.com"),
        ("dd", "m2_role", "Editor"),
        ("click", "#add2"),
        ("click", "#page2 button:has-text('Next')"),
        ("check", "#i_github"),
        ("check", "#i_slack"),
        ("check", "#i_drive"),
        ("click", "#page3 button:has-text('Create Project')"),
    ],
}


async def _run_recipe(page, recipe):
    for op in recipe:
        kind = op[0]
        if kind == "fill":
            await page.fill(op[1], op[2])
        elif kind == "click":
            await page.click(op[1])
        elif kind == "check":
            await page.check(op[1])
        elif kind == "uncheck":
            await page.uncheck(op[1])
        elif kind == "dd":
            await page.click(f"#dd-{op[1]}")
            await page.click(f"#menu-{op[1]} .opt[data-v='{op[2]}']")
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unknown recipe op {kind!r}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_long_horizon_tasks_present_and_budgeted():
    tasks = _long_tasks()
    assert 5 <= len(tasks) <= 8, f"expected 5-8 long tasks, got {len(tasks)}"
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    for t in tasks:
        assert t.difficulty == "hard", f"{t.id}: long tasks must be hard"
        assert t.max_steps >= 40, f"{t.id}: long tasks must budget >= 40 steps"
        assert t.html is not None, f"{t.id}: long tasks must be self-contained"


def test_long_horizon_recipes_are_long():
    tasks = {t.id: t for t in _long_tasks()}
    assert set(RECIPES) == set(tasks), "a solver recipe must exist for every long task"
    for tid, recipe in RECIPES.items():
        # a `dd` op is two real actions (open the dropdown + pick the option)
        actions = len(recipe) + sum(1 for op in recipe if op[0] == "dd")
        assert actions >= MIN_ACTIONS, f"{tid}: only {actions} actions, want >= {MIN_ACTIONS}"


@pytest.mark.asyncio
async def test_long_horizon_answer_only_in_final_state(session):
    """The final answer must not be present before any work is done."""
    for task in _long_tasks():
        substrings = _expected_substrings(task)
        assert substrings, f"{task.id}: long task must grade on string substrings"
        await _serve(session, task)
        try:
            state = await extract_state(session.page, use_vision="dom_only")
            inner = await session.page.evaluate("() => document.body ? document.body.innerText : ''")
            score = await evaluate(session, task.evaluator)
        finally:
            await session.context.unroute(_FIXTURE_URL + "**")

        haystack = (state.dom_text or "") + "\n" + (inner or "")
        for s in substrings:
            assert s not in haystack, f"{task.id}: answer {s!r} leaks into the initial page"
        assert score == 0.0, f"{task.id}: fresh page should not already satisfy the evaluator"


@pytest.mark.asyncio
async def test_long_horizon_solvable_end_to_end(session):
    """Every required action has a reachable selector and leads to a passing evaluator."""
    for task in _long_tasks():
        recipe = RECIPES[task.id]
        await _serve(session, task)
        try:
            await _run_recipe(session.page, recipe)
            score = await evaluate(session, task.evaluator)
        finally:
            await session.context.unroute(_FIXTURE_URL + "**")
        assert score == 1.0, f"{task.id}: solver completed the flow but evaluator scored {score!r}"
