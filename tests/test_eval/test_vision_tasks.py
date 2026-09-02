"""Vision-required tasks: the answer lives only in the screenshot, never in DOM text.

These tasks are the multimodal stress test. A DOM-only text model (DeepSeek)
sees the linearized DOM — and for every task here, the linearized DOM is
*identical* for all candidate options (color, icon, shape, and real-photo tasks
alike), so no text model can pick the answer; only a vision model (qwen3-vl),
which also reads the screenshot, can. The tests in this module lock that
property in:

* every vision task exists, loads, and carries a self-contained fixture;
* the evaluator's expected answer never appears in ``dom_text`` or ``innerText``;
* two-button color tasks expose options whose accessible names are identical and
  whose *only* difference is CSS;
* SVG-based fixtures contain no ``<text>`` element (shapes/colors are pixel-only);
* real-object tasks embed real ``data:image/jpeg;base64`` photos with no ``alt``.
"""

import re
from pathlib import Path

import pytest

from minicua.controller.llm import FakeModel
from minicua.eval.evaluator import evaluate
from minicua.eval.runner import run_task
from minicua.eval.task import load_tasks
from minicua.perception.extract import extract_state

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
_FIXTURE_URL = "http://minicua.local/"

#: ids whose discriminating signal is *button background color* (same label).
COLOR_TASKS = {"click_red_button", "click_yellow_button"}
#: ids whose discriminating signal is a *real JPEG photograph*.
PHOTO_TASKS = {
    "click_cat_photo",
    "click_dog_photo",
    "click_bicycle_photo",
    "click_car_photo",
    "click_bird_photo",
}


def _load():
    return load_tasks(TASKS_DIR)


def _vision_tasks():
    return [t for t in _load() if t.vision_required]


def _answer_strings(task):
    """The string-typed expected value(s) a vision task is graded on."""
    out = []
    for expected in task.evaluator.expecteds:
        v = expected.get("expected")
        if isinstance(v, str):
            out.append(v)
    return out


async def _serve(session, task):
    html = task.html

    async def handler(route):
        await route.fulfill(status=200, content_type="text/html", body=html)

    await session.context.route(_FIXTURE_URL + "**", handler)
    await session.page.goto(_FIXTURE_URL)


def test_vision_required_tasks_present_and_load():
    tasks = _vision_tasks()
    assert len(tasks) >= 8, "expected a meaningful set of vision-required tasks"
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "vision task ids must be unique"
    for t in tasks:
        assert t.instruction
        assert t.html is not None, f"{t.id}: vision tasks must carry a self-contained fixture"


@pytest.mark.asyncio
async def test_vision_required_answer_not_in_dom_text(session):
    """A text-only model cannot read the answer: it never appears in the DOM."""
    for task in _vision_tasks():
        answers = _answer_strings(task)
        assert answers, f"{task.id}: vision task must be graded on a string answer"
        await _serve(session, task)
        try:
            state = await extract_state(session.page, use_vision="dom_only")
            inner = await session.page.evaluate(
                "() => document.body ? document.body.innerText : ''"
            )
        finally:
            await session.context.unroute(_FIXTURE_URL + "**")

        haystack = (state.dom_text or "") + "\n" + (inner or "")
        for a in answers:
            assert a not in haystack, (
                f"{task.id}: answer {a!r} leaks into DOM text / innerText"
            )


@pytest.mark.asyncio
async def test_vision_color_options_indistinguishable_by_dom(session):
    """Color-discrimination options share one accessible name; only CSS differs."""
    for task in _vision_tasks():
        if task.id not in COLOR_TASKS:
            continue
        await _serve(session, task)
        try:
            data = await session.page.evaluate(
                """() => Array.from(document.querySelectorAll('button')).map(b => ({
                    text: (b.innerText || '').trim(),
                    bg: getComputedStyle(b).backgroundColor,
                }))"""
            )
        finally:
            await session.context.unroute(_FIXTURE_URL + "**")

        texts = [d["text"] for d in data]
        bgs = [d["bg"] for d in data]
        assert len(set(texts)) == 1, (
            f"{task.id}: option buttons have distinct text {texts!r} — a text model could tell them apart"
        )
        assert len(set(bgs)) == len(bgs), (
            f"{task.id}: option buttons do not differ by background color"
        )


def test_vision_svg_fixtures_have_no_text_labels():
    """SVG geometry (shape/icon/color+shape/match) must not carry <text> labels."""
    for task in _vision_tasks():
        if "<svg" in task.html:
            assert "<text" not in task.html, (
                f"{task.id}: SVG must not contain a <text> element (label would leak the answer)"
            )


def test_vision_photo_tasks_embed_real_base64_jpegs_without_labels():
    """Real-object tasks use real JPEG photos (base64) with no alt text."""
    for task in _vision_tasks():
        if task.id not in PHOTO_TASKS:
            continue
        assert "data:image/jpeg;base64," in task.html, (
            f"{task.id}: must embed real JPEG photos, not SVG shapes"
        )
        assert "<svg" not in task.html, (
            f"{task.id}: real-object task must not substitute SVG shapes for photos"
        )
        imgs = re.findall(r"<img[^>]*>", task.html)
        assert len(imgs) >= 4, f"{task.id}: expected several photos side by side"
        for tag in imgs:
            alt = re.search(r"alt=['\"]([^'\"]*)['\"]", tag)
            assert alt is None or alt.group(1) == "", (
                f"{task.id}: <img> must carry no alt label (alt would leak the object name)"
            )


@pytest.mark.asyncio
async def test_solve_click_red_button_end_to_end(session):
    # A scripted model that "sees" which button is red (index 1) solves the task.
    task = next(t for t in _load() if t.id == "click_red_button")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_vision_click_red_button_wrong_choice_scores_zero(session):
    # Clicking the blue button (index 2) instead must NOT pass the evaluator.
    task = next(t for t in _load() if t.id == "click_red_button")
    await _serve(session, task)
    try:
        await session.page.locator("button").nth(1).click()
        assert await evaluate(session, task.evaluator) == 0.0
    finally:
        await session.context.unroute(_FIXTURE_URL + "**")
