# Computer-Use Agent (CUA) 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从零构建一个 browser-first 的 computer-use agent（`minicua` 包），覆盖 persistent session → perception → action/grounding → controller loop → recovery → state → eval → cli，全部 TDD。

**Architecture:** 分层六组件（Perception / Action / Controller / Recovery / State / Eval）+ Browser 封装，Playwright 为第一动作空间，DOM 线性化为第一感知，声明式 evaluator 判成败，六指标从 event log 提取。

**Tech Stack:** Python 3.12 · Playwright (async) · pydantic v2 · anthropic/openai SDK · typer · pytest + pytest-asyncio。

**前置说明：**
- 所有任务遵循 RED-GREEN-REFACTOR：先写失败测试 → 跑红 → 最小实现 → 跑绿 → 提交。
- 每条命令在仓库根目录 `D:\minicua` 下运行。
- LLM 相关测试一律用 **fake/stub model**（不真调 API），浏览器相关测试用 Playwright 的 `page.goto(data:...)` 内联 HTML fixture，避免依赖外部网站。
- 阶段间有依赖，必须按顺序推进（`browser` → `perception` → `action` → `controller` → `recovery` → `state` → `eval` → `cli`）。

---

## Stage 0: 脚手架

### Task 0.1: 初始化包结构

**Files:**
- Create: `D:\minicua\pyproject.toml`
- Create: `D:\minicua\src\minicua\__init__.py`
- Create: `D:\minicua\src\minicua\core\__init__.py`
- Create: `D:\minicua\src\minicua\core\errors.py`
- Test: `D:\minicua\tests\test_core\test_errors.py`

**Step 1: 写失败测试**

```python
# tests/test_core/test_errors.py
import pytest
from minicua.core.errors import StaleElementError, PageChangedError, CrashError, LoopDetected


def test_errors_have_messages():
    assert str(StaleElementError(index=5)) == "Element index 5 is stale"
    assert str(PageChangedError(before="a.com", after="b.com")) == "Page changed from a.com to b.com"


def test_loop_detected_is_soft():
    err = LoopDetected(repeat_count=5)
    assert err.repeat_count == 5
```

**Step 2: 跑红** `pytest tests/test_core/test_errors.py -v` → FAIL（模块不存在）

**Step 3: 最小实现**

```python
# src/minicua/core/errors.py
class CUAError(Exception): ...

class StaleElementError(CUAError):
    def __init__(self, index: int | None = None):
        self.index = index
        super().__init__(f"Element index {index} is stale")

class PageChangedError(CUAError):
    def __init__(self, before: str, after: str):
        super().__init__(f"Page changed from {before} to {after}")

class CrashError(CUAError): ...
class LoopDetected(CUAError):
    def __init__(self, repeat_count: int):
        self.repeat_count = repeat_count
        super().__init__(f"Loop detected: {repeat_count} repeats")
```

**Step 4: 跑绿** `pytest tests/test_core/test_errors.py -v` → PASS

**Step 5: 提交**

```bash
git init && git add -A && git commit -m "chore: scaffold minicua package + core errors"
```

---

## Stage 1: browser session（Playwright 封装）

### Task 1.1: BrowserSession 启动/关闭（persistent context）

**Files:**
- Create: `D:\minicua\src\minicua\browser\session.py`
- Create: `D:\minicua\src\minicua\browser\__init__.py`
- Test: `D:\minicua\tests\test_browser\test_session.py`

**Step 1: 写失败测试**

```python
# tests/test_browser/test_session.py
import pytest
from minicua.browser.session import BrowserSession

@pytest.mark.asyncio
async def test_session_start_and_close():
    s = BrowserSession(headless=True)
    await s.start()
    assert s.context is not None
    await s.close()
```

**Step 2: 跑红** `pytest tests/test_browser/test_session.py -v` → FAIL

**Step 3: 最小实现**

```python
# src/minicua/browser/session.py
from playwright.async_api import async_playwright

class BrowserSession:
    def __init__(self, headless: bool = True, user_data_dir: str | None = None):
        self.headless = headless
        self.user_data_dir = user_data_dir

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir or "", headless=self.headless
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()

    async def close(self) -> None:
        await self.context.close()
        await self._pw.stop()
```

**Step 4: 跑绿** → PASS

**Step 5: 提交** `git commit -m "feat(browser): persistent session start/close"`

### Task 1.2: storage_state 保存/加载（跨重启保持登录态）

**Files:**
- Modify: `D:\minicua\src\minicua\browser\session.py`
- Test: `D:\minicua\tests\test_browser\test_session.py`

**Step 1: 写失败测试**

```python
@pytest.mark.asyncio
async def test_storage_state_roundtrip(tmp_path):
    s = BrowserSession(headless=True)
    await s.start()
    await s.page.goto("data:text/html,<h1>hi</h1>")
    await s.page.evaluate("localStorage.setItem('k','v')")
    path = tmp_path / "state.json"
    await s.save_storage_state(path)
    await s.close()

    s2 = BrowserSession(headless=True, storage_state=path)
    await s2.start()
    val = await s2.page.evaluate("localStorage.getItem('k')")  # NOTE: 需要同 origin
    await s2.close()
```

**Step 2: 跑红** → FAIL（方法不存在）

**Step 3: 最小实现**

```python
    def __init__(self, headless=True, user_data_dir=None, storage_state=None):
        self.storage_state = storage_state

    async def start(self):
        ...
        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir or "",
            headless=self.headless,
            storage_state=self.storage_state,   # 加载
        )

    async def save_storage_state(self, path):
        await self.context.storage_state(path=str(path))
```

**Step 4: 跑绿** → PASS（storage_state 需同 origin，测试用 `file://` 或固定 data URL）

**Step 5: 提交** `git commit -m "feat(browser): storage_state save/load"`

### Task 1.3: navigate + 读取 URL/title

**Files:** Modify `session.py`；Test `test_session.py`

**Step 1:** 测试 `await s.navigate("data:text/html,<title>T</title>")` 后 `s.url == ...`、`s.title == "T"`。
**Step 2:** 跑红。**Step 3:** 加 `navigate/get_url/get_title` 方法。**Step 4:** 跑绿。**Step 5:** 提交。

### Task 1.4: crash watchdog（CDP 崩溃检测）

**Files:**
- Create: `D:\minicua\src\minicua\browser\crash_watchdog.py`
- Test: `D:\minicua\tests\test_browser\test_crash.py`

**Step 1: 写失败测试**（用 event handler 触发，不真崩溃）

```python
from minicua.browser.crash_watchdog import CrashWatchdog

@pytest.mark.asyncio
async def test_watchdog_emits_on_crash():
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    await wd._handle_target_crashed("tab-1")   # 内部处理
    assert events
```

**Step 2: 跑红** → FAIL

**Step 3: 最小实现**

```python
class CrashWatchdog:
    def __init__(self):
        self.on_crash: Callable[[str], None] = lambda msg: None
        self.crashed = False

    async def _handle_target_crashed(self, target_id: str):
        self.crashed = True
        self.on_crash(f"target {target_id} crashed")
```

**Step 4: 跑绿** → PASS（真实 CDP `targetCrashed` 监听在集成阶段接入）

**Step 5: 提交** `git commit -m "feat(browser): crash watchdog"`

---

## Stage 2: perception（DOM 序列化 + 截图）

### Task 2.1: DOMElement / BrowserState 模型

**Files:**
- Create: `D:\minicua\src\minicua\perception\dom.py`
- Test: `D:\minicua\tests\test_perception\test_dom.py`

**Step 1: 写失败测试**

```python
from minicua.perception.dom import DOMElement, BrowserState

def test_dom_element_defaults():
    el = DOMElement(index=1, tag="button", text="登录")
    assert el.index == 1 and el.stable_hash == ""

def test_browser_state_selector_map():
    st = BrowserState(url="x", dom_text="[1] <button>登录</button>", selector_map={1: DOMElement(index=1, tag="button", text="登录")})
    assert st.selector_map[1].tag == "button"
```

**Step 2: 跑红。Step 3: 实现两个 dataclass。Step 4: 跑绿。Step 5: 提交。**

### Task 2.2: DOM 线性化 + index（serializer）

**Files:**
- Create: `D:\minicua\src\minicua\perception\serializer.py`
- Test: `D:\minicua\tests\test_perception\test_serializer.py`

**Step 1: 写失败测试**

```python
from minicua.perception.serializer import serialize_dom

INTERACTIVE = {"button", "a", "input", "select", "textarea", "form"}

def test_serialize_assigns_indexes():
    # 用可注入的节点结构代替真实 DOM，保证纯函数可测
    nodes = [
        {"tag": "div", "text": "", "interactive": False, "attrs": {}},
        {"tag": "button", "text": "登录", "interactive": True, "attrs": {}},
        {"tag": "input", "text": "", "interactive": True, "attrs": {"type": "text"}},
    ]
    text, selector_map = serialize_dom(nodes)
    assert "[1]" in text and "[2]" in text
    assert selector_map[1].tag == "button" and selector_map[2].tag == "input"
```

**Step 2: 跑红。Step 3: 实现 `serialize_dom(nodes)`（遍历 → 交互元素分配 index → 生成线性化文本）。Step 4: 跑绿。Step 5: 提交。**

### Task 2.3: 从 Playwright 页面提取 selector_map（集成）

**Files:**
- Create: `D:\minicua\src\minicua\perception\extract.py`
- Test: `D:\minicua\tests\test_perception\test_extract.py`

**Step 1: 写失败测试**（用 inline HTML fixture）

```python
@pytest.mark.asyncio
async def test_extract_from_page(session):
    await session.page.set_content("<button>登录</button><input type=text placeholder=用户名>")
    state = await extract_state(session.page)
    assert state.selector_map[1].tag == "button"
    assert "[1]" in state.dom_text
```

**Step 2: 跑红。Step 3: 实现 `extract_state(page)`：`page.accessibility.snapshot()` 或注入 JS 遍历 → 复用 `serialize_dom`。Step 4: 跑绿。Step 5: 提交。**

### Task 2.4: 截图（use_vision 三态）

**Files:**
- Create: `D:\minicua\src\minicua\perception\screenshot.py`
- Test: `D:\minicua\tests\test_perception\test_screenshot.py`

**Step 1: 写失败测试**

```python
def test_should_capture():
    assert should_capture("dom_only", model_supports_vision=False) is False
    assert should_capture("vision", True) is True
    assert should_capture("auto", True) is True
    assert should_capture("auto", False) is False
```

**Step 2: 跑红。Step 3: 实现 `should_capture(mode, model_supports_vision)` + `capture(page) -> str(b64)`。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 3: action（动作定义 + grounding）

### Task 3.1: 动作 pydantic 定义

**Files:**
- Create: `D:\minicua\src\minicua\action\models.py`
- Test: `D:\minicua\tests\test_action\test_models.py`

**Step 1: 写失败测试**

```python
from minicua.action.models import Action, ClickParams, TypeParams

def test_click_action():
    a = Action(name="click", params=ClickParams(index=1))
    assert a.name == "click" and a.params.index == 1

def test_action_to_tool_schema():
    schema = Action.model_json_schema()
    assert "click" in schema  # 或 tool 函数名
```

**Step 2: 跑红。Step 3: 实现 `ClickParams/TypeParams/ScrollParams/NavigateParams/DoneParams` + `Action`（discriminated union）。Step 4: 跑绿。Step 5: 提交。**

### Task 3.2: grounding（index → locator / 坐标）

**Files:**
- Create: `D:\minicua\src\minicua\action\grounding.py`
- Test: `D:\minicua\tests\test_action\test_grounding.py`

**Step 1: 写失败测试**

```python
from minicua.action.grounding import ground
from minicua.perception.dom import DOMElement

def test_ground_by_index():
    el = DOMElement(index=1, tag="button", text="登录", xpath="//button[1]")
    selector_map = {1: el}
    assert ground(1, selector_map) == el

def test_ground_stale_raises():
    from minicua.core.errors import StaleElementError
    with pytest.raises(StaleElementError):
        ground(99, {1: DOMElement(index=1, tag="button", text="x")})
```

**Step 2: 跑红。Step 3: 实现 `ground(index, selector_map) -> DOMElement`（找不到抛 `StaleElementError`）+ `to_locator(el, page)`（xpath/stable hash → Playwright locator）。Step 4: 跑绿。Step 5: 提交。**

### Task 3.3: ActionRegistry（tool schema 生成）

**Files:**
- Create: `D:\minicua\src\minicua\action\registry.py`
- Test: `D:\minicua\tests\test_action\test_registry.py`

**Step 1: 写失败测试**

```python
from minicua.action.registry import ActionRegistry, register_action

@register_action("click", ClickParams)
async def click(params, page): ...

def test_registry_generates_tools():
    reg = ActionRegistry()
    tools = reg.to_tools()   # OpenAI function 格式
    assert any(t["function"]["name"] == "click" for t in tools)
```

**Step 2: 跑红。Step 3: 实现 `register_action` 装饰器 + `ActionRegistry.to_tools()`（pydantic → OpenAI/Anthropic tool schema）。Step 4: 跑绿。Step 5: 提交。**

### Task 3.4: executor（浏览器动作执行）

**Files:**
- Create: `D:\minicua\src\minicua\action\executor.py`
- Test: `D:\minicua\tests\test_action\test_executor.py`

**Step 1: 写失败测试**（inline HTML）

```python
@pytest.mark.asyncio
async def test_execute_click(session):
    await session.page.set_content("<button id=b onclick='this.textContent=\"clicked\"'>go</button>")
    state = await extract_state(session.page)
    res = await execute(Action(name="click", params=ClickParams(index=1)), session.page, state)
    assert "clicked" in await session.page.inner_text("#b")
```

**Step 2: 跑红。Step 3: 实现 `execute(action, page, state) -> ActionResult`（click/type/scroll/navigate/go_back/press/wait/done/fail 分发）。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 4: controller（agent loop + budget + retry）

### Task 4.1: ChatModel 抽象（anthropic/openai + fake）

**Files:**
- Create: `D:\minicua\src\minicua\controller\llm.py`
- Test: `D:\minicua\tests\test_controller\test_llm.py`

**Step 1: 写失败测试**

```python
from minicua.controller.llm import FakeModel, ChatModel

def test_fake_model_returns_scripted():
    m = FakeModel(responses=[{"name": "done", "params": {}}])
    calls = m.complete(messages=[], tools=[])
    assert calls[0]["name"] == "done"
```

**Step 2: 跑红。Step 3: 实现 `ChatModel` 协议 + `FakeModel`（脚本化返回，测试用）+ `AnthropicModel`/`OpenAIModel` 骨架（真调用留到集成）。Step 4: 跑绿。Step 5: 提交。**

### Task 4.2: agent step loop

**Files:**
- Create: `D:\minicua\src\minicua\controller\agent.py`
- Test: `D:\minicua\tests\test_controller\test_agent.py`

**Step 1: 写失败测试**（FakeModel 脚本化 3 步后 done）

```python
@pytest.mark.asyncio
async def test_agent_runs_to_done(session):
    model = FakeModel(responses=[
        {"name": "navigate", "params": {"url": "data:text/html,<button>ok</button>"}},
        {"name": "click", "params": {"index": 1}},
        {"name": "done", "params": {"success": True}},
    ])
    agent = Agent(session=session, model=model, max_steps=10)
    result = await agent.run(task="click the ok button")
    assert result.done and result.success
```

**Step 2: 跑红。Step 3: 实现 `Agent.step()` + `Agent.run()`（感知→LLM→执行→记录→done 判定）。Step 4: 跑绿。Step 5: 提交。**

### Task 4.3: budget + retry

**Files:**
- Create: `D:\minicua\src\minicua\controller\budget.py`
- Create: `D:\minicua\src\minicua\controller\retry.py`
- Test: `D:\minicua\tests\test_controller\test_budget_retry.py`

**Step 1: 写失败测试**

```python
def test_max_steps_stops():
    budget = Budget(max_steps=3)
    assert not budget.exhausted(step=2)
    assert budget.exhausted(step=3)

def test_retry_on_validation_error():
    retrier = Retry(max_retries=2)
    calls = []
    def f():
        calls.append(1)
        if len(calls) < 2: raise ValueError("bad schema")
        return "ok"
    assert retrier.run(f) == "ok" and len(calls) == 2
```

**Step 2: 跑红。Step 3: 实现 `Budget`（max_steps/max_failures/timeout）+ `Retry.run(fn)`（指数退避 + 重试上限）。接入 `Agent`。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 5: recovery（stale / page-change / loop / crash）

### Task 5.1: stale element recovery（多级降级）

**Files:**
- Create: `D:\minicua\src\minicua\recovery\stale.py`
- Test: `D:\minicua\tests\test_recovery\test_stale.py`

**Step 1: 写失败测试**

```python
from minicua.recovery.stale import relocalize

def test_relocalize_by_stable_hash():
    old = DOMElement(index=1, tag="button", text="登录", stable_hash="abc", xpath="//button")
    new_map = {3: DOMElement(index=3, tag="button", text="登录", stable_hash="abc", xpath="//button")}
    assert relocalize(old, new_map) == 3

def test_relocalize_by_ax_name_fallback():
    old = DOMElement(index=1, tag="button", text="登录", stable_hash="abc", ax_name="Login")
    new_map = {5: DOMElement(index=5, tag="button", text="登录", stable_hash="xyz", ax_name="Login")}
    assert relocalize(old, new_map) == 5
```

**Step 2: 跑红。Step 3: 实现 `relocalize(old, new_map)`：stable hash → xpath → ax_name → None（逐级降级）。Step 4: 跑绿。Step 5: 提交。**

### Task 5.2: page-change detection（DOM 指纹）

**Files:**
- Create: `D:\minicua\src\minicua\recovery\page_change.py`
- Test: `D:\minicua\tests\test_recovery\test_page_change.py`

**Step 1: 写失败测试**

```python
def test_page_fingerprint_changed():
    fp = lambda url, txt: PageFingerprint(url, len(txt), hashlib.sha256(txt.encode()).hexdigest()[:16])
    a = fp("a.com", "<button>1</button>")
    b = fp("b.com", "<button>1</button>")
    c = fp("a.com", "<button>1</button>")
    assert page_changed(a, b) and not page_changed(a, c)
```

**Step 2: 跑红。Step 3: 实现 `PageFingerprint`（url + element_count + text_hash）+ `page_changed(before, after)`。Step 4: 跑绿。Step 5: 提交。**

### Task 5.3: loop detection（动作重复 + 页面停滞）

**Files:**
- Create: `D:\minicua\src\minicua\recovery\loop.py`
- Test: `D:\minicua\tests\test_recovery\test_loop.py`

**Step 1: 写失败测试**

```python
from minicua.recovery.loop import LoopDetector

def test_detects_action_repetition():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(6):
        d.record_action("click", {"index": 1})
    assert d.is_looping()

def test_detects_stagnation():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(6):
        d.record_page_state("a.com", "<button>x</button>", 1)
    assert d.stagnant()
```

**Step 2: 跑红。Step 3: 实现 `LoopDetector`（rolling window 动作 hash + 连续停滞计数 + `nudge_message()`）。Step 4: 跑绿。Step 5: 提交。**

### Task 5.4: crash recovery（重建会话 + 恢复 storage_state + 续跑）

**Files:**
- Create: `D:\minicua\src\minicua\recovery\crash.py`
- Test: `D:\minicua\tests\test_recovery\test_crash.py`

**Step 1: 写失败测试**

```python
@pytest.mark.asyncio
async def test_recover_restarts_session(session, tmp_path):
    state_path = tmp_path / "state.json"
    await session.save_storage_state(state_path)
    await recover(session, checkpoint_dir=tmp_path)   # 模拟崩溃后重建
    assert session.page is not None
```

**Step 2: 跑红。Step 3: 实现 `recover(session, checkpoint_dir)`：close → 用 storage_state 重启 → 从 checkpoint 恢复 event log/task state。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 6: state（event log + checkpoint + trajectory）

### Task 6.1: event log（append-only）

**Files:**
- Create: `D:\minicua\src\minicua\state\events.py`
- Test: `D:\minicua\tests\test_state\test_events.py`

**Step 1: 写失败测试**

```python
from minicua.state.events import EventLog, StepEvent

def test_event_log_append_and_dump():
    log = EventLog()
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    assert len(log.events) == 1
    d = log.model_dump()
    assert d["events"][0]["step"] == 1
```

**Step 2: 跑红。Step 3: 实现 `StepEvent` + `EventLog`（append/iterate/to_jsonl）。Step 4: 跑绿。Step 5: 提交。**

### Task 6.2: checkpoint（save/load）

**Files:**
- Create: `D:\minicua\src\minicua\state\checkpoint.py`
- Test: `D:\minicua\tests\test_state\test_checkpoint.py`

**Step 1: 写失败测试**

```python
def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(step=5, event_log=EventLog(), task_state={"goal": "x"})
    path = tmp_path / "ckpt"
    cp.save(path)
    cp2 = Checkpoint.load(path)
    assert cp2.step == 5 and cp2.task_state["goal"] == "x"
```

**Step 2: 跑红。Step 3: 实现 `Checkpoint`（save/load：event_log + task_state + message_history + storage_state 路径引用）。Step 4: 跑绿。Step 5: 提交。**

### Task 6.3: trajectory recording（JSONL）

**Files:**
- Create: `D:\minicua\src\minicua\state\trajectory.py`
- Test: `D:\minicua\tests\test_state\test_trajectory.py`

**Step 1: 写失败测试**

```python
def test_trajectory_writes_jsonl(tmp_path):
    rec = TrajectoryRecorder(task_id="t1")
    rec.record(StepEvent(step=1, ts=0.0, phase="act"))
    path = tmp_path / "traj.jsonl"
    rec.dump(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["task_id"] == "t1"
```

**Step 2: 跑红。Step 3: 实现 `TrajectoryRecorder`（record + dump JSONL，对齐 OpenCUA 的 task_id + content 流）。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 7: eval（声明式 evaluator + metrics + runner + report）

### Task 7.1: getters（取环境状态）

**Files:**
- Create: `D:\minicua\src\minicua\eval\getters.py`
- Test: `D:\minicua\tests\test_eval\test_getters.py`

**Step 1: 写失败测试**

```python
from minicua.eval.getters import GETTERS, page_url, element_exists

@pytest.mark.asyncio
async def test_page_url_getter(session):
    await session.page.goto("data:text/html,<div id=b>x</div>")
    assert await page_url(session) == session.page.url

@pytest.mark.asyncio
async def test_element_exists_getter(session):
    await session.page.set_content("<div id=b>x</div>")
    assert await element_exists(session, selector="#b") is True
    assert await element_exists(session, selector="#none") is False
```

**Step 2: 跑红。Step 3: 实现 getter 注册表（`page_url`、`element_exists`、`element_text`、`cookie_exists`、`local_storage`、`screenshot`）。Step 4: 跑绿。Step 5: 提交。**

### Task 7.2: metrics（纯函数 0/1）

**Files:**
- Create: `D:\minicua\src\minicua\eval\metrics.py`
- Test: `D:\minicua\tests\test_eval\test_metrics.py`

**Step 1: 写失败测试**

```python
from minicua.eval.metrics import exact_match, contains, element_exists_metric

def test_exact_match():
    assert exact_match("a.com", {"expected": "a.com"}) == 1
    assert exact_match("b.com", {"expected": "a.com"}) == 0

def test_contains():
    assert contains("hello world", {"expected": "world"}) == 1
```

**Step 2: 跑红。Step 3: 实现 `exact_match/contains/regex_match/count_eq/element_exists_metric`（签名统一 `metric(actual, rule) -> float`）。Step 4: 跑绿。Step 5: 提交。**

### Task 7.3: 声明式 evaluator（metric + getter + conj）

**Files:**
- Create: `D:\minicua\src\minicua\eval\evaluator.py`
- Test: `D:\minicua\tests\test_eval\test_evaluator.py`

**Step 1: 写失败测试**

```python
@pytest.mark.asyncio
async def test_evaluator_and(session):
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": ["element_exists_metric", "exact_match"],
        "conj": "and",
        "result": [
            {"getter": "element_exists", "selector": "#ok"},
            {"getter": "element_text", "selector": "#ok"},
        ],
        "expected": [{"expected": True}, {"expected": "done"}],
    }
    assert await evaluate(session, spec) == 1.0
```

**Step 2: 跑红。Step 3: 实现 `evaluate(session, spec)`：对每个 `func[i]` 调 `getter` 取 actual → `metric(actual, expected[i])` → 按 `conj`（and=全 1 / or=任一 1）聚合。Step 4: 跑绿。Step 5: 提交。**

### Task 7.4: 六指标聚合（从 event log）

**Files:**
- Create: `D:\minicua\src\minicua\eval\metrics_aggregate.py`
- Test: `D:\minicua\tests\test_eval\test_aggregate.py`

**Step 1: 写失败测试**

```python
from minicua.eval.metrics_aggregate import aggregate

def test_aggregate_six_metrics():
    logs = [EventLog(...), EventLog(...)]   # 构造 1 成功 1 失败
    results = [1.0, 0.0]
    m = aggregate(logs, results)
    assert set(m.keys()) == {"task_success", "avg_tool_calls", "token_cost", "latency", "recovery_rate", "invalid_action_rate"}
    assert m["task_success"] == 0.5
```

**Step 2: 跑红。Step 3: 实现 `aggregate(event_logs, results)`（六指标从 StepEvent 提取：act 事件数=工具调用、tokens 求和、latency 求和、recovered 计数/触发、error 动作/总动作）。Step 4: 跑绿。Step 5: 提交。**

### Task 7.5: runner + report

**Files:**
- Create: `D:\minicua\src\minicua\eval\runner.py`
- Create: `D:\minicua\src\minicua\eval\report.py`
- Test: `D:\minicua\tests\test_eval\test_runner.py`

**Step 1: 写失败测试**（用 FakeModel + 一个最小任务定义）

```python
@pytest.mark.asyncio
async def test_runner_one_task():
    task = TaskDef(id="t1", instruction="click ok", setup=..., evaluator={...})
    res = await run_task(task, model=FakeModel([...]))
    assert res.task_id == "t1" and res.success in (True, False)
```

**Step 2: 跑红。Step 3: 实现 `TaskDef` + `run_task`（setup → agent.run → evaluate → 收集 event log）。Step 4: 跑绿。Step 5: 提交。**

---

## Stage 8: cli

### Task 8.1: 单任务运行

**Files:**
- Create: `D:\minicua\src\minicua\cli\main.py`
- Create: `D:\minicua\src\minicua\cli\run.py`
- Test: `D:\minicua\tests\test_cli\test_run.py`

**Step 1: 写失败测试**（typer `CliRunner`）

```python
from typer.testing import CliRunner
from minicua.cli.main import app

def test_run_help():
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
```

**Step 2: 跑红。Step 3: 实现 typer app + `run` 子命令（接 controller）。Step 4: 跑绿。Step 5: 提交。**

### Task 8.2: 评测套件

**Files:**
- Modify: `D:\minicua\src\minicua\cli\eval.py`
- Test: `D:\minicua\tests\test_cli\test_eval.py`

**Step 1: 写失败测试**（`eval --tasks dir` 输出 report 文件）。
**Step 2: 跑红。Step 3: 实现 `eval` 子命令（加载任务集 → runner → 六指标 → report 到文件）。Step 4: 跑绿。Step 5: 提交。**

---

## 依赖关系总览

```
Stage 0 scaffold
   └─► Stage 1 browser ─► Stage 2 perception ─► Stage 3 action ─► Stage 4 controller
                                                                        │
                                  Stage 5 recovery ◄────────────────────┘
                                        │
                                        ▼
                                  Stage 6 state ─► Stage 7 eval ─► Stage 8 cli
```

- **Stage 5 (recovery)** 依赖 Stage 2（selector_map）+ Stage 3（grounding）+ Stage 4（loop 接入）。
- **Stage 6 (state)** 被 Stage 4（记录）与 Stage 5（checkpoint 恢复）依赖。
- **Stage 7 (eval)** 依赖 Stage 6（event log 供六指标）+ Stage 4（runner 跑 agent）。
- **Stage 8 (cli)** 依赖 Stage 4 + Stage 7。

---

## 测试策略要点

1. **LLM 全部用 `FakeModel`**（脚本化返回），不真调 API——保证单测确定性、零成本。
2. **浏览器用 inline HTML fixture**（`page.set_content` / `data:` URL），不依赖外部网站。
3. **纯函数（serializer / grounding / relocalize / metrics / aggregate / loop detector）用纯单元测试**，不碰浏览器。
4. **集成点**（extract_state、execute、crash recover）用 `pytest-asyncio` + `BrowserSession(headless=True)` fixture，`session` 为 module-scoped。
5. 每个 Stage 结束跑 `pytest tests/test_<stage>/ -v` 全绿再进入下一 Stage。
