# Computer-Use Agent (CUA) 设计文档

> 日期：2026-08-31
> 状态：Draft（对应实现计划 `2026-08-31-cua-implementation.md`）
> 项目定位：Long-Horizon Computer-Use Agent（浏览器为主，桌面域可扩展）

---

## 1. 项目定位

这是一个 **computer-use agent**：以浏览器为第一动作空间，跑在 Playwright 之上，目标是完成「长视野」的浏览器任务（多步、跨页面、需要登录态/表单/动态页面），并在 OSWorld 2.0 / T3-bench / 自定义浏览器任务集上评测。

核心 pipeline：

```
Screenshot / DOM / Accessibility Tree
        │
        ▼
State Representation（DOM 线性化 + 元素 index + 可选截图）
        │
        ▼
Agent Policy（LLM，text-only 或 VLM）
        │
        ▼
Action（mouse / keyboard / browser / shell）
        │
        ▼
Environment State（Playwright 页面状态 + 事件日志）
        │
        ▼
Success Verification（声明式 evaluator，与 agent 自报 done 解耦）
        │
        ▼
Recovery / Replan（stale element / page-change / loop / crash）
```

**重点不是「点击网页」**。点击网页是任何浏览器自动化库都能做的事（Playwright 一行 `page.click()`）。本项目的价值在于长视野任务里那些「库不会替你做」的 **runtime primitive**，以及把这些 primitive 组合成一个可评测、可恢复、可记录轨迹的 agent loop。

---

## 2. 「新」在哪：6 个不可替代的 runtime primitive

延续「为什么 harness 薄、哪些 runtime primitive 不可替代」的思路：**harness（agent loop 的骨架）应该薄**，但以下 6 个 primitive 是 LLM + Playwright 之上、别人不会替你做的硬东西，是本项目不可替代的产出：

### 2.1 Persistent Session（持久会话）
- 浏览器实例跨 step 存活，**登录态 / cookie / localStorage / storage_state** 在任务内、甚至重启后保持。
- 与「每步开新 browser、每步重新登录」的玩具方案区分开——长视野任务（购物、后台、多页表单）依赖持久会话。
- 关键点：**storage_state 序列化/反序列化**（Playwright `context.storage_state()`），作为 checkpoint 的一部分。

### 2.2 Grounding（动作落地）
- Agent 输出的是抽象意图（「点登录按钮」），必须 **ground 到具体 DOM 元素**（`index` 或坐标）。
- 动作空间必须同时服务 **text-only 模型**（吃 index）和 **VLM**（吃坐标 + 截图）。
- 关键点：`selector_map: dict[int, Element]`（index → 元素），元素线性化时给可交互元素分配稳定 index。

### 2.3 Stale-Element Recovery（失效元素恢复）
- SPA 重渲染、导航、动态加载后，旧的 `index`/引用失效。这是 computer-use 里最高频的失败来源。
- 需要 **多级恢复**：stable hash → xpath → ax_name → 坐标 → 重定位，逐级降级匹配（借鉴 browser-use 的 `MatchLevel` + `compute_stable_hash`）。

### 2.4 Loop Detection（环路检测）
- Agent 卡在重复动作 / 页面无变化，是长任务最常见的「死循环」。
- 两级信号：**动作重复**（rolling window 动作 hash）+ **页面停滞**（DOM 指纹不变）。软检测（nudge 提示），不硬阻断。

### 2.5 Checkpoint + Crash Recovery（检查点 + 崩溃恢复）
- 长任务每 N 步落 checkpoint（event log + storage_state + message history），崩溃后可从最近 checkpoint 恢复，而不是从头来。
- 浏览器崩溃检测（CDP `Target.targetCrashed` / 进程死亡 / 连接断开）→ 自动重建会话 → 恢复状态。

### 2.6 Verification（验证）
- 任务成败由 **声明式 evaluator** 判定（getter 取环境状态 + metric 纯函数判 0/1），与 agent 自报的 `done`/`success` 解耦——agent 说完成不算，evaluator 说了算。

---

## 3. 总体架构：六大组件

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (typer)                       │
└─────────────────────────────────────────────────────────┘
        │  run task / eval suite
        ▼
┌─────────────────────────────────────────────────────────┐
│  Controller：agent loop + budget + retry + model 抽象     │
└───────┬──────────────┬──────────────┬──────────────────┘
        │ state        │ actions      │ errors
        ▼              ▼              ▼
┌────────────┐  ┌────────────┐  ┌────────────────────┐
│ Perception │  │  Action     │  │  Recovery           │
│ DOM + shot │  │ 定义+ground │  │ stale/page/loop/crash│
└──────┬─────┘  └──────┬─────┘  └─────────┬──────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│  Browser（Playwright 封装）：persistent session + CDP      │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐        ┌──────────────────────────────┐
│  State       │◄──────►│  Eval：declarative evaluator  │
│ event log /  │        │ + 6 metrics + runner + report │
│ checkpoint / │        └──────────────────────────────┘
│ trajectory   │
└─────────────┘
```

六大组件职责：

| 组件 | 目录 | 职责 |
|---|---|---|
| **Perception** | `src/minicua/perception/` | DOM 序列化（accessibility tree 线性化 + index）、截图、`use_vision` 三态 |
| **Action** | `src/minicua/action/` | 动作空间定义（pydantic）、grounding（index→元素）、registry |
| **Controller** | `src/minicua/controller/` | agent loop、step、budget/step 上限、retry、模型抽象（anthropic/openai） |
| **Recovery** | `src/minicua/recovery/` | stale-element、page-change、loop、crash 四类恢复策略 |
| **State** | `src/minicua/state/` | event log、long-horizon task state、checkpoint、trajectory recording |
| **Eval** | `src/minicua/eval/` | 声明式 evaluator（metric+getter+conj）、6 指标、runner、report |
| **Browser** | `src/minicua/browser/` | Playwright 封装：persistent session、CDP、崩溃检测 |
| **core** | `src/minicua/core/` | 跨组件的公共模型、配置、错误、日志 |

---

## 4. 六大组件详细设计

### 4.1 Perception

**输入**：Playwright `Page`（或 CDP 连接）。
**输出**：`BrowserState`（DOM 线性化文本 + selector_map + 可选截图）。

```python
# src/minicua/perception/dom.py
@dataclass
class DOMElement:
    index: int | None            # 可交互元素才有 index
    tag: str
    text: str                    # 截断后的文本
    attributes: dict[str, str]   # 白名单属性：id/name/type/role/aria-label/title/...
    xpath: str
    stable_hash: str             # 用于 stale-element 恢复（过滤动态 class）
    ax_name: str | None
    bbox: Box | None             # 供 VLM 坐标 grounding

@dataclass
class BrowserState:
    url: str
    dom_text: str                          # 线性化后的 DOM（喂给 LLM）
    selector_map: dict[int, DOMElement]    # index → 元素
    screenshot_b64: str | None             # 可选
```

**DOM 序列化**（`perception/serializer.py`）：
- 用 Playwright 的 accessibility snapshot（`page.accessibility.snapshot()`）或注入 JS 遍历 DOM，**只保留可交互 + 有语义的元素**（button/a/input/select/textarea/form/label/h[1-6]/[role]/[onclick] 等）。
- 每个可交互元素分配 `index`（1..N），非交互的语义节点折叠成文本。
- 输出格式（借鉴 browser-use 的 `DOMTreeSerializer`）：

```
[1] <button> 登录 </button>
[2] <input type=text placeholder="用户名" />
[3] <input type=password placeholder="密码" />
[4] <a> 忘记密码? </a>
```

**`use_vision` 三态**（`core/config.py`）：

| 态 | DOM | 截图 | 适用模型 |
|---|---|---|---|
| `dom_only` | ✅ | ❌ | text-only（DeepSeek、o 系列 text） |
| `vision` | ✅ | ✅ | VLM（gpt-4o、claude、gemini） |
| `auto` | ✅ | 模型支持则加 | 默认；按模型能力自动判定 |

> **DOM-only 是核心兼容性保证**：让 text-only 模型也能跑，不依赖 VLM。

**Perception 层只负责「看」，不负责「动」**；grounding（index→坐标→点击）在 Action 层。

### 4.2 Action

**动作空间定义**（`action/models.py`）：用 pydantic 建模，可直接转成 OpenAI/Anthropic 的 tool schema。

```python
class ClickParams(BaseModel):
    index: int                      # DOM-only grounding
    x: float | None = None          # 可选：坐标（VLM）
    y: float | None = None

class TypeParams(BaseModel):
    index: int
    text: str
    press_enter: bool = False

class Action(BaseModel):
    name: Literal["click","type","scroll","navigate","go_back","press","wait","extract","done","fail"]
    params: ClickParams | TypeParams | ...   # discriminated union
```

**Grounding**（`action/grounding.py`）：把 `index` 解析为真实元素引用，供浏览器执行：
- `index` → `selector_map[index]` → Playwright locator（用 stable hash / xpath 定位）。
- 如果 index 失效 → 抛 `StaleElementError`，交给 Recovery 层（4.4）。

**动作空间分两层**（关键技术决策 1）：
1. **浏览器层（第一动作空间）**：click/type/scroll/navigate/go_back/press/wait/extract/done/fail——通过 Playwright 实现。
2. **桌面层（可扩展第二动作空间）**：mouse/keyboard/PyAutoGUI——通过统一 `ActionExecutor` 接口接入，默认不启用，跑 OSWorld 桌面任务时启用。

```python
# src/minicua/action/executor.py
class ActionExecutor(Protocol):
    async def execute(self, action: Action, state: BrowserState) -> ActionResult: ...
```

**ActionRegistry**（`action/registry.py`）：借鉴 browser-use 的 `Registry`——动作是「pydantic param model + 执行函数」的注册项，自动生成 tool schema（OpenAI function / Anthropic tool）。

### 4.3 Controller

**Agent loop**（`controller/agent.py`），借鉴 browser-use 的 `step()`：

```
while not done and step < max_steps:
    state = perception.get_state()          # 1. 看
    if recovery.should_intervene(state):    # 2. 恢复检查（loop/page-change）
        recovery.intervene()
    output = llm.complete(messages, tools)  # 3. 想（带 retry）
    results = action.execute_all(output)    # 4. 动
    state.record(step, output, results)     # 5. 记（event log）
    if results.is_done: break
```

**Budget + Retry**（`controller/budget.py` + `controller/retry.py`）：
- 预算：`max_steps`、`max_failures`（连续失败上限）、`llm_timeout`、`step_timeout`。
- Retry：LLM 输出 schema 校验失败 → 重试；LLM 超时 → 重试；连续失败达到 `max_failures` → 停止（可选一次 final recovery call）。

**模型抽象**（`controller/llm.py`）：统一 `ChatModel` 接口，两个实现：
- `AnthropicModel`（anthropic SDK，tool use）
- `OpenAIModel`（openai SDK，function calling）
- 统一 `complete(messages, tools) -> ToolCall[]`，屏蔽 SDK 差异。

### 4.4 Recovery

四类恢复策略，独立可测（`recovery/`）：

| 模块 | 触发 | 策略 |
|---|---|---|
| `stale.py` | `StaleElementError`（index 失效） | 多级降级：stable hash → xpath → ax_name → 坐标 → 重新 `get_state` 重定位 |
| `page_change.py` | 动作后 DOM 指纹变化（导航/重渲染） | 检测 URL + DOM 指纹 diff；重渲染则刷新 selector_map 后重试原动作一次 |
| `loop.py` | 动作重复 N 次 / 页面停滞 M 步 | 软 nudge 注入 prompt（「你已重复 X 次，换个方法」），不硬阻断 |
| `crash.py` | CDP `targetCrashed` / 进程死亡 / 连接断开 | 重建 session → 恢复 storage_state → 从最近 checkpoint 续跑 |

**Recovery 的哲学**（借鉴 browser-use `ActionLoopDetector`）：**软检测、注入提示、不硬阻断**——agent 有最终决定权，但给足信号。

### 4.5 State

**Event Log**（`state/events.py`）：append-only 的 `StepEvent` 流，是六指标的唯一数据源。

```python
class StepEvent(BaseModel):
    step: int
    ts: float
    phase: Literal["perceive","think","act","recover","verify"]
    url: str | None
    actions: list[Action] | None
    results: list[ActionResult] | None
    error: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    recovered: bool
```

**Long-Horizon Task State**（`state/task.py`）：`TaskState` = 目标 + 当前子目标 + 已做 + 剩余 + 计划（借鉴 browser-use 的 `PlanItem`）。

**Checkpoint**（`state/checkpoint.py`）：每 N 步序列化一次：
- `event_log` + `task_state` + `message_history`
- `storage_state`（cookie/localStorage）
- 可 `save(dir)` / `load(dir)`，用于崩溃恢复。

**Trajectory Recording**（`state/trajectory.py`）：记录完整轨迹（每步的 DOM 快照 + 截图 + 动作 + 结果 + 时间戳），输出 JSONL（对齐 OpenCUA 的 `Trajectory` 思路：`task_id` + content 流），供后续训练/分析。

### 4.6 Eval

**声明式 evaluator**（`eval/evaluator.py`），完整借用 OSWorld 的「**metric + getter + options + conj**」方法论：

```json
{
  "evaluator": {
    "func": ["url_match", "element_exists"],
    "conj": "and",
    "result": [
      {"type": "getter", "getter": "page_url"},
      {"type": "getter", "getter": "element_exists", "selector": "#success-banner"}
    ],
    "expected": [
      {"type": "rule", "expected": "https://checkout/complete"},
      {"type": "rule", "expected": true}
    ]
  }
}
```

- **Getter**（`eval/getters/`）：从环境取状态——`page_url`、`element_exists`、`element_text`、`cookie_exists`、`local_storage`、`screenshot`、`shell_output`（桌面域）。
- **Metric**（`eval/metrics/`）：纯函数 `metric(actual, rule) -> float(0|1)`——`exact_match`、`contains`、`element_exists`、`count_eq`、`regex_match`。
- **conj**：`and` / `or` 组合多个 metric 结果。

**六指标**（`eval/metrics_aggregate.py`），全部从 event log 提取：

| 指标 | 定义 |
|---|---|
| Task Success | evaluator 判定成功 / 任务总数 |
| Avg Tool Calls | 平均每任务工具调用次数（`act` 事件数） |
| Token Cost | 总 input/output tokens（可折算 cost） |
| Latency | 端到端耗时 + 每 step 耗时 |
| Recovery Rate | 成功恢复的 recovery 事件 / 触发事件 |
| Invalid Action Rate | 无效动作（schema 失败 / index 失效 / 执行报错）/ 总动作 |

**Runner + Report**（`eval/runner.py` + `eval/report.py`）：跑任务集 → 逐任务判定 → 聚合六指标 → 输出 Markdown/JSON report。

---

## 5. 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.12 |
| 浏览器 | **Playwright**（`async_api`，Chromium；CDP 直连做崩溃检测） |
| 桌面域（可扩展） | PyAutoGUI（第二动作空间，可选依赖） |
| 数据模型 | **pydantic** v2（动作/状态/事件/配置全部 pydantic） |
| LLM | **anthropic** SDK + **openai** SDK（统一 `ChatModel` 抽象） |
| CLI | **typer** |
| 测试 | **pytest** + pytest-asyncio（DOM 序列化/grounding/recovery 用 fixture 页面） |
| 打包 | `pyproject.toml` + uv/poetry |

---

## 6. 目录结构

```
minicua/
├── pyproject.toml
├── src/minicua/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py          # CUAConfig（use_vision、max_steps、budget、模型）
│   │   ├── models.py          # 公共 pydantic：ActionResult、Box、StepInfo
│   │   ├── errors.py          # StaleElementError、PageChangedError、CrashError、LoopDetected
│   │   └── logging.py
│   ├── browser/
│   │   ├── session.py         # BrowserSession：persistent session + storage_state
│   │   ├── cdpc.py            # CDP 连接/崩溃监听
│   │   └── crash_watchdog.py  # 崩溃/连接检测
│   ├── perception/
│   │   ├── dom.py             # DOMElement / BrowserState / selector_map
│   │   ├── serializer.py      # DOM → 线性化文本 + index
│   │   └── screenshot.py      # use_vision 三态
│   ├── action/
│   │   ├── models.py          # 动作 pydantic 定义
│   │   ├── grounding.py       # index → locator / 坐标
│   │   ├── registry.py        # ActionRegistry → tool schema
│   │   └── executor.py        # ActionExecutor 接口（browser + desktop 两实现）
│   ├── controller/
│   │   ├── agent.py           # Agent.step() / run()
│   │   ├── budget.py          # 步数/失败/超时预算
│   │   ├── retry.py           # LLM 重试
│   │   └── llm.py             # ChatModel 抽象 + anthropic/openai 实现
│   ├── recovery/
│   │   ├── stale.py
│   │   ├── page_change.py
│   │   ├── loop.py
│   │   └── crash.py
│   ├── state/
│   │   ├── events.py          # StepEvent / EventLog
│   │   ├── task.py            # TaskState / PlanItem
│   │   ├── checkpoint.py      # checkpoint save/load
│   │   └── trajectory.py      # trajectory recording (JSONL)
│   ├── eval/
│   │   ├── evaluator.py       # 声明式 evaluator（metric+getter+conj）
│   │   ├── getters.py         # 状态 getter
│   │   ├── metrics.py         # 纯函数 metric
│   │   ├── metrics_aggregate.py  # 六指标
│   │   ├── runner.py          # 任务集 runner
│   │   └── report.py
│   └── cli/
│       ├── main.py            # typer 入口
│       ├── run.py             # 单任务
│       └── eval.py            # 评测套件
└── tests/
    ├── test_perception/
    ├── test_action/
    ├── test_controller/
    ├── test_recovery/
    ├── test_state/
    └── test_eval/
```

---

## 7. 关键技术决策（写死）

1. **浏览器为主（Playwright），桌面域（mouse/keyboard/PyAutoGUI）作为可扩展的第二动作空间。** 第一动作空间全部走 Playwright；桌面域通过统一 `ActionExecutor` 接口接入，默认不启用，只在跑 OSWorld 桌面任务时启用。

2. **Perception 以 DOM（accessibility tree 线性化 + 元素 index）为主，screenshot 可选。** `use_vision` 三态：`dom_only`（兼容 text-only 模型如 DeepSeek）/ `vision`（DOM+screenshot 用 VLM）/ `auto`（默认）。**DOM-only 意味着 text-only 模型也能跑**，是本项目的兼容性底线。

3. **Eval 走务实路线**：自造 30~50 个本地浏览器任务（带声明式 evaluator）+ 完整借用 OSWorld 的「metric + getter + options + conj」声明式判定方法论；**完整 OSWorld 需图形桌面 VM + 代理，作为 stretch goal**，不阻塞主线。

4. **六指标（Task Success / 平均 Tool Calls / Token Cost / Latency / Recovery Rate / Invalid Action Rate）从 event log 提取。** 指标计算与 agent loop 解耦，只依赖 `state/events.py` 的 append-only 日志。

---

## 8. 范围

**In scope（主线）**：
- 浏览器 persistent session + storage_state
- DOM + screenshot hybrid perception（`use_vision` 三态）
- 动作定义 + grounding（index/坐标）
- agent loop + budget + retry + 模型抽象
- 四类 recovery（stale / page-change / loop / crash）
- event log + checkpoint + trajectory recording
- 声明式 evaluator + 六指标 + runner + report
- CLI（单任务 + 评测套件）
- 30~50 个本地浏览器任务（含 evaluator）

**Stretch goal（写清楚但不阻塞主线）**：
- 完整 OSWorld 2.0（需图形桌面 VM + 代理）——`eval/osworld/` adapter
- T3-bench
- 桌面域第二动作空间（PyAutoGUI）全量启用
- checkpoint 跨进程恢复 + 多轮 resume
- 轨迹可视化

**Explicitly out of scope**：
- 训练/微调模型（OpenCUA 的 model/inference 部分）
- 多 agent 协作、sensitive data 加密/脱敏（浏览器密码管理器级）

---

## 9. 参考项目架构（我们借什么）

### browser-use（`D:\minicua-reference\browser-use\`）
- `dom/views.py`：`EnhancedDOMTreeNode`（DOM+AX+Snapshot 三树融合）+ `DOMSelectorMap`（index→节点）+ `compute_stable_hash`/`MatchLevel`（stale 恢复）。
- `agent/views.py`：`ActionLoopDetector`（动作 hash rolling window + `PageFingerprint` 停滞检测）、`AgentState`、`AgentHistoryList`（checkpoint）。
- `tools/registry/service.py`：`Registry` 动作注册 + 自动生成 tool schema。
- `browser/watchdogs/crash_watchdog.py`：CDP 崩溃/网络超时检测。
- **借**：DOM 序列化 + index、stable hash、loop detector、checkpoint、crash watchdog。

### OSWorld（`D:\minicua-reference\OSWorld\`）
- `desktop_env/actions.py`：`ACTION_SPACE`（MOVE_TO/CLICK/TYPING/…/WAIT/FAIL/DONE）。
- `desktop_env/desktop_env.py`：`DesktopEnv(gym.Env)` 的 `reset/step/evaluate` 生命周期。
- `evaluators/getters/` + `evaluators/metrics/` + task config 的 `evaluator.func/conj/result/expected`：**声明式判定方法论**。
- **借**：metric + getter + conj 判定方法论、gym 式 reset/step/evaluate 生命周期。

### OpenCUA（`D:\minicua-reference\OpenCUA\`）
- `data/data-process/src/schema/action.py`：多平台动作 schema（`GUIAction` 含 PyAutoGUI/Browser/Mobile 动作 + `GUIElement` 归一化/像素 bbox）。
- `data/data-process/src/schema/trajectory.py`：`Trajectory`（task_id + content 流，grounding/end2end 两类）。
- **借**：动作 schema 分层思想、trajectory 的 `task_id + content` 结构。
