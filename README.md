# OmniUse

一个长程、浏览器优先的 **computer-use agent**。它驱动真实 Chromium（Playwright）感知页面为线性化 DOM、把模型的工具调用落回页面元素，并从 stale 元素 / 页面变化 / 死循环 / 崩溃中恢复；另有 desktop 动作空间可驱动本地桌面或 SSH 连接的 VM（OSWorld 风格）。运行结束后用声明式 evaluator 给自己打分。

> 包名 / 命令行入口为 `minicua`，项目名 OmniUse。

**架构（一句话）：** *感知 → 动作/落点 → 控制循环 → 恢复 → 状态 → 评测* 的分层流水线，外包一个持久浏览器会话；Playwright 是主动作空间、DOM 线性化是主感知、声明式 evaluator（getter → metric → conj）当裁判。

**两个动作空间。** `browser/` 流水线通过 Playwright 驱动真实 Chromium。并行的 `desktop/` 流水线驱动本地桌面或 SSH 连接的 VM（`pyautogui` + 截图，走持久 `paramiko` 通道），用于 OSWorld 风格 GUI 任务——同一个 agent 循环与恢复机制对两者通用，无需改动。

## 安装与测试

```bash
# Python 3.12 + uv
uv sync
export PLAYWRIGHT_BROWSERS_PATH=/d/playwright-browsers   # Windows（D 盘）；见 tests/conftest.py
uv run pytest -v                                        # 全量（TDD，fake model + 内联 HTML fixture）
```

## 运行

```bash
# 单任务（默认 FakeModel — 无需 API key）
uv run minicua run tasks/click_button.json --script script.json

# 一批任务 -> report.md / report.csv / results.json
uv run minicua eval tasks/ --output out/

# 从已存 results.json 重渲染报告（无浏览器）
uv run minicua report out/results.json --output out/
```

`--script` 是模型响应的脚本化 JSON 列表，例如
`[{"name": "click", "params": {"index": 1}}, {"name": "done", "params": {"success": true}}]`。

## 任务格式（声明式——新任务纯 JSON）

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

- **getter**（`result`）——读取最终浏览器状态（`page_url`、`page_text`、`element_exists`、`element_text`、`element_attribute`、`element_count`、`cookie_exists`、`local_storage`、`screenshot`、`page_title`）。
- **metric**（`func`）——把 getter 的值与 `expected` 比较（`exact_match`、`contains`、`regex_match`、`count_eq`、`element_exists_metric`、`match_in_list`、`is_in_list`）。
- **conj**——`"and"`（每项检查都过）或 `"or"`（任一检查过）。

## 六项运行指标

`task_success`、`avg_tool_calls`、`token_cost`、`latency`、`recovery_rate`、`invalid_action_rate`——从类型化事件日志提取，并做了除零保护（见 `src/minicua/eval/metrics_aggregate.py`）。

## 目录结构

```
src/minicua/
  browser/     持久 Playwright 会话 + 崩溃看门狗
  perception/  DOM 线性化、selector 映射、截图
  action/      动作模型、落点（index → locator）、执行器、注册表
  controller/  agent 循环、预算、重试、ChatModel + FakeModel
  desktop/     desktop 动作空间：本地 + SSH 驱动 VM（OSWorld 风格）
  recovery/    stale 重定位、页面变化、死循环、崩溃重建
  state/       append-only 事件日志、检查点、轨迹（JSONL）
  eval/        getter、metric、evaluator、六指标聚合、runner、report
  cli/         run / eval / report 命令
```
