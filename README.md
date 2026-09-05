<div align="center">
  <h1>OmniUse</h1>
  <h3>长程 computer-use agent —— browser + desktop 双空间，重点是 recovery，不是点网页</h3>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Playwright-browser-45ba4b" alt="Playwright">
  <img src="https://img.shields.io/badge/OSWorld-desktop-ff69b4" alt="OSWorld">
</div>

<br>

## 能做什么

OmniUse 让 AI agent 像人一样操作电脑——打开网页、点击、输入，也能驱动真实桌面 / VM。但重点不是「能点一下」，而是**长程任务里不崩、不迷路、能恢复**：持久会话、stale 元素重定位、页面变化检测、死循环检测、崩溃重建。

> 包名 / 命令行入口为 `minicua`，项目名 OmniUse。

## 核心能力

- **browser persistent session** — Playwright 持久会话 + 崩溃看门狗
- **hybrid perception** — DOM 线性化 + screenshot 双通道
- **action grounding** — 模型输出 index → 真实 locator
- **recovery** — stale element 重定位 / page-change 检测 / loop 检测 / crash 重建
- **long-horizon state** — 任务状态、checkpoint、append-only event log（JSONL 轨迹）
- **verification** — 声明式 evaluator（getter → metric → conj）
- **两个动作空间** — `browser/`（Playwright 驱动 Chromium）+ `desktop/`（本地桌面 / SSH VM，OSWorld 风格）

## 架构

```
Screenshot / DOM / Accessibility Tree
                  ↓
        State Representation
                  ↓
           Agent Policy
                  ↓
  mouse / keyboard / browser / shell
                  ↓
         Environment State
                  ↓
       Success Verification
                  ↓
          Recovery / Replan
```

## 快速开始

```bash
uv sync
uv tool install .   # 装成全局命令 minicua
export PLAYWRIGHT_BROWSERS_PATH=/d/playwright-browsers   # Windows（D 盘）
```

### 对话（claude 风格）

```bash
minicua                              # 直接进对话，auto 模式自动判断 browser / desktop
> 打开 https://xxx.com 帮我填登录表单   # 自动走 browser
> 打开记事本写点东西                   # 自动走 desktop
> exit
```

默认视觉模型 `deepseek-v4-flash-vision-exp`；想强制指定加 `--mode browser` 或 `--mode desktop`。

```bash
minicua run tasks/click_button.json --script script.json   # 单任务
minicua eval tasks/ --output out/                          # 一批任务
```

## 任务格式

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

- **getter**（`result`）读取最终状态（`page_url` / `page_text` / `element_text` / `screenshot` …）
- **metric**（`func`）比较（`exact_match` / `contains` / `regex_match` / `count_eq` …）
- **conj** — `and`（每项都过）/ `or`（任一过）

## 六项运行指标

`task_success`、`avg_tool_calls`、`token_cost`、`latency`、`recovery_rate`、`invalid_action_rate`（从类型化 event log 提取，做除零保护）。

## 目录结构

```
src/minicua/
  browser/     持久 Playwright 会话 + 崩溃看门狗
  perception/  DOM 线性化、selector 映射、截图
  action/      动作模型、落点（index → locator）、执行器
  controller/  agent 循环、预算、重试、ChatModel + FakeModel
  desktop/     desktop 动作空间（本地 / SSH VM，OSWorld 风格）
  recovery/    stale 重定位、页面变化、死循环、崩溃重建
  state/       append-only event log、checkpoint、轨迹
  eval/        getter、metric、evaluator、runner、report
  cli/         run / eval / report 命令
```

## 参考

- [Browser Use](https://github.com/browser-use/browser-use) — browser harness + persistent state + recovery loop
- [OSWorld](https://github.com/xlang-ai/OSWorld) — 长程真实桌面任务 benchmark
