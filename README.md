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

## 快速开始

```bash
uv sync
export PLAYWRIGHT_BROWSERS_PATH=/d/playwright-browsers   # Windows（D 盘）

uv run minicua run tasks/click_button.json --script script.json   # 单任务
uv run minicua eval tasks/ --output out/                          # 一批任务
uv run minicua report out/results.json --output out/              # 重渲染报告
```

## 参考

- [Browser Use](https://github.com/browser-use/browser-use) — browser harness + persistent state + recovery loop
- [OSWorld](https://github.com/xlang-ai/OSWorld) — 长程真实桌面任务 benchmark
