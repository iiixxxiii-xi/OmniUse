# OmniUse

长程 **computer-use agent** —— 重点不是「点击网页」，而是 **browser harness + persistent state + recovery loop** 的完整 Runtime，覆盖 browser 与 desktop（VM）两个动作空间。

> 包名 / 命令行入口为 `minicua`，项目名 OmniUse。

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

## 核心能力

不是浏览器自动填表，而是：

- **browser persistent session** —— Playwright 持久会话 + 崩溃看门狗
- **hybrid perception** —— DOM 线性化 + screenshot 双通道
- **action grounding** —— 模型输出 index → 真实 locator
- **recovery** —— stale element 重定位 / page-change 检测 / loop 检测 / crash 重建
- **long-horizon state** —— 任务状态、checkpoint、append-only event log（JSONL 轨迹）
- **verification** —— 声明式 evaluator（getter → metric → conj）
- **两个动作空间** —— `browser/`（Playwright 驱动 Chromium）+ `desktop/`（本地桌面 / SSH VM，`pyautogui`，OSWorld 风格），同一套 agent 循环与恢复机制通用

## 快速开始

```bash
uv sync
export PLAYWRIGHT_BROWSERS_PATH=/d/playwright-browsers   # Windows（D 盘）

uv run minicua run tasks/click_button.json --script script.json   # 单任务
uv run minicua eval tasks/ --output out/                          # 一批任务
uv run minicua report out/results.json --output out/              # 重渲染报告
```

## 参考

[Browser Use](https://github.com/browser-use/browser-use)（browser harness + persistent state + recovery loop）· [OSWorld](https://github.com/xlang-ai/OSWorld)（长程真实桌面任务）· OpenCUA
