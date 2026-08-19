# 图表

文档中的图表是**手绘风格、带颜色编码的 Mermaid 草图**，会预先渲染为 PNG 并嵌入文档。之所以预渲染（而不是依赖实时 ```` ```mermaid ```` 代码块），是因为 GitHub 的 Mermaid 实时渲染器无法稳定支持 `look: handDrawn` 样式，且一些 Markdown 查看器（例如 VS Code 预览）会阻止 SVG；嵌入 **PNG** 能让手绘效果在所有位置显示。每篇文档也会在图片下方的可折叠 *“Mermaid 源码”* 区块中保留可编辑源码。

## 文件

- `src/*.mmd`：规范的手绘 Mermaid 源文件（使用 `look: handDrawn` + 颜色调色板）。
- `*.png`：文档嵌入的渲染图片（顶层 README 还使用 `README.png`）。

## 编辑后重新生成

编辑相关的 `src/<name>.mmd`，然后在仓库根目录运行：

```bash
bash scripts/render_diagrams.sh
```

该命令会将所有 `src/*.mmd` 重新渲染至 `docs/diagrams/<name>.png`。它需要 Mermaid CLI（`npm i -g @mermaid-js/mermaid-cli`）和用于无头渲染的 Chrome / Chromium（若 Chrome 不在 `/usr/bin/google-chrome-stable`，请设置 `CHROME=/path/to/chrome`）。

## 颜色图例

🟩 数据 / 语料库 · 🟦 预处理 · 青绿色存储（HDF5 / JSONL） · 🟨 模型 / 训练循环 · 🟧 RL / 奖励 · 🟥 损失 / 目标函数 · 🟪 评估 · ⬜ 检查点
