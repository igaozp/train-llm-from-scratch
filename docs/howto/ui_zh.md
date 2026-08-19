# 控制面板 UI

这是一个精致的 **Streamlit** 应用，包装了整条流水线：理论、一键训练、实时日志、指标图表、评估和对话。因此无需记住命令也能操作全部流程。

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

## 包含的内容

- **首页**：流水线图、实时任务状态和 GPU 状态一览。
- **数据**：启动 `prepare_*` 脚本，并观察文件在 `/ephemeral/data` 下出现。
- **每个阶段一个页面**（Pretrain、SFT、Reward、DPO、PPO、GRPO）。每个页面提供：
    1. 该阶段的**理论 + 手绘图**（直接取自这些文档）；
    2. 将配置写入阶段 `configs/*.json` 的**配置表单**；
    3. 用于在后台运行真实训练脚本的 **Launch** 按钮；
    4. 从 JSONL 读取的**实时日志尾部** + **指标图表**（loss / reward / KL / accuracy）。
- **评估**：在任意检查点上运行 GSM8K 准确率，并检查样例生成。
- **对话**：在进程内与任意检查点对话，提供 temperature / top-p / top-k 控制。

## 启动机制

应用会构建与你手动输入相同的命令（`torchrun … scripts/train_sft.py --config …`），以分离的后台进程运行它，并将日志文件流式显示在页面上。因此，即使你切换页面或刷新，任务仍会继续运行。**Stop** 按钮会干净地终止任务（以及所有 `torchrun` worker）。

!!! warning "一次只能运行一个 GPU 任务"
    两个多 GPU `torchrun` 任务会争用同一批显卡并导致 OOM。当另一个 GPU 任务正在运行时（例如预训练占用两张 H100），UI 的 **GPU 忙碌保护**会禁用其他 GPU 阶段的 *Launch*。CPU / smoke 运行和进程内**对话**页面始终可用。

!!! tip "Smoke 模式"
    每个阶段页面都有一个 *smoke* 开关，使用微型 `configs/smoke/*.json`（小模型、少量步骤、CPU），从而可在数秒内验证完整流程，再投入真正的训练。
