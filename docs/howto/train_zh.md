# 训练（UI 与 CLI）

每个阶段有两种运行方式：使用**命令行**（完整控制，最适合长时间任务），或使用 **Streamlit 控制面板**（表单、一键启动、实时日志和图表——参见 [UI](ui_zh.md)）。

## 安装

```bash
pip install -e ".[train]"     # 可编辑安装——不再需要 PYTHONPATH=.
export HF_HOME=/ephemeral/hf_cache
```

## 完整流水线（CLI）

每条命令会从 `configs/` 读取其阶段 JSON（可通过 `--field` 覆盖任意字段，或用 `--config` 指向另一份文件）。单 GPU 使用 `python`，多 GPU 使用 `torchrun`。

=== "1. 数据"

    ```bash
    python scripts/prepare_pretrain_data.py --split val   --out /ephemeral/data/pile_dev.h5
    python scripts/prepare_pretrain_data.py --split train --num_shards 1 --out /ephemeral/data/pile_train.h5
    python scripts/prepare_sft_data.py
    python scripts/prepare_preference_data.py --source both
    python scripts/prepare_rl_prompts.py
    ```

=== "2. 预训练"

    ```bash
    # 单 GPU
    python scripts/pretrain_base.py --config configs/pretrain.json
    # 两张 GPU（有效 batch = batch_size * grad_accum * num_gpus）
    torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py --config configs/pretrain.json
    ```

=== "3. 对齐"

    ```bash
    torchrun --standalone --nproc_per_node=2 scripts/train_sft.py
    torchrun --standalone --nproc_per_node=2 scripts/train_reward.py
    torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py --loss_type dpo
    torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py --reward_source verifier
    torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py
    ```

一键运行完整对齐链路：

```bash
bash scripts/run_posttraining.sh        # SFT → RM → DPO → PPO → GRPO → eval table
```

## 多 GPU 注意事项

- `torchrun --standalone --nproc_per_node=N` 会启动 N 个数据并行 rank（DDP + bf16）。只有 rank 0 会记录日志并保存检查点。
- 在开发机器（2×H100、无 NVLink）上，教学型注意力会为每个 block 实体化一个 `(B, n_head, T, T)` 张量，因而内存随序列长度增长。在上下文长度为 1024 时，使用 `--batch_size 8 --grad_accum 12`，再通过累积恢复有效 batch。请设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 输出位置

- 检查点 → `/ephemeral/ckpts/<stage>.pt`（每个都携带自己的解析后 `cfg`）。
- 指标 → `/ephemeral/logs/<stage>_<timestamp>.jsonl`（每个记录步骤一行 JSON）。UI 会实时绘制这些数据；也可使用 `--use_wandb true` 镜像到 Weights & Biases。

随后可在 GSM8K 上[评估](../08_evaluation_zh.md)，并与任意检查点[对话](../09_inference_zh.md)。
