# 从头实现后训练：SFT · 奖励模型 · PPO · DPO · GRPO

本仓库在自有 `Transformer` 之上增加了一套完整的、**从头实现**的（纯 PyTorch——不依赖 `trl` / `peft` / `transformers`）后训练工具，覆盖将基础语言模型转变为对齐推理模型的完整现代流水线：

```
基础模型（预训练） ──► SFT ──► 奖励模型 ──► PPO ┐
                         │                       ├─► GRPO / RLVR（GSM8K）
                         └──────► DPO / ORPO / KTO ─┘
```

所有内容都基于仓库自定义模型，并在**真实公开数据集**（Alpaca、Dolly、Anthropic HH-RLHF、UltraFeedback、GSM8K）上训练 / 评估。最核心的指标是**跨阶段的贪心 GSM8K 准确率**。

> **预期：** 在 2×H100 上从头预训练的约 4 亿参数模型能够产生连贯文本、遵循指令，并在每个阶段展示真实的前后提升，但其*绝对* GSM8K 分数仍然有限——前沿级别的数字需要多得多的预训练算力。这里的价值在于：使用真实数据与真实评估得到一条真实的端到端流水线。

---

## 0. 环境

H100 需要 CUDA-12 wheel（原始 `requirements.txt` 为旧预训练路径固定了 cu118，因此没有改动）。请使用独立 venv + `requirements-post.txt`，并将所有大型产物放在 `/ephemeral` 磁盘上。

```bash
python3 -m venv /ephemeral/venv && source /ephemeral/venv/bin/activate
pip install -r requirements-post.txt        # torch cu121 + datasets/wandb/tiktoken/h5py
export HF_HOME=/ephemeral/hf_cache
```

请从仓库根目录、带 `PYTHONPATH=.` 运行所有命令。单 GPU 使用 `python scripts/X.py`；双 GPU 使用 `torchrun --standalone --nproc_per_node=2 scripts/X.py`（DDP + bf16，统一代码路径）。

配置以按阶段划分的 dataclass 形式存放在 [config/post_training_config.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/post_training_config.py) 中；**任意字段均可由 CLI 覆盖**，例如 `--lr 2e-5 --batch_size 16`。默认基础模型约为 4 亿参数（`n_embed=1024, n_head=16, n_blocks=24, context_length=1024`）。

---

## 1. 预训练基础模型（耗时最长的环节）

```bash
# 一次性数据准备（Pile -> /ephemeral 上的扁平 token HDF5）
PYTHONPATH=. python scripts/prepare_pretrain_data.py --split val   --out /ephemeral/data/pile_dev.h5
PYTHONPATH=. python scripts/prepare_pretrain_data.py --split train --num_shards 1 --out /ephemeral/data/pile_train.h5

# 预训练（完整质量需要多天；检查点会持续写入 /ephemeral/ckpts/base_pretrained.pt）
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py
```

[scripts/pretrain_base.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/pretrain_base.py) 在原始 `train_transformer.py` 配方上加入 DDP、bf16 autocast、梯度累积、带预热的余弦 LR 调度和定期检查点保存——即在 2×H100 上训练中型模型所需的一切。原始训练脚本保持不变。

---

## 2. SFT（指令微调）

```bash
PYTHONPATH=. python scripts/prepare_sft_data.py --context_length 1024      # Alpaca+Dolly+GSM8K -> packed HDF5
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_sft.py # -> /ephemeral/ckpts/sft.pt
```

- **从头实现的损失：** 位于 [src/post_training/sft.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/sft.py) 的 prompt-masked 下一个 token CE（`sft_loss`）。只训练助手 token（掩码来自对话模板）；序列**打包**会使每行填满上下文长度。
- **对话格式：** [src/post_training/chat_template.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/chat_template.py)。r50k_base 分词器只将 `<|endoftext|>` 视为特殊 token，因此角色标记（`<|user|>`、`<|assistant|>`、`<think>`、`<answer>`）都是模型学习的普通 token。GSM8K 会被重构为 `<think>…</think><answer>N</answer>`，使模型学习 RL 验证器所奖励的精确输出结构。
- **评估：** 带掩码的 dev 困惑度 + 贪心 GSM8K dev 准确率。

## 3. 奖励模型

```bash
PYTHONPATH=. python scripts/prepare_preference_data.py --source both   # HH-RLHF + UltraFeedback -> JSONL
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_reward.py  # -> reward.pt
```

- **从头实现：** 在 SFT 主干上添加标量奖励头（[src/post_training/reward_model.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/reward_model.py)），并使用 [src/post_training/reward_train.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/reward_train.py) 中的 **Bradley-Terry** 成对损失训练。奖励从最后一个真实 token 读取（因果注意力使右填充安全——不需要注意力掩码）。**评估：** 保留偏好准确率（在有噪声的真实数据上预期约为 0.65–0.75）。

## 4. DPO / ORPO / KTO

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py --loss_type dpo --beta 0.1
#   --loss_type orpo   (reference-free, folds SFT+alignment into one stage)
#   --loss_type kto    (unpaired, reference-KL baseline)
```

- **从头实现：** 三个目标函数均位于 [src/post_training/dpo.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/dpo.py)。策略从 SFT 初始化；冻结的深拷贝作为参考模型（ORPO 不需要）。它们作用于 [src/post_training/rollout.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/rollout.py) 中 `sequence_logprobs` 计算的回复对数概率之和。**评估：** 隐式奖励准确率 / 间隔 + GSM8K dev。

## 5. PPO（经典 RLHF）

```bash
PYTHONPATH=. python scripts/prepare_rl_prompts.py                 # GSM8K + arithmetic warm-up -> JSONL
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py --reward_source verifier
#   --reward_source rm   to use the trained reward model instead of the GSM8K checker
```

- **从头实现：** `src/post_training/ppo.py` 中的 GAE、裁剪策略 / 价值损失；actor-critic 通过 [src/post_training/value_head.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/value_head.py) 共享主干。每轮迭代：rollout → 评分（验证器或 RM）→ 加入逐 token **相对参考模型 KL** 惩罚 → GAE → 多轮裁剪替代目标。**评估：** reward / KL / 裁剪比例 / value-loss 曲线，以及贪心 GSM8K test 准确率。

## 6. GRPO / RLVR（2025 前沿；DeepSeek-R1 风格）

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py --group_size 8
```

- **从头实现：** [src/post_training/grpo.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/grpo.py) 中的组相对优势，以及带 k3 KL 惩罚的逐 token 裁剪替代目标。**没有 critic**：基线是每个提示词自己的 G 个样本。前 `curriculum_iters` 次迭代会运行**算术课程**，使策略在面对完整 GSM8K 前拥有非零奖励方差。**评估：** 平均组奖励、信息量充足组比例、KL、GSM8K test 准确率。

---

## 7. 推理 / 对话（任意阶段检查点）

[scripts/chat.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/chat.py) 可加载**任意**检查点（base / sft / dpo / ppo / grpo）——从检查点本身读取模型维度——并采用对话模板生成（指令模型）或原始续写（基础模型）：

```bash
# instruction-tuned models (chat template applied automatically)
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt  --prompt "What is 13 + 29?"
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/grpo.pt --prompt "..." --greedy
# base model continuation
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/base_pretrained.pt --raw --prompt "Once upon a time"
# interactive REPL (omit --prompt); sampling via --temperature/--top_p/--top_k or --greedy
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt
```

生成复用了与训练 / 评估相同、经过测试的核心（[src/post_training/rollout.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/rollout.py)、[src/post_training/inference.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/inference.py)）。

## 8. 跨阶段结果表

```bash
for s in base_pretrained sft dpo ppo grpo; do
  PYTHONPATH=. python scripts/eval_post_training.py --ckpt /ephemeral/ckpts/$s.pt \
    --label $s --limit 200 --append /ephemeral/logs/stage_table.jsonl
done
PYTHONPATH=. python scripts/eval_post_training.py --table /ephemeral/logs/stage_table.jsonl
```

每个训练器也会在 `/ephemeral/logs/` 下写入 JSONL 指标文件（无需外部服务即可绘图）；传入 `--use_wandb true` 还可镜像到 Weights & Biases。

---

## 设计说明（为何这样构建）

- **包装，不重写。** 教学用的 `Transformer` / `Block` / `Head` / `MLP` 均未修改，仅添加了 `forward_hidden` 方法（返回各 head 消费的 final-LN 后隐藏状态）。价值头、奖励头和所有 RL 对数概率计算都围绕它进行组合。
- **因果注意力 ⇒ 右填充安全。** 最后一个真实 token 永远不会关注它之后的填充，因此 RM（最后 token 奖励）和 DPO（带掩码回复）均不需要注意力掩码；回复掩码会在损失中将填充位置置零。
- **fp32 对数概率。** PPO / GRPO / DPO 会对对数概率作差，因此即便在 bf16 autocast 下也始终在 fp32 中计算。
- **上下文上限。** 学习到的绝对位置会将任意序列限制在 `context_length`；rollout 会强制 `prompt + generation ≤ context_length`。
- **奖励投机 / KL 控制。** 验证器奖励以正确性为主导，附加较小且有上限的格式奖励；相对参考模型的 KL 惩罚将 RL 锚定在 SFT 策略上。

## 测试

```bash
PYTHONPATH=. python tests/test_post_training_smoke.py   # core math: log-probs, heads, parsing, masking
```

每个训练器也可在数秒内以小模型端到端运行（参见开发过程中使用的 smoke 命令）；PPO / GRPO 数学部分还具有独立单元检查（GAE、裁剪损失、组优势、k3 KL）。
