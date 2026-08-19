# LLM 基础

本节解释训练代码默认你已经了解的概念。它不是 PyTorch 入门教程，而是本仓库源文件与现代仅解码器语言模型核心概念之间的一座桥梁。

优先目标是：

1. 理解基础模型与预训练机制；
2. 将这些机制与站点中已有的每个阶段联系起来；
3. 在之后讨论 SFT、奖励模型、DPO、PPO 和 GRPO 时使用同一套语言。

## 完整训练故事

```mermaid
flowchart LR
    RAW[原始文本] --> TOK[分词器]
    TOK --> IDS[token id]
    IDS --> BASE[仅解码器 Transformer]
    BASE --> CE[下一个 token 交叉熵]
    CE --> CKPT[基础检查点]
    CKPT --> SFT[SFT：指令遵循]
    SFT --> PREF[偏好优化]
    PREF --> RL[RL / 验证器优化]
    RL --> CHAT[推理与对话]

    classDef data fill:#d6ffd9,stroke:#27ae60,stroke-width:2px,color:#143d1a;
    classDef model fill:#ffe8a3,stroke:#d48806,stroke-width:2px,color:#5a3d00;
    classDef loss fill:#ffd6d6,stroke:#c0392b,stroke-width:2px,color:#5c1212;
    class RAW,TOK,IDS data;
    class BASE,CKPT,SFT,PREF,RL,CHAT model;
    class CE loss;
```

在基础层面，LLM 是一个条件概率模型：

\[
p_\theta(x_1, x_2, \ldots, x_T)
= \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
\]

Transformer 并不会直接学习“真相”。它学习的是下一个 token 上的概率分布。大量有用的行为之所以会出现，是因为下一个 token 的任务迫使模型将语法、事实、格式、风格和推理轨迹压缩进其权重。

## 本仓库中每个概念的位置

| 概念 | 为什么重要 | 主要代码 |
|---|---|---|
| 分词 | 文本必须先变成整数 id，模型才能训练。 | `scripts/prepare_pretrain_data.py`、`src/post_training/chat_template.py` |
| 固定上下文窗口 | 训练样本是连续的 token 窗口。 | `data_loader/data_loader.py` |
| token 与位置嵌入 | 整数 id 变成向量和位置信息。 | `src/models/transformer.py` |
| 因果自注意力 | 每个 token 只能混合先前 token 的信息。 | `src/models/attention.py` |
| Transformer block | 注意力加 MLP，使用 pre-norm 残差结构。 | `src/models/transformer_block.py` |
| Logits | 隐藏状态变为未归一化的词表分数。 | `src/models/transformer.py` |
| 交叉熵 | 基础目标函数奖励真实下一个 token 上的概率。 | `src/models/transformer.py`、`src/post_training/sft.py` |
| AdamW 与 LR 调度 | 决定训练是否稳定的优化细节。 | `src/post_training/optim.py` |
| 梯度累积 | 在显存受限时模拟更大的 batch。 | `scripts/pretrain_base.py`、`scripts/train_sft.py` |
| 生成 | 模型将采样出的 token 回馈给自身。 | `src/models/transformer.py`、`src/post_training/inference.py` |

## 学习路径

请按以下顺序阅读：

1. [分词与数据形状](tokenization_zh.md)：文本如何成为 batch。
2. [仅解码器 Transformer](transformer_zh.md)：模型骨架。
3. [注意力、掩码与头](attention_zh.md)：核心运算。
4. [目标函数、损失与困惑度](objectives_zh.md)：模型被优化去做什么。
5. [优化与训练系统](optimization_zh.md)：训练循环如何保持稳定。
6. [生成与采样](generation_zh.md)：logits 如何变成文本。

然后继续阅读流水线页面：

- [数据处理](../01_data_pipeline_zh.md)
- [预训练](../02_pretraining_zh.md)
- [SFT](../03_sft_zh.md)
- [奖励模型](../04_reward_model_zh.md)
- [DPO / ORPO / KTO](../05_dpo_zh.md)
- [PPO](../06_ppo_zh.md)
- [GRPO / RLVR](../07_grpo_zh.md)

## 心智模型

基础训练循环十分紧凑：

\[
\text{文本} \to \text{token id} \to \text{嵌入} \to \text{Transformer blocks}
\to \text{logits} \to \text{交叉熵} \to \nabla_\theta
\]

后训练主要改变数据和损失：

- SFT 保留下一个 token 预测，但将损失掩码限制在助手 token 上。
- 奖励建模将输出从词表 logits 改为一个标量分数。
- DPO 比较 chosen 和 rejected 回复的序列对数概率。
- PPO 和 GRPO 会采样补全、为其评分，并用受约束的 RL 目标更新策略。

同一个主干会贯穿复用。这是本仓库最重要的设计理念。

## 主要参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 提出了 Transformer 和缩放点积注意力。
- [The Pile](https://arxiv.org/abs/2101.00027) 描述了预训练路径所使用的 825 GiB 文本语料库。
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) 提出了现代分词器所基于的 BPE 子词思想。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) 解释了 AdamW 的动机。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) 给出了经典的 SFT → 奖励模型 → PPO RLHF 配方。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) 是 DPO 页面背后的理论依据。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) 在数学推理场景中提出了 GRPO。
