# 目标函数、损失与困惑度

架构定义模型能够计算什么；目标函数定义训练会奖励什么行为。

对于仅解码器语言模型，基础目标函数是预测下一个 token。

## 从 logits 到概率

在位置 \(t\)，模型输出 logits：

\[
z_t \in \mathbb{R}^{V}
\]

Softmax 将 logits 变为分布：

\[
p_\theta(x_{t+1}=i \mid x_{\leq t})
= \frac{\exp(z_{t,i})}{\sum_{j=1}^{V}\exp(z_{t,j})}
\]

目标是一个整数 token id \(y_t\)。该位置的交叉熵为：

\[
\ell_t = -\log p_\theta(y_t \mid x_{\leq t})
\]

批次损失是所有位置的平均值：

\[
\mathcal{L}_{\text{LM}}
= -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta(y_{b,t} \mid x_{b,\leq t})
\]

## 移位

模型接收：

\[
x = [t_0,t_1,\ldots,t_{T-1}]
\]

并预测：

\[
y = [t_1,t_2,\ldots,t_T]
\]

在 `src/models/transformer.py` 中，简单的 `forward` 路径会在全部位置上计算交叉熵：

```python
logits, loss = model(idx, targets)
flat_logits = logits.view(B * T, C)
targets = targets.view(B * T).long()
loss = F.cross_entropy(flat_logits, targets)
```

后训练的 SFT 路径将移位显式化，因为它需要掩码：

```python
logits = logits[:, :-1, :]
targets = tokens[:, 1:]
mask = loss_mask[:, 1:].to(logits.dtype)
```

## 困惑度

困惑度是取指数后的交叉熵：

\[
\text{PPL} = \exp(\mathcal{L})
\]

如果模型为真实的下一个 token 分配高概率，loss 会下降，perplexity 也会下降。

解释：

- loss 接近 \(\log(V)\) 表明模型接近均匀随机猜测；
- 较低 loss 表明模型正将概率集中到合理的下一个 token；
- 对泛化而言，验证损失比训练损失更重要。

当 `V = 50304` 时：

\[
\log(V) \approx 10.83
\]

因此，未训练模型通常从接近该值的地方开始。

## SFT 带掩码损失

SFT 仍使用下一个 token 交叉熵，但只计入助手补全 token：

\[
\mathcal{L}_{\text{SFT}} =
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}
{\sum_{b,t} m_{b,t}}
\]

其中，助手 token 的 \(m_{b,t}=1\)，提示词 token 的 \(m_{b,t}=0\)。

实现在 `src/post_training/sft.py` 中：

```python
ce = F.cross_entropy(
    logits.reshape(-1, V).float(),
    targets.reshape(-1).long(),
    reduction="none",
)
ce = ce.view(targets.shape) * mask
return ce.sum() / mask.sum().clamp(min=1.0)
```

这种区别至关重要。模型应学习如何回答提示词，而不是学习预测提示词本身。

## 序列对数概率

偏好优化和 RL 需要整个回复的对数概率，而不只是单个 token。

对于提示词 \(p\) 之后的回复 token \(a_1,\ldots,a_L\)：

\[
\log \pi_\theta(a \mid p)
= \sum_{t=1}^{L}
\log \pi_\theta(a_t \mid p, a_{<t})
\]

本仓库在 `src/post_training/rollout.py` 中通过 `sequence_logprobs` 计算它。该函数应用回复掩码，并在答案位置对 token 对数概率求和。

这一基础原语会被以下部分复用：

- DPO、ORPO 和 KTO；
- PPO 策略比率；
- GRPO 策略比率；
- 相对于冻结参考模型的 KL 测量。

## DPO 目标函数

DPO 使用偏好对：chosen 回复 \(y_w\) 与 rejected 回复 \(y_l\)。它将策略与冻结参考模型进行比较：

\[
\Delta_\pi =
\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)
\]

\[
\Delta_{\text{ref}} =
\log \pi_{\text{ref}}(y_w \mid x) - \log \pi_{\text{ref}}(y_l \mid x)
\]

\[
\mathcal{L}_{\text{DPO}} =
-\log \sigma\left(\beta(\Delta_\pi - \Delta_{\text{ref}})\right)
\]

在 `src/post_training/dpo.py` 中：

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits).mean()
```

直觉上：让 chosen 回复比 rejected 回复更可能，但相对参考模型衡量这一改变，使策略不会不受约束地漂移。

## 一图理解 PPO 目标函数

PPO 采样回复、为其评分，并使用新旧动作概率的比率更新策略：

\[
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\text{old}}(a_t \mid s_t)}
= \exp(\log \pi_\theta - \log \pi_{\text{old}})
\]

裁剪策略目标函数为：

\[
\mathcal{L}_{\text{PPO}} =
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon) A_t
\right)
\right]
\]

本仓库在 `src/post_training/ppo.py` 中实现它：

```python
ratio = torch.exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
loss = -masked_mean(torch.min(surr1, surr2), mask)
```

裁剪可防止一次更新偏离产生样本的策略过远。

## 一图理解 GRPO 目标函数

GRPO 不使用学习得到的价值函数。对每个提示词，它采样一组回复，并在该组内归一化其奖励：

\[
A_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}
{\text{std}(r_1,\ldots,r_G)+\epsilon}
\]

这个优势意味着：“在相同提示词下，这个答案比它的同组答案更好还是更差？”

在 `src/post_training/grpo.py` 中：

```python
r = rewards.view(-1, group_size)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
```

对于可验证奖励的推理任务，这很有用，因为它移除了 PPO 的 value head 与 critic 训练循环。

## 目标函数比较

| 阶段 | 数据 | 主要信号 | 学习内容 |
|---|---|---|---|
| 预训练 | 原始 token 流 | 下一个 token CE | 语言建模 |
| SFT | prompt / answer 样本 | 带掩码的下一个 token CE | 指令遵循格式 |
| 奖励模型 | chosen / rejected 对 | Bradley-Terry 偏好损失 | 标量偏好评分 |
| DPO | chosen / rejected 对 | 序列对数概率偏好损失 | 无 RL rollout 的偏好对齐 |
| PPO | 采样回复 + 奖励 | 裁剪策略梯度 | 在 KL 控制下追求奖励的行为 |
| GRPO | 分组采样回复 + 验证器 | 组相对裁剪策略梯度 | 无 critic 的验证器驱动推理 |

## 下一步

损失给出方向；优化决定模型能否可靠地遵循这个方向。继续阅读[优化与训练系统](optimization_zh.md)。
