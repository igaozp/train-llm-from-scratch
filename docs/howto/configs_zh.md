# 使用 JSON 配置

每个训练阶段都由 [`configs/`](https://github.com/FareedKhan-dev/train-llm-from-scratch/tree/main/configs) 下一个便于人工编辑的小型 JSON 文件配置。你只需编辑一个文件，就能看到该阶段的所有可调参数；模型架构和运行时配置则位于共享的 `configs/base.json`。

```
configs/
  base.json        # 共享模型 + 运行时配置（vocab、n_embed、device、amp_dtype、...）
  pretrain.json    # 每个阶段一个文件——只包含该阶段的超参数
  sft.json  reward.json  dpo.json  ppo.json  grpo.json
  smoke/           # 用于快速测试的小型 CPU 变体（小模型 + 少量步骤）
    base.json  pretrain.json  sft.json  ...
```

## 配置如何解析

[`config/loader.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/loader.py) 合并四层配置，**优先级从低到高**：

```
dataclass 默认值  <  configs/base.json  <  configs/<stage>.json  <  CLI --field 覆盖
```

`config/post_training_config.py` 中的 dataclass 是带类型的模式；JSON 是可编辑来源；CLI 适合快速一次性修改并拥有最高优先级。JSON 的 `null` 会映射为 Python 的 `None`（用于设置 `amp_dtype` 这类 `str | None` 字段的简洁方式）。

!!! tip "Smoke 配置会自动缩小模型"
    子文件夹中的阶段 JSON 会使用该文件夹的 `base.json`。因此 `configs/smoke/sft.json` 会自动使用 `configs/smoke/base.json`（小模型、CPU），很适合快速的端到端测试。

## 编辑与查看

=== "编辑 JSON"

    ```jsonc
    // configs/sft.json
    {
      "pretrained_ckpt": "/ephemeral/ckpts/base_pretrained.pt",
      "data_path": "/ephemeral/data/sft_packed.h5",
      "lr": 1e-5,
      "epochs": 3,
      "batch_size": 16
    }
    ```

=== "在 CLI 中覆盖"

    ```bash
    # --field 的优先级高于 JSON
    python scripts/train_sft.py --config configs/sft.json --lr 2e-5 --batch_size 8
    ```

=== "打印解析后的配置"

    ```bash
    python scripts/train_sft.py --config configs/sft.json --print-config
    # 将完整合并后的配置以 JSON 输出，然后退出
    ```

每个训练器都接受 `--config <path>`（默认使用其阶段文件）、所有 `--field` 覆盖项，以及 `--print-config`。旧预训练路径（`scripts/train_transformer.py` + `config/config.py`）保持未修改，仍可按 README 的方式工作。
