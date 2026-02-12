# MiniMind 新手训练手册（RTX 3050 台式机）

> 目标人群：0 基础、第一次接触大模型训练。

## 1. 先理解项目结构（你要训练什么）

MiniMind 把完整流程拆成多个脚本：

- `trainer/train_pretrain.py`：预训练（学通用知识）。
- `trainer/train_full_sft.py`：全参数指令微调（学“对话方式”）。
- `trainer/train_lora.py`：LoRA 微调（更省显存，适合个人显卡）。
- `eval_llm.py`：测试模型输出效果。

如果你是 3050（通常 8GB 显存），建议学习顺序：

1) 跑通推理（确认环境正确）
2) 跑通 SFT 或 LoRA（先做“会说话”）
3) 再尝试预训练（成本更高，放后面）

## 2. 你的硬件现实（3050 训练策略）

3050 显存偏小，核心策略只有三条：

- **减小 `batch_size`**（例如 1~4）。
- **增大 `accumulation_steps`**（例如 8~32）。
- **减小 `max_seq_len`**（例如 128~256 起步）。

你不用一开始追求“快”，先追求“能稳定跑完一轮”。

## 3. 上机前准备清单（Windows 台式机）

### 3.1 必装软件

- NVIDIA 驱动（最新稳定版）
- CUDA 对应版本（或直接用带 CUDA 的 PyTorch）
- Python 3.10+（建议 3.10 或 3.11）
- Git

### 3.2 创建虚拟环境并安装依赖

在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 3.3 检查 GPU 是否可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

输出里有 `True` 和 `RTX 3050` 就可以继续。

## 4. 第一次跑通：最小闭环（推荐）

## 第 0 步：准备数据

把数据放到 `dataset/` 目录。推荐先用小数据：

- 预训练：`pretrain_hq.jsonl`
- 指令微调：`sft_mini_512.jsonl`

## 第 1 步：先跑预训练（轻量参数）

进入 `trainer` 目录执行（推荐先把规模调小）：

```bash
python train_pretrain.py \
  --batch_size 2 \
  --accumulation_steps 16 \
  --max_seq_len 128 \
  --num_workers 2 \
  --dtype float16
```

如果爆显存：继续把 `--batch_size` 降到 1，或把 `--max_seq_len` 降到 96。

## 第 2 步：再跑 SFT（让回答更像助手）

```bash
python train_full_sft.py \
  --batch_size 2 \
  --accumulation_steps 16 \
  --max_seq_len 128 \
  --num_workers 2 \
  --dtype float16 \
  --from_weight pretrain
```

## 第 3 步：验证效果

回到项目根目录：

```bash
python eval_llm.py --weight full_sft
```

## 5. 断点续训（非常重要）

训练过程中断电/崩溃很常见，MiniMind 支持续训：

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
```

你可以把这两条当“保险开关”。

## 6. 你应该怎么学习这个项目（4 周路线图）

### 第 1 周：只做“跑通”

- 完成环境安装、CUDA 可用检查。
- 跑通 `eval_llm.py`（先推理后训练）。
- 跑通一次缩小参数的 `train_full_sft.py`。

目标：你知道“从命令到输出文件”全过程。

### 第 2 周：理解参数和日志

重点理解 5 个参数：

- `batch_size`
- `accumulation_steps`
- `max_seq_len`
- `learning_rate`
- `from_resume`

并观察 loss 曲线变化，记录每次改动是否更稳定。

### 第 3 周：做一次可复现实验

固定随机种子和参数，完整跑一次：

- pretrain → full_sft → eval

输出一份你的实验记录（参数、耗时、显存、结果样例）。

### 第 4 周：上 LoRA 和小领域数据

尝试 `train_lora.py` 在你自己的小数据上做迁移。

目标：你能回答“为什么 LoRA 更适合 3050”。

## 7. 3050 常见问题与解决

### 7.1 CUDA out of memory

顺序处理：

1) `batch_size` 减半
2) `max_seq_len` 降低
3) `num_workers` 降到 0~2
4) 确保没开多余程序（浏览器/游戏）

### 7.2 训练速度很慢

- 个人卡正常现象。
- 先用小数据和短序列验证流程，再放大。
- 你也可以先只练 SFT，不做完整预训练。

### 7.3 结果看起来“会说但不准”

这是小模型常见现象：

- 需要更高质量数据；
- 需要更长训练；
- 需要更合理超参。

先保证稳定，再追求效果。

## 8. 给 0 基础的行动建议（最实用）

- **每天固定 45~90 分钟**，只做一件事：跑通、记录、复现。
- **每次只改一个参数**，否则你不知道效果来自哪里。
- **建立训练日志表**（日期、命令、显存、loss、样例输出）。

你最先要掌握的不是“最强模型”，而是“可复现训练流程”。

---

如果你希望，我可以在下一步直接给你一份“RTX 3050 一键训练脚本”（包含分阶段参数模板：保守/标准/激进三档）。
