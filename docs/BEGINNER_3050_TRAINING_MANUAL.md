# MiniMind 新手训练手册（RTX 3050 台式机）

> 目标人群：0 基础、第一次接触大模型训练。


## 0.5 进阶学习总路线（新增）

如果你希望从“会跑命令”进阶到“系统掌握 LLM 理论 + 实验能力”，请继续阅读：

- `docs/LLM_ZERO_TO_ADVANCED_CURRICULUM_CN.md`

这份路线图是按 9 个月设计的“边学边干”计划，覆盖数学基础、机器学习、深度学习、Transformer、LLM 对齐和毕业项目。

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

## 1.1 从“原理”看各训练阶段到底在做什么

你可以把训练一个对话模型想象成“培养一个新人助手”：

- **预训练（Pretrain）= 先让它大量阅读，建立常识与语言能力**
- **监督微调（SFT）= 再教它按指令说人话、按格式回答**
- **偏好学习（DPO/RL）= 再教它什么答案更符合人类偏好**

下面按 MiniMind 的阶段逐个解释（尽量不用公式）。

### A. Tokenizer（分词器）阶段：把文字变成模型能吃的“编号”

对应脚本：`trainer/train_tokenizer.py`（项目也明确说一般不建议重训）。

模型本质只认识数字，不认识中文字符串。
所以“今天下雨了”会先被切成 token，再映射成 id，比如 `[231, 98, 640, ...]`。

**这一阶段做的事**：

1. 统计语料里高频片段（字、词、子词）
2. 形成一个“词表”（vocab）
3. 约定文本到 id 的编码规则

**对你的意义**：

- 分词器决定模型“看世界的颗粒度”。
- 一旦模型开始训练，分词器最好固定；中途改词表会导致前后权重不兼容。

### B. 预训练（Pretrain）阶段：学“语言本体能力”

对应脚本：`trainer/train_pretrain.py`。

预训练本质是**下一词预测**：给模型一句话的前半段，让它猜后一个 token。

**模型在这个阶段学到什么**：

- 基本语法和语言流畅性
- 常见事实模式（不是严格知识库）
- 长程依赖（前后文关系）

**为什么它重要**：

- 没有预训练，后面的 SFT 像“没读书先做面试题”，很难学好。
- 预训练越扎实，SFT 收敛越快、生成越自然。

### C. 监督微调（SFT）阶段：学“如何按人类要求回答”

对应脚本：`trainer/train_full_sft.py` / `trainer/train_lora.py`。

SFT 使用的是“指令-回答”样本：

- 输入：用户问题（instruction）
- 目标：高质量参考回答（response）

**这一阶段做的事**：

- 把预训练得到的“会说话”能力，变成“会按要求说话”。
- 强化格式习惯（分点、礼貌、角色设定、步骤化表达）。

**Full SFT vs LoRA 的原理区别**：

- Full SFT：更新几乎全部参数，效果潜力高，但显存开销大。
- LoRA：冻结原模型，只训练少量低秩增量矩阵，显存省很多。

对 3050 来说，LoRA 通常是更现实的第一选择。

### D. 偏好学习（DPO）阶段：学“两个答案里哪个更好”

对应脚本：`trainer/train_dpo.py`。

DPO 的数据通常是一组偏好对：

- 同一个问题下，`chosen`（更优） vs `rejected`（较差）

**这一阶段做的事**：

- 让模型提高“偏好答案”的概率，降低“差答案”的概率。
- 不靠在线环境打分，更多是离线偏好学习。

**你能感受到的变化**：

- 回答更像“人类愿意接受的风格”
- 但事实正确性不一定自动提升（数据质量决定上限）

### E. 强化学习（PPO/GRPO/SPO）阶段：学“按奖励函数优化行为”

对应脚本：`trainer/train_ppo.py`、`trainer/train_grpo.py`、`trainer/train_spo.py`。

可以理解为：模型先回答，再依据奖励信号调整策略。

**这一阶段做的事**：

- 直接优化“高奖励行为”（如更有用、更合规、更符合格式）
- 通过 KL 等约束避免模型偏离基座太远

**为什么门槛更高**：

- 超参数多，训练容易不稳定
- 显存和工程复杂度普遍高于 SFT/DPO

所以 0 基础建议：先把 pretrain + SFT + eval 跑稳，再碰 RL。

### F. 蒸馏（Distillation）阶段：把“老师模型能力”压缩给“小模型”

对应脚本：`trainer/train_distillation.py`。

核心思想：让小模型模仿大模型输出分布/中间行为。

**好处**：

- 在较小参数量下拿到更好的效果
- 对个人设备更友好

### G. 评测（Eval）阶段：检查你学到的是不是“真的能力”

对应脚本：`eval_llm.py`。

评测不是走流程，而是防止“看起来在学，实际在退化”。

至少看三类指标：

1. **训练指标**：loss 是否稳定下降
2. **功能样例**：问答、摘要、改写是否可用
3. **坏例对比**：和上一个 checkpoint 对比是否真的变好

---

一句话总结：

- **Pretrain** 决定“会不会说”；
- **SFT/LoRA** 决定“会不会按你想要的方式说”；
- **DPO/RL** 决定“说得是否更符合人类偏好与目标”。

你真正要掌握的，是每一阶段的“目标函数”与“代价（显存、稳定性、数据要求）”之间的平衡。

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
