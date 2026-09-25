# xiaojev（小 Jev）

**面向浏览器操作、RAG、游戏和概率推理的轻量 0.6B 决策模型。**

xiaojev 对动态候选集打分并返回概率分布，供应用选择动作、排序证据和评估不确定性。
模型可在单张 GPU 上本地运行，直接选择给定候选，无需自回归生成文本。

- **浏览器操作**：为搜索、表单填写、筛选和导航选择操作及可见元素目标。
- **RAG 检索**：评估段落相关性，融合重排模型与 dense 检索器的排序。
- **证据判断**：选择支撑段落，判断问题可答性和上下文充分性。
- **游戏策略**：在 Maze、Snake 和 Doom 环境中选择动作。
- **概率推理**：预测硬币、骰子、抽签、抽球等任务的结果分布。

[English README](README.md) · [评测详情](docs/V4_REPAIR.md) · [研究结果](docs/RESULTS.md)

## 评测结果

| 能力 | 评测 | 指标 |
|---|---|---:|
| 浏览器操作 | 本地酒店任务，独立核验最终页面 | **4/4** |
| RAG 检索 | MuSiQue test R@5，101 题，dense + xiaojev 融合 | **77.31%** |
| RAG 问答流程 | MuSiQue test EM / F1，101 题，top-4 上下文 | **36.63% / 46.60%** |
| RAG 可答性门控 | 不可答题幻觉率；reader prompt tokens | **32.7% → 1.0%；−83.7%** |
| 证据判断 | 语义 test 准确率，2,384 条决策 | **83.52%** |
| 游戏策略 | test / OOD 加权宏平均成功率 | **53.26% / 26.72%** |
| 概率推理 | 概率 test 准确率 / 平均 TV，8,145 条决策 | **85.62% / 0.1264** |
| 推理速度 | 单次决策跨域 p50，单张 RTX 3090 | **27.84–69.49 ms** |

浏览器指标使用单独适配的权重，覆盖三个本地 fixture 任务，其中一个重复执行。
训练包含相同页面布局，验收案例参与了权重选择。RAG 检索使用 v4 权重和排名融合，
问答答案由独立的 Qwen reader 生成。可答性门控使用 v3，阈值只在 98 道校准题上选定后冻结；
它以可答题覆盖率为代价（保留 27.7%——瓶颈在 BM25 一阶段的证据完整率，不在门控本身）
换来 33 倍幻觉下降；本地 gold 全可答，不可答样本是移除金标文档的合成构造，
详见[门控报告](rag_eval/GATE_REPORT.md)。其余模型指标使用 v4。TV 越低表示概率校准越好；
延迟统计包含分词的预热后模型推理，不包含浏览器执行或问答 reader。

```bash
# 离线复算 RAG 指标及校准选择，不需要 GPU 或下载模型
python -m rag_eval.evaluate_fusion --verify-calibration
```

[实时 RAG 接口](rag_eval/README.md) · [浏览器安装](integrations/jev-ultrafast/README.md) ·
[浏览器适配训练](experiments/v4_browser/README.md)

## 研究背景

通用语言模型的 token 概率可能与它用语言表达的概率估计存在明显差异。
xiaojev 使用独立评分头和分布监督来学习决策接口：概率任务采用解析真值，
语义与 RAG 任务采用 QA 金标，浏览器任务采用程序化交互标签，游戏任务采用教师或专家策略。
相关探针、校准测量和训练实验保留在[研究报告](docs/RESULTS.md)中。

## 扩容实验：规模定律的作用边界（4B LoRA）

把同一套五域数据、同样 2500 步训练搬到冻结 Qwen3-4B + LoRA rank32（固定末轮 checkpoint）：
所有离线域全面上涨，游戏队列在对方主场打平——**唯独**真实浏览器 fixture 原地不动：

| | 0.6B v4 | 4B LoRA |
|---|---:|---:|
| 概率 test 准确率 | 85.62% | **90.28%** |
| 概率 OOD 准确率（并列修正） | 81.04% | **92.45%** |
| 语义 test 准确率 | 83.52% | **89.18%** |
| RAG 难负例准确率 | 89.80% | **91.69%** |
| 548 例游戏 macro，test / OOD | 53.26% / 26.72% | **65.31% / 41.30%** |
| —— 同一队列参照 | Jev API：65.39% / 43.72% | NanoJev：66.85% / 45.47% |
| 单独重排 R@5，独立 test | 65.35% | 72.03%（dense：73.35%） |
| 真实浏览器 fixture（统一权重） | 0/4 | 0/4 |
| 真实浏览器 fixture（修数据后） | 4/4 | 4/4 |

这是本研究线的**第七条发现——规模定律的作用边界**（第一至六条见
[docs/RESULTS.md](docs/RESULTS.md)）：训练数据与部署对齐的地方，扩容就有效；
对齐缺失的地方，扩容无能为力。浏览器的病根是合成训练状态与真实 DOM 的序列化差异，
不是容量：修好序列化并加入真实状态反事实数据后，0.6B（`ckpt/v4_browser_dom/step100`）
和 4B（`ckpt/qwen3_4b_browser/step50`）都能 4/4。诚实边界：模型规模与微调方式同时改变
（全参数 vs LoRA），增益不能单独归因于容量；fixture 4/4 是同一本地 fixture 的回归验证，
案例参与了部署权重选择，不是独立通用网页基准。完整表格与逐题审计见
[docs/RESULTS.md](docs/RESULTS.md) 第 10–11 节。

## 架构

```
 state + question + K 个候选（动态集合）
          │
          ▼
 chat 模板（单条 user 消息）
          │
          ▼
 K 条候选路径：[prompt][label k][EOS]    label：0-9、A-Z（单 token）
          │
          ▼
 Qwen3-0.6B 主干 —— 一次打包前向（bf16 autocast，fp32 权重）
          │
          ▼
 每条路径 EOS 位置隐状态
          │
          ▼
 LayerNorm → Linear（每路径一个标量分）
          │
          ▼
 组内 softmax ——► 候选上的完整概率分布（不生成文本）
```

训练损失：组内 softmax 之后对目标分布的软标签交叉熵 + 0.1 × Brier。

## Quickstart

```bash
pip install -r requirements.txt
```

### 复现数据

```bash
# 概率题：9 万行、精确解析目标、seed 固定（20260921）
python data/datagen.py --n 90000 --out data/train_v1.jsonl
pytest data/test_datagen.py        # 11 个自洽性测试（暴力枚举 oracle）

# 游戏决策：从 NanoJev-Data 转换（Jev 教师 / 视觉专家软标签）
huggingface-cli download C-Tianyu/NanoJev-Data --repo-type dataset --local-dir /path/to/NanoJev-Data
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data
python data/make_gamedata.py       # -> data/games_v1.jsonl

# 语义决策：2.4 万行，来自 HotpotQA / 2WikiMultiHopQA / MuSiQue 的
# 金标 supporting facts —— 零 LLM 调用，seed 固定（20260922）
export XIAOJEV_RAG_DATA=/path/to/rag_datasets   # hotpotqa/ 2wikimultihopqa/ musique/，
                                                # 各含 raw/<name>.json + gold.jsonl + corpus.jsonl
python data/make_semanticdata.py   # -> data/semantic_v1.jsonl
pytest data/test_semanticdata.py   # 真值映射、split 泄漏、类别均衡、负例过滤
```

浏览器决策数据通过 `python data/make_browserdata.py` 生成，难负例 RAG 数据按
[RAG 数据流程](rag_eval/README.md)准备。训练源和浏览器适配数据的各 100 行样本
已提交在 `data/samples/`，不下载完整数据也能查看格式。

### 训练

```bash
# 概率、游戏、语义、浏览器、RAG 五类数据等权混合
python training/train.py --steps 2500 \
    --mix data/train_v1.jsonl:0.2 --mix data/games_v1.jsonl:0.2 \
    --mix data/semantic_v1.jsonl:0.2 --mix data/browser_v1.jsonl:0.2 \
    --mix data/rag_v1.jsonl:0.2 \
    --out ckpt/v4 --log results/train_log_v4.jsonl
```

主干默认 `Qwen/Qwen3-0.6B`（可用 `XIAOJEV_BASE_MODEL` 覆盖）。

### 评测

```bash
python training/evaluate.py --ckpt ckpt/v4 --split test --output results/eval_v4_test.json
python training/evaluate.py --ckpt ckpt/v4 --split ood  --output results/eval_v4_ood.json
python training/evaluate.py --ckpt ckpt/v4 --split test --data data/semantic_v1.jsonl \
    --output results/eval_v4_semantic_test.json
python training/evaluate.py --zero-shot --split test --data data/semantic_v1.jsonl \
    --output results/eval_zs_semantic_test.json
```

指标：acc / NLL / Brier / ECE(10) / 平均 TV（总体 + 按机制×难度）+ 配对 Choice-vs-Noul 一致性。

### 复现零样本探针

需要本地 vLLM 服务；我们用的是 `cyankiwi/Qwen3.8-27B-AWQ-INT4`：

```bash
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 --port 8020 --served-model-name qwen3.8-27b
export XIAOJEV_VLLM_URL=http://127.0.0.1:8020/v1   # 默认值；XIAOJEV_VLLM_MODEL/XIAOJEV_MODEL_PATH 同理

python probes/probe_distributions.py   # token 通道失真、跨原语矛盾
python probes/probe_verbalized.py      # 语言通道校准
python probes/probe_novel.py           # 全新（不可记忆）机制
python probes/probe_elicitation.py     # 五种诱导方法对比
```

### 评测游戏策略

```bash
export XIAOJEV_NANOJEV_REPO=/path/to/NanoJev        # github.com/TianyuCodings/NanoJev 的克隆
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data

python comparison/compare_sanity.py      # 复核 NanoJev 已发表的 548 例数字
python comparison/compare_rollout.py --engine vcdm --checkpoint ckpt/v4 \
    --output results/compare_v4_frozen_episodes.jsonl
python comparison/compare_report.py results/compare_v4_frozen_episodes.jsonl \
    results/compare_v4_frozen.json
```

（`vcdm` 是 xiaojev checkpoint 的内部代号，为兼容产物文件而保留。Doom 场景的游戏 rollout 另需 `vizdoom`。）

## 浏览器 agent 集成

[`integrations/jev-ultrafast/`](integrations/jev-ultrafast/) 提供本地决策后端、固定上游版本的补丁和安装器。
设置 `JEV_BACKEND=local` 后，xiaojev 根据可见控件及其状态选择操作和目标；
独立的 OpenAI 兼容文本助手仅负责 `TYPE_TEXT` 填写值。候选通过有界微批次打分，零自回归解码。

浏览器权重在两轮本地酒店验收中均完成 4/4，使用 5、5、5、4 个动作，
最终 URL 与筛选条件均独立核验。配置方式见[安装说明](integrations/jev-ultrafast/README.md)，
逐次调用记录见[验收数据](results/v4_repair/browser_final_summary.json)。

## 完整结果

当前模型的指标、权重选择、评测范围及证据见 **[评测报告](docs/V4_REPAIR.md)**。
可答性门控端到端见 **[门控报告](rag_eval/GATE_REPORT.md)**；4B LoRA 扩容实验与 4B 浏览器修复见
[docs/RESULTS.md](docs/RESULTS.md) 第 10–11 节，历史 v1–v3 研究同在其中。
`results/` 包含汇总指标、本地页面轨迹和冻结的文档排名数据。

## Checkpoints

- `ckpt/v4`：概率、游戏、语义、浏览器、难负例 RAG 混合训练权重。
- `ckpt/v4_browser_dom/step100`：浏览器适配权重；本地别名为 `ckpt/v4_browser`。
- `ckpt/qwen3_4b_lora_v1/step2500`：4B LoRA 五域扩容实验权重。
- `ckpt/qwen3_4b_browser/step50`：4B 浏览器适配权重（自扩容 checkpoint 续训）。

本仓库提供训练代码、数据生成入口、权重校验值和评测证据，模型二进制尚未公开托管。
浏览器使用适配权重，其余已评测领域使用 v4。训练与适配命令见[评测报告](docs/V4_REPAIR.md)。

## 环境要求

- 单卡 24GB GPU（开发用一张 RTX 3090）
- 核心代码 Python 3.10+；浏览器集成 Python 3.12+
- `torch` 2.10、`transformers` 4.57、`httpx`、`pytest`；Doom 游戏 rollout 另需 `vizdoom`；27B 探针目标需 vLLM 部署

## 限制

- **研究原型**：小规模权重实验、一张卡、一个训练随机种子，未做大规模超参搜索。
- **8K 上下文**：超出 8192 token 预算的 state 会被截断。
- **域覆盖**：训练覆盖程序概率机制、四个游戏任务、QA 金标构造的语义决策、浏览器交互和难负例 RAG 决策。浏览器适配仅在本地 fixture 上验证，任意网站的可靠性尚未评测。
- **游戏监督是教师蒸馏**（Jev native_probs / 视觉专家策略），不是真值；概率域与语义域有精确目标（解析解 / 金标推导）。
- **与 TypeSafe 无任何关联**：Jev 仅作为基准出现（NanoJev 公开产物与公开 API 回执），xiaojev 是独立的复现研究。

## 致谢

- [NanoJev](https://github.com/TianyuCodings/NanoJev)（MIT）——游戏训练数据（NanoJev-Data）、548 例冻结对比协议与验证器、Jev API 回执。
- [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（MIT）——浏览器 agent，xiaojev 作为其本地决策后端接入（`integrations/jev-ultrafast/`）。
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)（Apache 2.0）——基座（0.6B）；27B 探针目标为 Qwen3.8-27B 的社区 AWQ 量化。
- HotpotQA、2WikiMultiHopQA、MuSiQue —— 语义域监督来源 QA 数据集（各自遵循其许可证）。

## License

MIT —— 见 [LICENSE](LICENSE)。衍生自 NanoJev 与 Qwen3 的数据和权重仍遵循各自许可证。
