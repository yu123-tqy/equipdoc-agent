# EquipDoc-Agent

[![CI](https://github.com/yu123-tqy/equipdoc-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yu123-tqy/equipdoc-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Portfolio License](https://img.shields.io/badge/license-portfolio%20review-lightgrey)](LICENSE)

面向机电设备运维场景的可审核、可降级 Agent。项目将确定性安全策略、LangGraph 人工审核、轴承振动信号诊断工具、可选 RAG 和证据化报告组织为一个可复现的公开工程作品。

> 当前公开版本默认运行在无模型 Demo 模式：可以完整演示 Agent 工作流，但故障类型是明确标注的固定案例回放，不能作为真实设备诊断结果。Full 模式需要单独配置 Qwen 服务、CNN 权重和可选向量库。

![EquipDoc-Agent Demo 首页](docs/assets/demo-overview.png)

## 30 秒了解项目

| 项目维度 | 当前实现 |
|---|---|
| 目标场景 | 轴承振动信号辅助分析与运维报告生成 |
| Agent 编排 | LangGraph 状态图、条件路由、可恢复中断；P2.2 可选受限规划与结构化证据闭环 |
| 人机协同 | 诊断工具执行前必须 Approve/Reject |
| 工具能力 | `.npy` 信号校验、CNN 诊断接口、知识检索、报告生成 |
| 安全边界 | 上传沙箱、大小与数值检查、路径白名单、显式降级 |
| 运行方式 | 本地 Gradio、Docker、AutoDL Full 模式 |
| 当前证据 | 167 项单元测试、三次 13-turn 面试 Demo、56-case / 64-turn 正式评测、失败演进和限制说明 |

## 为什么需要这个 Agent

通用大模型无法直接分析振动时序信号，也容易在设备信息不足时生成缺少依据的维修建议。本项目把自然语言交互与专用诊断工具分离，并在高影响工具调用前加入人工审核：

```mermaid
flowchart LR
    A[上传振动信号] --> B[安全校验]
    B --> C[策略判断任务边界]
    C --> D{人工审核}
    D -->|Approve| E[轴承诊断工具]
    D -->|Reject| F[终止工具调用]
    E --> G[检索故障机理与维护依据]
    G --> H[生成带证据与边界的报告]
```

项目不控制真实设备，不替代工程师作出高风险维修决策，也不根据单段信号预测精确剩余寿命。

## 核心能力

- **可审核工作流**：使用 LangGraph interrupt/resume，在工具执行前展示调用参数并等待审批；
- **确定性安全策略**：信号和诊断意图同时满足时才进入诊断分支，关键判断不依赖 LLM 自由发挥；
- **受限信号工具**：只接受沙箱内、受限大小、有限数值的 `.npy` 一维信号；
- **显式降级**：缺少 Qwen、CNN 权重或 Chroma 时不静默伪装，Demo 模式和词法检索会明确标注；
- **证据化输出**：报告区分输入事实、工具结果、检索依据、建议和适用边界；
- **可选 Agentic 链路**：Full 模式下可启用结构化意图规划、三工具白名单、观察后决策、主动澄清和短期任务记忆；
- **可检索历史会话**：侧栏支持标题/正文搜索、分页、新建、切换、重命名、归档、恢复和确认后永久删除，重启后继续保留完整消息、Top 5 快照与待审核任务；
- **多用户与生产存储**：可选 Gradio 登录，服务端按登录账号隔离所有会话、消息和运行状态；本地默认 SQLite，部署时可切换 PostgreSQL；
- **长会话压缩**：用户可见历史始终完整保存，Agent 每隔固定轮数换代 checkpoint，并通过“旧消息摘要 + 最近消息 + 结构化任务记忆”延续上下文；
- **可复现工程骨架**：包含 `pyproject.toml`、环境变量、Docker、健康检查、Smoke Test 和 CI。

未配置登录时使用 `local` 命名空间，适合个人演示；公共部署应配置 `EQUIPDOC_AUTH_USERS` 并使用 PostgreSQL。Gradio 基础登录适合作品演示，不等同于企业 SSO、细粒度 RBAC 或审计平台。

## 演示结果

诊断工具调用前，系统会暂停工作流并等待人工审核。审核界面只展示工具名称和经过脱敏的文件名，不暴露服务器内部路径。

![EquipDoc-Agent 人工审核工具调用](docs/assets/demo-review.png)

审批通过后，系统输出带 Demo 标识、检索证据、处理建议和适用边界的报告：

![EquipDoc-Agent Demo 诊断报告](docs/assets/demo-report.png)

## 快速运行 Demo

### 1. 环境要求

- Python 3.10、3.11 或 3.12；
- Demo 模式不需要 GPU、Qwen 模型、CNN 权重或向量库。

### 2. 创建环境并安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[demo]"
Copy-Item .env.example .env
```

Linux / AutoDL：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[demo]"
cp .env.example .env
```

### 3. 健康检查与测试

```bash
python -m equipdoc_agent.health --strict
python scripts/demo_smoke.py
python -m unittest discover -s tests -v
```

Demo 模式下，`bearing_model`、`bearing_norm` 和 `rag_vector_db` 可以显示为不存在，因为它们不是必需项。

### 4. 启动页面

```bash
python app_gradio.py
```

打开终端显示的本地地址，默认是：

```text
http://127.0.0.1:7860
```

若 `7860` 已被占用，应用会自动尝试 `7861`、`7862` 等端口。

### 5. 体验审核流程

1. 保持“使用仓库内置演示信号”为勾选状态；
2. 使用默认问题“请诊断这段轴承振动信号，并给出判断依据和处理建议”；
3. 点击“提交”，查看待审核的工具名称和参数；
4. 点击 `Approve` 继续生成报告，或点击 `Reject` 验证拒绝分支；
5. Demo 报告会明确说明结果是固定案例，不是真实模型推理。

### 6. 体验历史会话第二期

1. 连续提交两轮问题，确认同一会话显示此前全部消息；
2. 新建另一会话，再从左侧列表切回第一段会话；
3. 用“搜索历史”检索标题或消息正文，并测试分页；
4. 归档会话，勾选“包含归档”后恢复；
5. 只有勾选“确认永久删除”后才可永久删除，会话消息和对应 Agent checkpoint 会一起清理；
6. 重启应用，确认未删除的历史和待审核状态仍存在。

完整演示脚本见 [`docs/conversation-history-demo-runbook.md`](docs/conversation-history-demo-runbook.md)。

## 运行模式

| 模式 | 用途 | 必需资源 | 输出边界 |
|---|---|---|---|
| Demo | 公开仓库复现、工作流演示 | Demo 依赖、内置信号 | 固定故障案例，明确标注 |
| Full P2 baseline | 复现已发布 P2 真实模型评测 | Qwen 服务、CNN 权重，可选向量库 | 规则路由，Qwen 选择证据句 ID |
| Full P2.1 Agentic | 结构化规划与多工具实验 | 与 Full 相同 | Qwen 参与规划、观察后决策和证据化综合；仍受确定性安全门约束 |
| Full P2.2 Agentic | 面试 Demo 质量与延迟优化 | 与 Full 相同 | 槽位化证据选择、结构化回答、逐句引用和可审计生成路径 |

## Docker Demo

```bash
docker compose up --build
```

浏览器打开 `http://127.0.0.1:7860`。Docker 镜像只包含 Demo 所需依赖，不包含7B模型和 Torch。

## AutoDL Full 模式

### 1. 安装完整依赖

```bash
pip install -e ".[demo,ml,rag]"
```

### 2. 准备本地模型文件

把文件放在配置指定位置，不要提交 GitHub：

```text
models/bearing_cnn.pth
data/processed/norm.npy
```

### 3. 配置服务

在 `.env` 中修改：

```dotenv
EQUIPDOC_DEMO_MODE=false
EQUIPDOC_AGENTIC_MODE=false
EQUIPDOC_LLM_BASE_URL=http://127.0.0.1:8000/v1
EQUIPDOC_LLM_MODEL=qwen-equipdoc
EQUIPDOC_LLM_API_KEY=EMPTY
```

Qwen 服务应提供 OpenAI-compatible `/chat/completions` 接口。模型继续保留在 AutoDL，不应把大模型权重上传 GitHub。

Full 模式的知识问答会先做设备/故障聚焦检索，再按问题意图重排并去除单一切片冗余。P2.2 将多子问题拆成机理、信号特征、现场复核和安全边界等证据槽位，最终答案与 `doc_id#chunk_id` 引用由系统按已校验证据结构化组织；不合格的模型草稿不会直接展示。

若要进入 P2.1 Agentic 链路，显式设置：

```dotenv
EQUIPDOC_DEMO_MODE=false
EQUIPDOC_AGENTIC_MODE=true
EQUIPDOC_AGENT_MAX_STEPS=3
```

P2.1/P2.2 使用“严格 JSON Prompt → 本地 Schema/白名单校验 → 系统执行工具”，不是原生 Function Calling。模型可以选择知识检索、只读信号检查和轴承诊断三个工具；只有诊断工具需要 Approve/Reject。规划不合格时会明确进入确定性路由；回答草稿不满足引用和 grounded 约束时，P2.2 使用已通过槽位检查的结构化证据答案，不展示未验证草稿。

2026-07-29 的 RTX 4090 实测基线完成20/20次真实模型调用：严格自动通过14/20，平均必需关键词召回91.25%，引用原文逐字匹配率100%，一次预热后的串行端到端 p50/p95 为0.414/0.433秒。结果同时暴露了多子问题证据选择不完整和严格关键词门槛假阴性，不能解释为人工正确率或工业诊断准确率。完整报告见 [`docs/p2-full-evaluation-report.md`](docs/p2-full-evaluation-report.md)，复现步骤见 [`docs/p2-autodl-full-evaluation.md`](docs/p2-autodl-full-evaluation.md)。

2026-08-01 的 P2.1 正式评测在冻结的56-case / 64-turn集上完成：最终自动合同通过64/64，平均 / p50 / p95 端到端延迟为7.472 / 8.541 / 16.758秒。52个规划turn中25个模型计划被首轮或重试接受、27个使用确定性fallback；38个证据回答全部使用`extractive_fallback`。完整失败演进为57/64 → 63/64 → 64/64，正式人工复核工作簿已生成但当前为0/64。因此这些数字不能解释为人工回答正确率或工业诊断准确率。完整报告见 [`docs/p2-1-agentic-evaluation-report.md`](docs/p2-1-agentic-evaluation-report.md)。

同日完成的 P2.2 在不放宽冻结合同的前提下优化证据槽位、结构化回答和无效重试：13-turn 面试 Demo 连续三次均为13/13，最终正式回归仍为64/64，平均 / p50 / p95 延迟降至5.100 / 5.889 / 9.926秒；38个证据回答均为`structured_evidence_answer`，答案层`safe_fallback`为0。52个规划turn仍有26个确定性fallback，Demo人工复核当前为0/13，因此不能把自动通过率解释为人工正确率或模型规划准确率。完整报告见 [`docs/p2-2-demo-quality-evaluation-report.md`](docs/p2-2-demo-quality-evaluation-report.md)。

随后针对 Full Gradio 现场暴露的 JSON 计划与 `signal_file` 依赖问题增加了规划热修复：两个指定演示问题连续三轮均不再显示规划降级，当前共有130项单元测试；热修复代码上的13-turn Demo 为13/13，但64-turn正式回归为63/64，仍有1个诊断回答未通过引用与 Grounded Guard。该结果不能替代上面的 Fix 3 归档快照，也不能把当前代码描述为64/64；本次按用户决定保留这一已知回归并直接进行 Git 收口。

P2.2 收口后还完成了本地 Gradio 页面级验证，覆盖知识问题与样例信号的路由隔离、Approve/Reject、移除信号后的状态清理、审核载荷隐私和剩余寿命安全抢占；验证过程中发现的三个 UI/路由问题均已修复并加入回归测试。记录见 [`docs/p2-2-gradio-demo-validation.md`](docs/p2-2-gradio-demo-validation.md)。

### 4. 可选：构建向量库

```bash
python scripts/build_rag_index.py
```

没有向量库时，系统会降级到词法检索，并在健康信息中说明 Dense Retrieval 未启用。

## 项目结构

```text
equipdoc-agent/
├─ app_gradio.py               # Gradio 演示入口
├─ pyproject.toml              # 包、依赖和工具配置
├─ src/equipdoc_agent/
│  ├─ agent/                   # LangGraph、策略和报告
│  ├─ tools/                   # 安全信号诊断工具
│  ├─ rag/                     # Dense/词法检索与降级
│  ├─ models/                  # CNN 结构定义
│  ├─ conversation_store.py    # 双后端会话库、迁移、隔离、搜索与压缩
│  ├─ persistence.py           # LangGraph SQLite/PostgreSQL checkpoint
│  ├─ config.py
│  └─ health.py
├─ data/
│  ├─ samples/                 # 可公开演示信号
│  ├─ knowledge/               # 当前知识笔记
│  └─ eval/                    # 评测输入
├─ tests/                      # 不依赖大模型的基础测试
├─ scripts/                    # Smoke Test、索引和历史脚本
├─ docs/                       # 架构、迁移说明与展示素材
└─ artifacts/legacy/          # 原 AutoDL 历史证据
```

详细设计见 [`docs/architecture.md`](docs/architecture.md)，迁移说明见 [`docs/migration-notes.md`](docs/migration-notes.md)。

## 配置说明

完整示例见 [`.env.example`](.env.example)。

| 变量 | 用途 | 安全默认值 |
|---|---|---|
| `EQUIPDOC_DEMO_MODE` | 是否使用无模型固定案例 | `true` |
| `EQUIPDOC_AGENTIC_MODE` | 是否在 Full 模式启用 P2.1 受限规划 | `false` |
| `EQUIPDOC_AGENT_MAX_STEPS` | P2.1 单轮最大工具步数，限制为1～4 | `3` |
| `EQUIPDOC_LLM_BASE_URL` | OpenAI-compatible 服务地址 | 本机8000端口 |
| `EQUIPDOC_BEARING_MODEL_PATH` | CNN 权重路径 | `models/bearing_cnn.pth` |
| `EQUIPDOC_UPLOAD_ROOT` | 上传沙箱 | `runtime/uploads` |
| `EQUIPDOC_CONVERSATION_DB_PATH` | 历史会话与消息数据库 | `runtime/equipdoc_conversations.sqlite3` |
| `EQUIPDOC_CHECKPOINT_DB_PATH` | LangGraph 持久化 checkpoint | `runtime/langgraph_checkpoints.sqlite3` |
| `EQUIPDOC_DATABASE_URL` | 可选 PostgreSQL 会话库 URL；非空时覆盖 SQLite 路径 | 空 |
| `EQUIPDOC_CHECKPOINT_DATABASE_URL` | 可选 PostgreSQL checkpoint URL | 空 |
| `EQUIPDOC_AUTH_USERS` | 可选登录账号，格式 `user:password,user2:password2` | 空（本地单用户） |
| `EQUIPDOC_HISTORY_PAGE_SIZE` | 历史列表每页数量，限制为5～50 | `12` |
| `EQUIPDOC_CONVERSATION_RETENTION_DAYS` | 已归档会话自动清理天数 | `90` |
| `EQUIPDOC_RECENT_CONTEXT_MESSAGES` | 长会话压缩后保留的最近消息数 | `8` |
| `EQUIPDOC_MEMORY_COMPACTION_TURNS` | Agent checkpoint 换代轮数 | `12` |
| `EQUIPDOC_MAX_UPLOAD_MB` | 上传大小限制 | `8` |
| `EQUIPDOC_UPLOAD_TTL_HOURS` | 应用暂存信号保留时长 | `24` |
| `EQUIPDOC_RAG_DB_DIR` | Chroma 目录 | `vector_db/chroma_equipdoc` |
| `EQUIPDOC_EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-small-zh-v1.5` |

## 测试与持续集成

本地测试：

```bash
python -m unittest discover -s tests -v
```

当前本地实现包含167项 `unittest`，覆盖旧 Demo/P2 回归以及 P2.1/P2.2 规划校验、系统信号依赖归一化、未知依赖拒绝、三工具权限、人工审核、最大步数、多轮记忆、逐句引用、槽位化证据覆盖、知识检索锚定、历史会话持久化、用户隔离、正文搜索与分页、归档/恢复/永久删除、一期数据库原地迁移、并发运行唯一性、长会话摘要与 checkpoint 换代、SQLite/PostgreSQL 后端选择、checkpoint 重启恢复、取消任务防止迟到结果写入、安全策略优先级、隐私与运行时清理、索引清单和正式评测集合同。

`.github/workflows/ci.yml` 会在 GitHub 上使用 Python 3.10、3.11 和 3.12 自动运行单元测试、健康检查和 Demo Smoke Test。

P1 还把 Agent 与 RAG 指标设为回归门槛，防止后续改动静默降低当前基线：

```bash
python scripts/build_knowledge_chunks.py --check
python scripts/eval_agent_workflow.py --min-case-pass-rate 1.00
python scripts/eval_rag_retrieval.py --min-hit-at-5 0.90 --min-mrr-at-10 0.75
python scripts/eval_safety_grounding.py --min-case-pass-rate 1.00
```

## 评测证据与适用边界

当前可复现的 P1 基线：

| 模块 | 结果 | 口径 |
|---|---:|---|
| Agent 工作流 | 30 条总通过率 100% | 无模型 Demo；确定性路由、知识覆盖与人工审核流程 |
| 高风险边界 | 20 条固定用例通过率 100% | 确定性规则、引用有效性与抽取证据一致性 |
| RAG 检索 | Hit@5 91.0%，MRR@10 76.8% | 100 条旧测试；14篇知识文档；文档级相关性 |
| Qwen Full 模式 | 严格通过14/20；关键词召回91.25%；p95 0.433秒 | RTX 4090；BM25；模型选择证据ID；引用原文匹配100%；非人工正确率 |
| P2.1 Agentic | 正式自动合同通过64/64；p95 16.758秒 | RTX 4090；真实 Qwen + CNN；52个规划turn中25个模型计划被接受、27个确定性fallback；38个证据回答全部抽取式fallback；人工复核0/64 |
| P2.2 Agentic | 三次Demo均13/13；正式自动合同64/64；p95 9.926秒 | RTX 4090；真实 Qwen + CNN；38个结构化证据回答；26/52确定性规划fallback；Demo人工复核0/13 |
| P2.2 Live规划热修复 | Demo 13/13；正式自动合同63/64；p95 8.954秒 | 指定Full页面问题已无规划降级；正式集仍有1个诊断引用/Guard回归，尚未定位 |
| CNN | 暂不报告准确率 | 旧数据不具备可信文件级 Group Split 条件 |

P1 原始口径见 [`docs/evaluation-report.md`](docs/evaluation-report.md)，P1.2 安全与证据评测见 [`docs/p1-2-safety-grounding-report.md`](docs/p1-2-safety-grounding-report.md)，P2 真实模型报告见 [`docs/p2-full-evaluation-report.md`](docs/p2-full-evaluation-report.md)，P2.1 Smoke、正式评测和失败演进见 [`docs/p2-1-agentic-evaluation-report.md`](docs/p2-1-agentic-evaluation-report.md)，P2.2 三次 Demo、最终正式回归和限制见 [`docs/p2-2-demo-quality-evaluation-report.md`](docs/p2-2-demo-quality-evaluation-report.md)，P2.1 正式评测合同与执行记录见 [`docs/p2-1-formal-evaluation-plan.md`](docs/p2-1-formal-evaluation-plan.md)，后续本地/AutoDL 操作见 [`docs/p1-autodl-runbook.md`](docs/p1-autodl-runbook.md)。

`artifacts/legacy/` 保存原 AutoDL 结果，用于保留实验链路，不作为最终性能结论：

- 30条 Agent 评测主要验证规则路由和审核分支；
- FP16结果只有9次串行请求，且没有记录GPU型号；
- BNB 4-bit结果是单问题测试；
- 旧CNN随机窗口拆分存在同源数据泄漏风险；
- 100条 RAG 测试集存在，但当前仓库没有原实验的最终 RAG 输出。

在完成跨工况 Group Split、人工 groundedness 审查和可复现实验之前，本项目不宣称“CNN准确率100%”“工具路由100%”或未经复核的“幻觉降低率”。

## 安全与公开边界

- UI只接受上传文件或内置样例，不接受服务器路径输入；
- 上传文件被复制到 `runtime/uploads` 并使用随机文件名，过期暂存文件会自动清理；
- 登录启用后，用户归属只取自服务端 `gr.Request.username`，不接受浏览器传入 owner id；所有会话、消息、运行和管理操作都再次校验 owner；
- 永久删除采用显式确认，并同步删除该会话历次 Agent checkpoint；归档数据按配置的保留期清理；
- 工具只允许读取 `data/samples` 和 `runtime/uploads`；
- 仅接受受限大小的数值型一维 `.npy`；
- 向量索引必须通过切片哈希、Embedding 模型和集合名清单校验，陈旧索引会禁用而不是静默使用；
- Qwen 服务非回环地址默认要求 Bearer Token，GPU 推理并发数默认限制为1；
- Demo 标签不能删除，避免固定案例被误解为真实推理；
- 真实单位代码、内部手册、客户数据、模型权重和密钥不得进入仓库；
- 当前知识库为项目笔记，正式评测前仍需补充权威来源和版本信息。

## Roadmap

下一阶段聚焦人工质量复核和 P2.2 剩余稳定性问题：

1. 使用 `artifacts/p2_2/demo_human_review.xlsx` 完成13个面试 Demo turn 的人工复核，并保留 P2.1 正式集0/64的未审状态；
2. 先定位当前 Live 热修复代码的1个正式诊断回归，再分别报告 Fix 3 快照与 Live 代码的首轮、重试和确定性路由分布；
3. 提升带引用的模型自然综合通过率，避免把38个结构化证据答案误解为自由生成已稳定；
4. 优化 `knowledge_qa` 延迟，其当前 p50 为7.055秒，仍高于6秒目标；
5. 保留旧 CNN 数据泄漏限制，后续按原始文件和工况进行 Group Split；
6. 为知识库补充权威来源和版本信息，人工复核完成后再更新简历质量指标。

## License

本仓库使用作品展示许可，允许招聘、教育和个人作品评审。第三方模型、数据集、文档和商标仍遵循各自许可，详见 [`LICENSE`](LICENSE) 与 [`NOTICE.md`](NOTICE.md)。
