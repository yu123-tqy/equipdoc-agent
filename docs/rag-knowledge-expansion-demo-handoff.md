# EquipDoc-Agent RAG 知识库扩展与演示交接文档

> 更新时间：2026-08-08
>
> 发布分支：`main`
>
> GitHub：https://github.com/yu123-tqy/equipdoc-agent.git
>
> 本地仓库示例：`<本地工作目录>/equipdoc-agent`
>
> AutoDL 演示仓库示例：`/root/autodl-tmp/equipdoc-agent-final-demo`

## 1. 交接结论

本轮工作已经完成以下目标：

1. 扩充轴承运维知识库，并将“吊舱推进器推力轴承故障诊断方案”和“试验台合同/技术协议”加工为可直接 Embedding、可写入 Chroma 的结构化知识切片。
2. 明确资料冲突时以实验方案为主，合同技术资料作为补充依据。
3. 改善项目参数、轴承型号、转速和故障原因/处理建议等问题的召回与回答组织。
4. 放宽可修复的模型规划输出，减少不必要的“规划降级”。
5. 在页面中增加“查看本次召回 Top 5”，可直接查看本轮实际使用的前五段检索内容。
6. 增加“停止当前任务”，输入错误时可停止本轮并立即换问题。
7. 将知识问答调整为“直接回答 → 补充依据 → 已知边界”，使演示结果更易读。

当前知识库构建结果为 **58 篇知识文档、426 个切片**。当前功能适合作为可审计的项目演示，不应表述为已经达到工业生产部署或能自动替代现场工程师。

## 2. 项目做什么

EquipDoc-Agent 是一个面向机电设备运维的受约束 Agent，主要包含四类能力：

- 知识问答：从轴承知识库、试验方案和试验台资料中检索证据并组织回答。
- 信号检查：读取 `.npy` 振动信号，输出采样点数、均值、标准差、RMS、峰值等统计量。
- 轴承诊断：调用既有 CNN 模型给出故障类别和置信度，诊断工具需要人工 Approve/Reject。
- Agent 工作流：由本地 Qwen 规划工具调用，程序负责 Schema、白名单、参数、路径、最大步数、引用和安全边界校验。

核心原则是：**模型负责理解与受约束规划，程序负责验证、授权、证据组织和失败兜底。**

## 3. 本轮修改总览

| 提交 | 修改内容 | 结果 |
|---|---|---|
| `db020ec` | 扩充轴承知识，导入两份项目资料 | 建成 58 文档、426 切片的项目知识库 |
| `442a1f7` | 增加 Top 5 召回查看功能 | 页面可展开本轮真实检索快照 |
| `f80938a` | 放宽可修复规划输出 | 减少格式性问题造成的确定性规划降级 |
| `6f63772` | 优先直接回答参数类问题 | 转速、型号等问题先给明确结论 |
| `99b5125` | 修复滚动体故障知识问答 | 原因与处理类问题召回更准确 |
| `77af5d2` | 强制补足处理建议并增加停止任务 | 回答覆盖原因和建议，页面支持取消 |

## 4. RAG 知识库新增内容

### 4.1 通用轴承知识

本轮补充了适合知识检索的轴承知识，覆盖但不限于：

- 轴承分类：深沟球、角接触球、调心球、圆柱滚子、圆锥滚子、调心滚子、推力球、推力滚子、滚针和滑动轴承等。
- 轴承选型：载荷方向、载荷大小、转速、精度、刚度、调心能力、安装空间、润滑、温度和寿命等。
- 典型故障：内圈、外圈、滚动体、保持架、润滑、污染、腐蚀、磨损、点蚀、剥落、裂纹、塑性变形和电蚀等。
- 故障机理：过载、冲击、安装偏斜、游隙异常、配合不当、润滑不足、污染、温升、轴电流和长期交变载荷等。
- 信号特征：时域冲击、RMS、峰值、峭度、包络谱以及 BPFO、BPFI、BSF、FTF 等特征频率关系。
- 现场复核：转速、载荷、温度、润滑、安装、配合、游隙、噪声和历史趋势等检查项。
- 处理建议：清洁与润滑检查、载荷与冲击排查、对中与配合复核、保持架检查、表面损伤评估及必要时更换。
- 安全边界：在缺少原始信号、完整工况或人工检查时，不编造精确寿命、维修工单、故障位置或确定性结论。

针对最后发现的“只回答滚动体故障原因、不回答怎么处理”，知识切片和回答槽位均已补强：

- `bearing_ball_fault_c001`：滚动体故障机理和常见原因。
- `bearing_ball_fault_c002`：典型信号特征。
- `bearing_ball_fault_c003`：润滑、温度、载荷冲击、安装状态、金属磨屑和是否换轴承等处理建议。

### 4.2 两份项目资料的处理

本轮处理了用户提供的两份 Word 文档：

1. `吊舱推进器推力轴承故障诊断方案.docx`
2. `试验台合同.docx`

处理时保留了文档标题、一级/二级/三级标题、章节编号、段落层级、表格语义和图片对应关系，没有把整份 Word 简单转成一段纯文本。

主要加工结果：

- 实验方案按章节拆分为 `pod_thrust_bearing_plan_ch1` 至 `pod_thrust_bearing_plan_ch6` 等知识文档。
- 合同技术内容整理为 `test_rig_technical_spec`。
- 图片资产保存到 `data/knowledge_assets/` 下，并在文本中保留对应说明或位置关系。
- 表格按行列语义转为 Markdown，尽量保留型号、参数、编号、单位和备注。
- 通过 chunk override/source anchor 文件固定重要参数段落，避免关键数值被错误切开。
- 为每个切片保存 `doc_id`、`chunk_id`、标题、章节、来源类型和来源优先级等元数据。

项目资料说明见：

- `docs/rag_project_sources.md`
- `docs/rag_knowledge_source_catalog.md`
- `docs/rag_knowledge_coverage_matrix.md`

### 4.3 资料冲突优先级

用户已经明确：**所有疑点以实验方案为主。** 当前实现采用以下优先级：

| 来源 | 角色 | `source_priority` |
|---|---|---:|
| 吊舱推进器推力轴承故障诊断方案 | 主要事实来源 | 100 |
| 试验台合同/技术协议 | 补充与交叉验证 | 70 |
| 项目自编通用知识 | 通用说明与运维建议 | 40（按文档配置） |

例如试验台转速问题：

- 实验方案给出的试验台设计范围是 **0～2000 rpm**，作为直接答案。
- 合同资料中的电机调速或常用运行范围可作为补充，但不能覆盖实验方案的设计指标。

### 4.4 切片策略与 Chroma 兼容性

当前切片策略为：

- 目标 chunk：约 420 个字符。
- 最大 chunk：约 500 个字符。
- overlap：约 80 个字符。
- 优先按标题、章节、段落和表格边界切分。
- 重要表格行、参数说明和跨段语义通过 override/anchor 保持完整。
- 最终输出为 `data/knowledge_chunks.jsonl`，每行一个 JSON 对象，适合批量生成向量。
- 向量库为 Chroma，集合名为 `equipdoc_rag`。
- 当前 Embedding 模型为本地 `bge-small-zh-v1.5`，输出维度 512。

切片不是单纯固定长度截断。标题和章节会进入检索文本，原始层级则保存在元数据中，用于检索加权、引用和 Top 5 展示。

## 5. 检索与回答链路如何实现

### 5.1 检索流程

当前 RAG 不是单一向量相似度，主要流程为：

1. 对用户问题进行中文词法/关键词分析。
2. 使用本地 BGE 生成查询向量，从 Chroma 获取密集检索候选。
3. 同时进行词法检索，强化型号、转速、故障部位和专业缩写等精确词。
4. 通过 RRF 等方式融合词法与向量排名。
5. 进行来源优先级、文档多样性和问题焦点重排。
6. 保存本轮 Top 5 检索快照，供回答和页面展开查看。

项目参数问题会强化“转速、型号、范围、多少”等参数词；故障问答会识别内圈、外圈、滚动体和保持架等对象，并在“原因 + 怎么处理”问题中同时保留机理切片和处理建议切片。

### 5.2 回答组织

原页面曾直接显示“补充依据”，对普通用户不够自然。本轮改为：

1. **直接回答**：先针对问题给结论或可执行说明。
2. **补充依据**：列出支持结论的文档事实和引用。
3. **已知边界**：说明缺少哪些现场信息，以及哪些结论不能直接推断。

参数类问题优先抽取数值和单位；故障类问题按“原因/表现/处理建议”槽位检查覆盖度。如果生成结果缺少用户明确要求的“怎么处理”，系统会补入有证据支持的维护建议，而不是只返回故障机理。

### 5.3 规划降级修复

此前 Qwen 两次没有返回完全符合严格 Schema 的计划时，系统会显示：

> 规划降级：模型两次未返回合格计划，本轮使用确定性安全路由。

问题主要是校验过严：模型已经表达了正确意图和工具，但可能带 Markdown 代码块、字段别名、额外说明或局部可修复字段，仍被整体判为失败。

本轮修改后：

- 接受可识别的 JSON 包裹和常见字段形式。
- 对可安全推断的字段做规范化。
- 校验并修复工具名、参数结构和计划步骤。
- 只有两轮输出都无法安全修复时才进入确定性 fallback。
- 工具白名单、路径边界、最大步骤数和人工审核要求没有放松。

因此放宽的是**格式容错**，不是执行权限。

## 6. 页面新增功能

### 6.1 查看本次召回 Top 5

回答下方新增“查看本次召回 Top 5”。展开后会显示：

- 排名；
- 文档标题和 `doc_id`；
- `chunk_id`；
- 原始章节；
- RRF、词法、向量和来源优先级等排序信息；
- 该切片的完整文本。

这里展示的是**生成本次回答时保存的检索快照**，点击按钮不会重新检索，因此能真实说明回答当时召回了什么。

### 6.2 停止当前任务

提交后页面新增“停止当前任务”：

- 空闲时不可点击，任务执行时启用。
- 点击后取消本轮 Submit/Approve/Reject 事件。
- 清理本轮 thread、回答、待审核工具和 Top 5 状态。
- 保留问题输入框，用户可以修改后立即重新提交。
- Gradio 并发设置允许停止后开始新的问题。

边界说明：如果底层同步 Qwen 请求已经发送，Python/显卡上的旧请求可能继续计算到结束，但结果会被丢弃，不再覆盖当前页面。该功能实现的是可靠的交互取消和状态隔离，不是强制中断 GPU 内核。

## 7. 本轮验证情况

本轮修改完成时已执行：

- Python 单元测试：147 项通过。
- Ruff 代码检查通过。
- 知识构建检查：58 篇文档、426 个切片。
- 本地 BGE 离线加载验证：`embedding_shape=(1, 512)`。
- Chroma 索引构建成功：集合 `equipdoc_rag`。
- AutoDL 严格健康检查：`ready=true`、`mode=full_agentic`。
- Qwen OpenAI-compatible 服务检查：模型 `qwen-equipdoc` 可列出并返回 `READY`。
- 浏览器人工验证：项目型号、项目说明、0～2000 rpm、滚动体故障原因/建议、Top 5 和停止按钮均已展示。

最后一次滚动体问题截图中的结果已经符合本轮目标：直接回答包含原因和处理建议，Top 5 的前两条分别命中机理与处理建议切片，第三条补充通用维修决策。

## 8. 后续日常启动演示：完整流程

以下流程以当前 AutoDL 环境为准。建议始终使用三个终端：

| 终端 | 作用 | 是否保持运行 |
|---|---|---|
| 终端 A | Qwen 模型服务（8001） | 是 |
| 终端 B | 检查、同步、排错 | 否 |
| 终端 C | Gradio 页面（7860） | 是 |

### 8.1 启动 AutoDL 实例

1. 在 AutoDL 控制台启动原来的 RTX 4090 实例。
2. 等待实例状态变为运行中。
3. 使用 AutoDL 网页终端，或者使用本地 SSH 连接。
4. 进入仓库时必须使用 `cd`，不能直接执行目录：

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
```

### 8.2 同步 GitHub 最新代码

在终端 B 执行：

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate

git status --short --branch
git branch --show-current
git fetch origin
git pull --ff-only origin main
git log -3 --oneline
```

正常结果应满足：

- 分支为 `main`。
- HEAD 与 `origin/main` 一致。
- `artifacts/p2/service_check_latest.json` 等运行产物即使是 untracked，也不影响 `--ff-only` 拉取。

如果出现已跟踪代码文件被修改，不要使用 `git reset --hard`。先执行 `git status`，确认修改来源，再决定备份、提交或恢复。

### 8.3 激活环境并核对资源

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate

ls -lh /root/autodl-tmp/equipdoc-agent/models/bearing_cnn.pth
ls -lh /root/autodl-tmp/equipdoc-agent/data/processed/norm.npy
du -sh /root/autodl-tmp/models_llm/Qwen2.5-7B-Instruct-EquipDoc
du -sh /root/autodl-tmp/models_embedding/bge-small-zh-v1.5
```

当前部署已验证的路径：

```text
CNN:       /root/autodl-tmp/equipdoc-agent/models/bearing_cnn.pth
Norm:      /root/autodl-tmp/equipdoc-agent/data/processed/norm.npy
Qwen:      /root/autodl-tmp/models_llm/Qwen2.5-7B-Instruct-EquipDoc
Embedding: /root/autodl-tmp/models_embedding/bge-small-zh-v1.5
```

### 8.4 核对 `.env`

`.env` 至少应包含以下配置：

```dotenv
EQUIPDOC_DEMO_MODE=false
EQUIPDOC_AGENTIC_MODE=true
EQUIPDOC_AGENT_MAX_STEPS=3

EQUIPDOC_LLM_BASE_URL=http://127.0.0.1:8001/v1
EQUIPDOC_LLM_MODEL=qwen-equipdoc
EQUIPDOC_LLM_API_KEY=EMPTY
EQUIPDOC_LLM_TIMEOUT_SECONDS=120

EQUIPDOC_BEARING_MODEL_PATH=/root/autodl-tmp/equipdoc-agent/models/bearing_cnn.pth
EQUIPDOC_BEARING_NORM_PATH=/root/autodl-tmp/equipdoc-agent/data/processed/norm.npy

EQUIPDOC_RAG_ENABLED=true
EQUIPDOC_RAG_CHUNKS_PATH=data/knowledge_chunks.jsonl
EQUIPDOC_RAG_DB_DIR=vector_db/chroma_equipdoc
EQUIPDOC_RAG_COLLECTION=equipdoc_rag
EQUIPDOC_EMBEDDING_MODEL=/root/autodl-tmp/models_embedding/bge-small-zh-v1.5
EQUIPDOC_RAG_TOP_K=5

EQUIPDOC_SERVER_HOST=0.0.0.0
EQUIPDOC_SERVER_PORT=7860
EQUIPDOC_GRADIO_SHARE=false
```

可用下面命令快速检查：

```bash
grep -E '^EQUIPDOC_(DEMO_MODE|AGENTIC_MODE|LLM_BASE_URL|LLM_MODEL|BEARING_MODEL_PATH|BEARING_NORM_PATH|EMBEDDING_MODEL|RAG_DB_DIR|SERVER_PORT)=' .env
```

不要把 `.env`、模型权重、向量数据库或服务器运行文件提交到 Git。

### 8.5 判断是否需要重建 RAG 索引

**普通重启演示不需要每次重建。** 只有以下情况才需要执行：

- `data/knowledge_chunks.jsonl` 更新；
- 新增或修改知识文档后重新生成了切片；
- 更换 Embedding 模型；
- 健康检查提示索引清单与当前知识不匹配；
- 向量库目录损坏或丢失。

需要重建时，在终端 B 执行：

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/build_rag_index.py --reset
```

预期输出：

```text
Built 426 chunks in .../vector_db/chroma_equipdoc
Collection: equipdoc_rag
Embedding: /root/autodl-tmp/models_embedding/bge-small-zh-v1.5
Manifest: equipdoc_index_manifest.json
```

### 8.6 终端 A：启动 Qwen 服务

先检查服务是否已经运行：

```bash
curl -s http://127.0.0.1:8001/v1/models
```

如果能够返回 `qwen-equipdoc`，无需重复启动。若连接失败，执行：

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/serve_qwen_openai.py \
  --model-path /root/autodl-tmp/models_llm/Qwen2.5-7B-Instruct-EquipDoc \
  --served-model-name qwen-equipdoc \
  --host 127.0.0.1 \
  --port 8001
```

看到以下信息表示成功：

```text
Model ready: http://127.0.0.1:8001/v1
Uvicorn running on http://127.0.0.1:8001
```

这个终端必须保持运行。日志中的 generation flags 警告不影响当前服务可用性。

### 8.7 终端 B：服务和健康检查

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate

python scripts/check_full_service.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen-equipdoc \
  --output artifacts/p2/service_check_latest.json

python -m equipdoc_agent.health --strict
```

预期结果：

- `check_full_service.py` 返回 `ready: true`。
- `health --strict` 返回 `ready: true` 和 `mode: full_agentic`。
- `sample_signal`、`bearing_model`、`bearing_norm`、`rag_chunks`、`rag_vector_db` 和 `llm_configuration` 均正常。

健康检查只验证配置与文件；Qwen 的实际连通性由 `check_full_service.py` 验证。

### 8.8 终端 C：启动 Gradio 页面

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate
python app_gradio.py
```

看到服务监听 `0.0.0.0:7860` 后保持该终端运行。

如果之前已有旧页面进程，应先在旧终端按 `Ctrl+C`，再启动新版本。启动后浏览器使用 `Ctrl+F5` 强制刷新，避免仍显示旧按钮或旧 JavaScript。

### 8.9 从本地浏览器访问

优先使用 AutoDL 控制台的自定义服务/端口映射暴露 `7860`，然后点击生成的访问地址。

如果使用 SSH 端口转发，在本地 PowerShell 新开终端，按 AutoDL 提供的 SSH 主机和端口执行：

```powershell
ssh -N -L 7860:127.0.0.1:7860 root@<AutoDL-SSH-Host> -p <SSH-Port>
```

保持该 PowerShell 运行，然后打开：

```text
http://127.0.0.1:7860
```

8001 只供 AutoDL 内部的 Agent 调用，不需要暴露到公网。

## 9. 推荐现场演示顺序

### 9.1 知识问答演示

知识问答时取消勾选“使用仓库内置演示信号”，不要上传 `.npy`。

建议依次提问：

1. `结合故障诊断方案和试验台合同，说明该项目如何完成推力轴承故障模拟、信号采集和诊断验证。`
2. `吊舱推进器推力轴承故障诊断试验台的设计转速范围是多少？请给出文档依据。`
3. `试验台使用的轴承型号是什么？请给出文档依据。`
4. `滚动体发生故障的原因是什么，应该怎么处理？`

每题都可以展开“查看本次召回 Top 5”，重点讲解：

- 回答中的 `doc_id#chunk_id` 可以追溯到知识切片；
- 0～2000 rpm 为什么优先于合同中的其他转速描述；
- 滚动体问题为什么同时召回原因切片和处理建议切片；
- Top 5 是本轮快照，不是点击后重新检索。

### 9.2 停止功能演示

1. 输入一个问题并点击“提交”。
2. 在模型生成期间点击“停止当前任务”。
3. 修改输入框中的问题。
4. 重新点击“提交”。

说明：停止后旧请求可能在后台完成计算，但不会覆盖新任务页面。

### 9.3 轴承诊断演示

1. 勾选“使用仓库内置演示信号”，或上传合法 `.npy`。
2. 输入：`请诊断当前轴承信号，并结合知识库说明故障类型、诊断依据、现场复核方法和处理建议。`
3. 点击“提交”。
4. 页面出现待审核工具时，解释诊断属于高影响操作，所以必须人工 Approve/Reject。
5. 点击 Approve，展示 CNN 直接输出、信号统计、知识依据和安全边界。

不要把单次 CNN 输出或 `60.87%` 置信度表述为模型总体准确率。

### 9.4 安全边界演示

可提问：

```text
请根据这一段信号给出轴承精确剩余寿命，并自动生成维修工单。
```

系统应拒绝编造精确寿命或执行真实维修动作，同时给出需要补充的监测、工况和人工检查信息。

## 10. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `...equipdoc-agent-final-demo: Is a directory` | 把目录当命令执行 | 使用 `cd /root/autodl-tmp/equipdoc-agent-final-demo` |
| Qwen 连接拒绝 | 8001 服务未启动 | 在终端 A 启动 `serve_qwen_openai.py` |
| Qwen 启动后终端不能继续输入 | 服务需要占用前台 | 保持终端 A，另开终端 B/C |
| 页面没有停止按钮或 Top 5 | 旧 Gradio 进程或浏览器缓存 | 拉取最新分支、重启 Gradio、`Ctrl+F5` |
| 回答仍是旧知识 | 代码已更新但索引仍旧 | 确认 chunks 有变化后执行 `build_rag_index.py --reset` |
| BGE 尝试联网 | 未启用离线模式或路径错误 | 使用本地模型绝对路径和两个 `*_OFFLINE=1` |
| `address already in use` | 8001 或 7860 已有进程 | 先确认是否是可复用的旧服务；必要时停止对应进程后重启 |
| `service_check_latest.json` 未跟踪 | 运行检查生成的本地产物 | 可保留，不影响演示和 `git pull --ff-only` |
| `git pull` 被本地修改阻止 | AutoDL 有已跟踪文件修改 | `git status` 查明来源；不要直接 hard reset |
| 仍出现规划降级 | 两轮计划均无法安全修复或服务异常 | 先检查 Qwen 日志；确定性路由是安全兜底，不代表整轮失败 |
| 回答缺少现场结论 | 问题需要信号/工况而当前只有知识文档 | 上传信号并补充转速、载荷、温度、润滑和安装信息 |

## 11. 正常结束与节省费用

演示结束后：

1. 在终端 C 按 `Ctrl+C` 停止 Gradio。
2. 在终端 A 按 `Ctrl+C` 停止 Qwen 服务并释放显存。
3. 确认没有仍需保存的日志或测试结果。
4. 回到 AutoDL 控制台停止实例，避免继续计费。

不需要删除模型、向量库或虚拟环境；下次启动可直接复用。

## 12. 后续继续扩充知识库的标准流程

后续新增资料时按以下顺序进行：

1. 保存原始资料，只读核对版本、来源、日期和适用设备。
2. 转为结构化 Markdown，保留标题、章节、编号、段落、表格和图片说明。
3. 为文档设置稳定 `doc_id`、来源类型、来源优先级和适用范围。
4. 对关键参数、型号、阈值和安全条款设置 source anchor/override。
5. 生成并校验 `data/knowledge_chunks.jsonl`。
6. 运行知识库一致性测试和检索回归。
7. 在 AutoDL 使用同一 BGE 模型重建 Chroma。
8. 用“直接问题 + 同义表达 + 冲突来源 + 边界问题”进行人工复核。
9. 提交源文档加工结果、chunks、测试和说明；不要提交 Chroma 目录或模型权重。

新增资料若与现有实验方案冲突，不能仅依赖向量相似度决定答案，必须显式设置来源优先级并增加冲突用例。

## 13. 已知边界和后续建议

- 当前知识库已经明显扩展，但通用知识仍以项目整理材料为主，后续应增加标准、教材、厂商手册的版本化来源。
- Top 5 展示能证明召回效果，不能单独证明最终答案事实正确，仍需查看来源质量和回答是否忠实。
- 停止按钮不会强杀已经进入 GPU 的同步生成，只保证旧结果不再写回页面。
- 当前 CNN 只用于项目演示，不能从单次置信度推导现场准确率或剩余寿命。
- 项目没有控制真实设备，也不会自动下发维修工单。
- `.env`、Qwen 权重、CNN 权重、归一化文件、Chroma 数据库和运行上传文件都不应推送到 GitHub。
- Windows 工作区现有三个 `artifacts/p1/*.json` 本地修改属于用户已有内容，本轮没有编辑、暂存或提交。

后续若继续优化，建议优先顺序是：

1. 增加带来源版本和适用范围的权威轴承资料。
2. 为新增知识建立参数冲突、同义问法和多意图回归集。
3. 统计规划首轮接受率、修复成功率和确定性 fallback 率，而不仅看最终回答完成率。
4. 增加回答人工评分，包括问题覆盖度、事实正确性、证据支持度和可操作性。
5. 使用多来源、多工况和 Group Split 数据重新评估 CNN。

## 14. 快速启动命令清单

### 终端 A

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/serve_qwen_openai.py \
  --model-path /root/autodl-tmp/models_llm/Qwen2.5-7B-Instruct-EquipDoc \
  --served-model-name qwen-equipdoc \
  --host 127.0.0.1 \
  --port 8001
```

### 终端 B

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate
python scripts/check_full_service.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen-equipdoc \
  --output artifacts/p2/service_check_latest.json
python -m equipdoc_agent.health --strict
```

### 终端 C

```bash
cd /root/autodl-tmp/equipdoc-agent-final-demo
source .venv/bin/activate
python app_gradio.py
```

浏览器访问 AutoDL 映射后的 7860 地址，即可开始演示。
