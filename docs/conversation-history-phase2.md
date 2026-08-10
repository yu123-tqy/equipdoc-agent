# 历史会话与记忆第二期设计

## 目标与结果

第二期把一期的“本地单用户历史列表”升级为可公开演示、可切换生产数据库的会话系统，同时保持 Agent 的安全门和人工审核语义不变。

当前实现包含：

- 服务端登录身份与 owner 隔离；
- 标题/消息正文搜索、分页、归档筛选和恢复；
- 带确认的永久删除，以及归档数据保留期；
- 一期 SQLite 数据库自动原地迁移；
- SQLite 本地模式和 PostgreSQL 部署模式；
- 同一会话只能存在一个运行中/待审核任务的数据库唯一约束；
- 完整可见历史与有界 Agent 上下文分离；
- 删除会话时同步清理该会话所有 LangGraph checkpoint 线程。

## 数据边界

`conversation_id` 是用户可见会话的稳定主键；`agent_thread_id` 是 LangGraph 内部 checkpoint 主键。两者刻意分开：停止任务、进程恢复或达到长会话压缩阈值时，只换代内部线程，不改变用户在侧栏看到的会话，也不删除其完整消息。

所有读取和写入都在服务端根据 `gr.Request.username` 绑定 `owner_id`。浏览器只保存当前 `conversation_id`、搜索词和页码；即使用户猜到其他人的 UUID，数据库查询仍会因为 owner 不匹配而返回空结果或拒绝操作。

## 长会话策略

历史库始终保存每条用户/助手消息。Agent 上下文采用三层结构：

1. `summary`：较早消息的确定性压缩摘录，限制最大字符数；
2. `recent_messages`：最近若干条消息；
3. `agent_memory`：最近诊断、检索、澄清等受控结构字段。

达到 `EQUIPDOC_MEMORY_COMPACTION_TURNS` 后创建新的 `agent_thread_id`，下一轮把上述结构重新注入新 checkpoint。这样不会让 LangGraph 当前状态随可见历史无限增长，也不需要为了节省上下文而删除用户历史。

## 数据库与迁移

默认配置继续使用：

```dotenv
EQUIPDOC_CONVERSATION_DB_PATH=runtime/equipdoc_conversations.sqlite3
EQUIPDOC_CHECKPOINT_DB_PATH=runtime/langgraph_checkpoints.sqlite3
```

SQLite 使用 WAL、外键、busy timeout 和短连接；运行唯一性与消息序号由数据库约束/原子更新保证。

PostgreSQL 部署安装：

```bash
pip install -e ".[demo,postgres]"
```

配置示例：

```dotenv
EQUIPDOC_DATABASE_URL=postgresql+psycopg://equipdoc:password@127.0.0.1:5432/equipdoc
EQUIPDOC_CHECKPOINT_DATABASE_URL=postgresql://equipdoc:password@127.0.0.1:5432/equipdoc
```

会话库由 SQLAlchemy 管理连接，LangGraph checkpoint 使用官方 `PostgresSaver` 和连接池。首次启动自动建表；已有一期 SQLite 表会补充 `agent_thread_id`、`summary`、`next_sequence_no`、`deleted_at`、线程映射和 schema 版本记录，旧消息不搬迁、不丢失。

上线前仍建议先备份 `runtime/*.sqlite3`，生产 PostgreSQL 使用独立数据库账号、TLS、网络访问控制和常规备份。

## 登录与保留策略

本地演示保持无登录：

```dotenv
EQUIPDOC_AUTH_USERS=
```

多人演示示例：

```dotenv
EQUIPDOC_AUTH_USERS=alice:change-me,bob:change-me-too
EQUIPDOC_CONVERSATION_RETENTION_DAYS=90
```

该登录是 Gradio 基础认证，适合作品展示和受控演示。真正企业部署还应接入 SSO/OIDC、密码哈希与密钥管理、RBAC、审计日志、限流和 CSRF/反向代理安全策略。

归档是可恢复操作；归档超过保留天数后，在该用户首次访问时清理。永久删除要求用户显式勾选确认，会同步删除消息、运行记录、线程映射和 checkpoint，无法恢复。
