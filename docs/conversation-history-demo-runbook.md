# 历史会话两期合并演示脚本

## 启动前

```powershell
cd C:\Users\MSI\Documents\秋招简历\projects\equipdoc-agent-portfolio
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -q
python app_gradio.py
```

终端会给出实际端口；默认访问 `http://127.0.0.1:7860`。

## 3～5 分钟演示顺序

1. 指出左侧是历史会话列表，右侧是当前会话完整消息；系统重启后仍会保留。
2. 在默认会话提交样例信号诊断，展示待审核调用，然后选择 `Approve`。
3. 在同一会话继续问第二个问题，说明本轮页面会同时显示上一轮问答。
4. 点击“新建对话”，提一个带明显关键词的问题，例如“泵站汽蚀如何现场复核”。
5. 切回第一段会话，证明不同会话内容相互独立。
6. 在“搜索历史”输入“泵站”或消息正文中的关键词，展示正文检索；会话较多时使用上一页/下一页。
7. 把第二段会话归档，勾选“包含归档”，再点击“恢复”。
8. 勾选“确认永久删除（不可恢复）”并删除一段测试会话，强调系统同时清理可见历史与对应 Agent checkpoint。
9. 停止并重新启动应用，再打开第一段会话，展示历史仍在。

## 多用户隔离演示（可选）

在 `.env` 中临时配置：

```dotenv
EQUIPDOC_AUTH_USERS=alice:alice-demo,bob:bob-demo
```

重启应用后：

1. 用 `alice` 登录并创建“只属于 Alice”的会话；
2. 访问 `/logout` 退出；
3. 用 `bob` 登录，确认列表中看不到 Alice 的会话；
4. Bob 新建会话后再切回 Alice，确认双方历史分别保留。

演示结束后不要把真实密码提交到 Git；`.env` 已被忽略，只提交 `.env.example`。

## 面试时的技术解释

可以用一句话概括：

> 我把用户可见会话和 LangGraph 内部线程拆成两个 ID；完整历史进关系库，Agent checkpoint 可换代。登录用户由服务端请求绑定 owner，所有查询再次校验 owner；长会话通过旧摘要、最近消息和结构化任务记忆恢复，而不是无限累积原始消息。

如果被追问生产化：默认 SQLite 是本地演示配置；PostgreSQL 模式使用 SQLAlchemy 和官方 PostgresSaver。基础登录仍不是完整企业 IAM，正式上线还要补 SSO、RBAC、审计和密钥管理。
