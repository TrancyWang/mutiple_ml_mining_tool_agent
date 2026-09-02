# Academic Agent

Academic Agent 是一个面向文本挖掘、机器学习、数据分析、可视化和文档问答的桌面 Agent 应用。客户端使用 PySide6，模型适配层使用 Qwen-Agent，底层能力通过注册工具统一暴露。

本项目当前的核心编排方式是：

```text
Routing（能力路由与意图聚焦）
        ↓
Plan Loop（补齐关键信息并形成可审阅的 Plan）
        ↓
Plan → Dynamic Task Board（把计划转换成带验收标准的任务板）
        ↓
Convergence Loop（逐任务执行、检查、重试、收敛）
        ↓
Deterministic Assembly（只拼装已完成任务的结果）
```

这套设计解决的是“一次生成一大段答案”的几个常见问题：模型过早执行、关键信息靠猜、复杂任务中途失控、任务完成标准不明确，以及失败后只能整段重做。

## 一、系统分层架构（当前实现）

这不是一个“模型直接接管整个应用”的单体 Agent，而是一个由 Qt 客户端、应用控制层、Agent 编排层、模型适配层、能力工具层和基础设施层组成的分层系统。每一层都有明确边界：模型负责理解和生成，程序负责权限、状态、顺序、重试和可验证的完成条件。

### 1. 从界面到基础设施的六层结构

```text
┌──────────────────────────────────────────────────────────────┐
│ Presentation：PySide6 客户端                                  │
│ qt_client.py / views / StreamWorker / 内嵌 Plan 操作卡片       │
└──────────────────────────────┬───────────────────────────────┘
                               │ 用户操作、进度事件、流式结果
┌──────────────────────────────▼───────────────────────────────┐
│ Application Control：应用控制层                               │
│ ApplicationController / AgentController / SessionController   │
│ ProjectController / AuthController / ApplicationState         │
└──────────────────────────────┬───────────────────────────────┘
                               │ AgentRequest、会话和配置
┌──────────────────────────────▼───────────────────────────────┐
│ Agent Orchestration：模型驱动但由程序控状态的编排层            │
│ AgentRuntime / TaskPlanner / Plan Loop / TaskBoard             │
│ TaskVerifier / Reflection / Replan / ExecutionScope            │
└──────────────────────────────┬───────────────────────────────┘
                               │ 规划调用、执行调用、结构化事件
┌──────────────────────────────▼───────────────────────────────┐
│ Model Adapter：模型适配层                                     │
│ AgentService / chat_agent / planning_agent / Provider Config  │
└──────────────────────────────┬───────────────────────────────┘
                               │ 受限工具调用
┌──────────────────────────────▼───────────────────────────────┐
│ Capability：业务能力层                                        │
│ Tool Registry / Tool Handlers / Tools / Services / Algorithms  │
│ RAG 检索、文本挖掘、机器学习、数据分析、可视化、项目操作        │
└──────────────────────────────┬───────────────────────────────┘
                               │ 文件、数据库、模型、向量索引
┌──────────────────────────────▼───────────────────────────────┐
│ Infrastructure：基础设施层                                   │
│ Workspace / Auth Store / SQLite Repositories / Artifacts       │
│ Memory Stores / Data Profiler / Document Generator / Paths     │
└──────────────────────────────────────────────────────────────┘
```

| 层次 | 当前职责 | 主要代码位置 | 明确不负责的事情 |
| --- | --- | --- | --- |
| Presentation | 展示聊天、上传、登录、模式切换、Plan 审批、任务进度和产物 | `qt_client.py`、`academic_agent/views/` | 不决定路由、任务状态或工具权限 |
| Application Control | 组合应用用例，管理当前窗口状态、登录、项目、会话和 Agent 生命周期 | `academic_agent/controllers/`、`academic_agent/models/state.py` | 不把 Qt 控件传入 Agent 核心 |
| Agent Orchestration | 执行 Routing → Plan → Task Board → Convergence，维护状态迁移、审批、重试、Reflection 和有限重规划 | `academic_agent/agent/runtime.py`、`orchestration/`、`planning/`、`types.py` | 不把模型的自然语言当成程序状态 |
| Model Adapter | 管理 Qwen-Agent 和其他 Provider 的配置，区分聊天模型、规划模型和主执行模型 | `academic_agent/agent/adapters/`、`providers/` | 规划模型不读取文件、不调用业务工具 |
| Capability | 暴露可执行的文本挖掘、机器学习、数据分析、可视化、RAG 和工作区操作 | `agent/tooling/`、`tools/`、`application/services/`、`algorithms/`、`rag/` | 不自行绕过任务级工具作用域 |
| Infrastructure | 提供认证存储、SQLite、文件工作区、产物目录、记忆/向量存储、依赖和模型路径 | `academic_agent/infrastructure/`、`repositories/` | 不决定用户意图和 Plan 内容 |

### 2. 一次请求经过哪些对象

```text
qt_client.main
  → CodexChatWindow
  → StreamWorker
  → AgentService.chat_stream
  → AgentRuntime.stream
  → TaskPlanner（本地能力路由）
  → planning_agent（语义 Routing / Plan Review / 拆任务 / 验收 / Reflection）
  → plan_review_callback（执行方式：请求批准 / 帮我批准）
  → TaskBoard + ExecutionScope
  → 主执行 Agent + ToolExecutor + 业务工具
  → TaskVerifier
  → 任务重试 / Reflection / 最多一次 Replan
  → completed 任务结果的确定性组装
  → Qt 进度事件和最终回复
```

这里有一个重要的分支：

- `Work` 模式进入完整的编排链路，允许在审批后使用任务级工具。
- `Chat` 模式设置 `chat_only=true`，走独立的 `chat_agent`，直接进行自然语言对话，不进入 Plan Review、任务板和工具执行。
- `上传文件` 是客户端输入链路，不是模型自己发现文件：用户先选文件，客户端确认登录后调用数据加载/画像，再把文件绑定到 `SessionContext`。
- `登录` 由 `AuthController` 和 `LoginDialog` 管理；访客可以先浏览界面和选择文件，需要真正初始化 Agent 或执行受保护能力时再完成认证。

### 3. 当前系统的核心状态对象

| 对象 | 生命周期 | 保存内容 | 谁可以修改 |
| --- | --- | --- | --- |
| `ApplicationState` | 当前 Qt 窗口 | 登录态、用户、模式、模型 Provider、当前会话 ID、消息和确认过的操作 | Controller 和界面适配层 |
| `SessionContext` | 当前 Agent 会话 | 上传文件、文件画像、确认信息、Plan 文书、Plan trace、任务板和产物 | `SessionStore`、Runtime 的会话同步逻辑 |
| `AgentPlan` | 一次 Work 请求 | 路由、重点、可用工具、预期产物、确认信息、Plan 文书 | Planner/Plan Loop 生成，程序通过 `replace` 更新 |
| `TaskBoard` / `TaskItem` | 一次 Work 请求的执行期 | 任务依赖、状态、尝试次数、反馈、证据、结果 | 程序状态机；模型只能提议任务字段 |
| `ExecutionScope` | 当前执行作用域 | 当前任务 ID、允许工具、工具观察、结构化事件 | Runtime 和 ToolExecutor |
| Repository / Store | 应用持久化或运行期 | 对话、项目、认证、记忆、向量索引、文件产物 | 对应基础设施组件 |

因此，项目的“控制面”和“执行面”是分开的：Plan、任务板、审批、重试和事件属于控制面；主 Agent 调用工具、生成分析结果和写出产物属于执行面。

## 二、核心编排设计

### 1. Routing：先确定能力边界，再判断讨论重点

Routing 分成两个互补部分：

1. `TaskPlanner` 做本地确定性路由。它根据请求把任务归入 `document`、`visualization`、`machine_learning`、`project`、`data` 或 `chat`，并根据技能目录和工具注册表生成允许使用的工具集合。
2. Work 模式下，规划模型再做一次语义 Routing，只负责回答“用户当前最关注什么”，例如结果质量、业务解释、技术实现或可复现性。语义 Routing 不能扩大本地已经确定的工具权限，也不能调用工具。

因此，模型可以理解重点，但不能通过自然语言自行越过项目的能力边界。

### 2. Plan Loop：先补充必要信息，再写计划书

Plan Loop 不生成最终答案，也不执行工具。规划模型每轮检查：

- 用户目标是否清楚；
- 最终交付物是什么；
- 是否缺少会改变方案方向的关键信息；
- 哪些细节可以采用低风险、可逆的默认假设。

在 Work 模式下，如果本地路由判断不出明确的学术方向，程序会先发起一轮确定性的学术意图澄清，只提供文本挖掘、机器学习、数据分析与可视化、学术文献与研究四类选项。这样“帮我分析一下”不会直接进入通用 Chat，也不会让模型自行选择一个无边界的工具流程；Chat 模式则保留普通自然语言问答能力。

如果缺少信息，模型返回 `information_frame`。程序会给每个缺口分配稳定编号，例如 `R1-F1`，每轮最多展示 3 个问题，并跳过已经确认过的问题。Qt 客户端通过交互卡片收集答案，答案会进入当前会话的 `confirmed_information`，后续规划和执行都可以使用。

信息补充现在以内嵌卡片呈现，而不是突然弹出输入框。卡片会同时展示“为什么需要这项信息”、建议回答格式、示例，并把候选答案做成可点击选项；用户不需要猜 Agent 想要什么，也不需要自由输入长文本。

如果信息已经足够，模型返回 `plan_ready=true`，然后单独生成一份自然语言 Markdown Plan。Plan 说明目标、边界、实施顺序和完成条件，但不等于最终产物。

`Plan Review` 是规划模块的内部输出契约，不是给用户填写的业务问题。它只负责判断信息是否足够：要么返回 `plan_ready=true` 和空的 `information_frame`，要么返回真正影响方案方向的业务缺口及可点击选项。如果模型误把 JSON、`information_frame` 或 “Plan Review 输出结构”当成缺口，Runtime 会自动要求模型修复一次；不会把这类内部字段展示成“补充信息”。连续两次仍不符合结构时，Runtime 会保留具体原因，并回退到不新增用户约束的本地安全 Plan，不让一次模型格式波动阻断整条任务链路。

当前 Work 模式会在对话输入区一开始就展示“执行方式”下拉框，位置紧挨模型按钮。用户可以在发送消息前选定本次任务的执行方式；Plan 详情卡片只负责展示目标、任务和完成标准，不再承载“请求批准 / 帮我批准”入口，也不会弹出模态审批窗口：

- `请求批准`：进入人工确认模式。Plan 卡片提供“确认执行计划”，算法配置由用户选择，关键信息缺口由用户点选，失败重规划也会再次等待确认。
- `帮我批准`：进入全自动模式。Plan、信息澄清、算法选择和重规划自动继续，不在这些 Agent 节点停下来等待用户。

文件生成、修改和删除仍保留原有操作确认；“帮我批准”只自动化 Agent 的规划和执行流程，不扩大文件操作权限。

对核心分析方法也采用同样的边界：用户只说“做聚类”时，人工流程会要求选择 KMeans、层次聚类或 DBSCAN；用户只说“做机器学习”时，会先选择回归、分类或因果推断，再选择对应模型/方法。人工请求批准不会默认替用户勾选推荐算法；只有用户主动选择算法确认窗口中的全自动化，或在输入区执行方式下拉框选择“帮我批准”，才使用推荐算法和默认参数。

客户端底部的“上传文件”入口适用于 Work 和 Chat 两种模式。它会先打开单文件选择器，用户选中文件后才检查登录状态；因此未登录时也不会先被登录窗口挡住。选择器支持常见表格、文本、PDF、Word、PPT、图片、音频和视频格式，并提供“所有文件”选项。文件被确认后会绑定到当前会话，后续 Agent 可以把它作为用户明确提供的分析输入。

未登录时，左侧账户区直接显示“登录账户”，不再要求用户从 `···` 菜单里猜入口。登录窗口会根据用户名提示管理员和普通用户分别需要填写的内容，按回车也可以提交登录。

### 3. Plan → Dynamic Task Board：语义拆分，状态由程序接管

确认 Plan 后，规划模型通过第二次独立调用把 Plan 拆成任务清单。每个任务必须包含：

| 字段 | 含义 |
| --- | --- |
| `task_title` | 面向用户的任务名称 |
| `task_goal` | 这一项具体要解决的问题 |
| `deliverable` | 当前任务应该产出的结果槽位 |
| `done_when` | 可以检查的完成条件 |

Plan 阶段的信息缺口必须带有可点击的 `options`，每项包含 `label`、`value` 和可选的 `description`。例如：

```json
{
  "topic": "会议工具",
  "question": "你希望使用哪种线上会议工具？",
  "options": [
    {"label": "腾讯会议", "value": "腾讯会议", "description": "适合当前团队的线上会议"},
    {"label": "飞书会议", "value": "飞书会议", "description": "适合需要协作文档的会议"}
  ]
}
```

模型只负责提供这些语义字段。程序负责补充和维护：

- 稳定任务 ID：`T1`、`T2`、`T3`……；
- 初始状态：`ready`；
- 依赖关系；
- 当前尝试次数 `attempts`；
- 最大尝试次数 `max_attempts`；
- 检查反馈 `feedback`；
- 工具观察证据 `evidence`；
- 生成结果和产物路径。

模型不能直接把任务标记为 `completed`，也不能修改任务 ID、尝试次数或程序状态。这样可以防止“模型自己说完成了”被误认为真正完成。
如果任务拆分结果提供 `suggested_tools`，程序会先过滤到本地 `available_tools` 范围内，再把它作为当前任务的工具作用域；任务不会因为主模型临时改变想法而随意调用其他任务的工具。

### 4. Convergence Loop：一次只执行当前任务

Scheduler 每次只从任务板选择第一个满足依赖的 `ready` 任务。主 Agent 收到的执行上下文只包含：

- 当前任务；
- 已确认的 Plan；
- 用户已经确认的信息；
- 已完成任务的结果；
- 当前任务上一次失败时的 `feedback`；
- 当前任务实际产生的工具观察结果。

执行完成后，规划模型作为无工具检查模块返回：

```json
{
  "passed": true,
  "feedback": "通过的依据，或下一次重试需要补齐的缺口"
}
```

程序根据这个结果做确定性状态迁移：

```text
ready → running → verifying → completed
                         └──→ ready（仍有重试次数）
                         └──→ blocked（达到最大尝试次数）
```

当前默认每个任务最多尝试 2 次。失败时只重做当前任务，不回滚已经完成的任务；达到上限后阻塞任务板，并在最终结果中明确说明未完成原因。
对于文本聚类，工具还会返回完整的 `cluster_distribution` 和 `cluster_profiles`。程序优先用这些结构化证据核对簇数量、分布和代表文本，不会因为主模型输出过长导致自然语言表格截断，就误判底层聚类失败。

### 5. Reflection：允许任务板动态变化

每个任务检查后，规划模型还可以做一次轻量 Reflection：

- `continue`：按当前任务板继续；
- `add_tasks`：根据新发现增加最多 5 个后续任务；
- `replan_plan`：指出整体目标或范围发生变化，请求重新规划，但不会自行修改原 Plan。

新增任务仍由程序分配 `Tn` 编号、初始状态和依赖关系。当前版本已经支持安全增加后续任务，并在 UI 中显示 `task_board_updated`。

当任务达到最大重试次数，Runtime 会把失败反馈、失败任务和已完成结果重新送入 Plan Loop，重新生成 Plan 和任务板，再继续执行。人工模式会再次请求用户批准，全自动模式会自动批准新的 Plan；当前最多重规划 1 次，避免失败后无限循环。

## 三、端到端执行时序

以“帮我出一份 6 人产品小组的线上用户访谈复盘会方案”为例，Work 模式的典型过程如下：

```text
用户请求
  ↓
本地 Routing：确定这是项目/文档/数据类任务及可用能力
  ↓
语义 Routing：提取本次最关注的重点
  ↓
Plan Review：发现“业务目标、线上工具”等关键缺口
  ↓
Qt 询问，每轮最多 3 个问题
  ↓
用户确认：分析购物车功能卡点，使用腾讯会议
  ↓
生成自然语言 Plan，用户确认执行方向
  ↓
Plan-to-Task：生成“目标背景 / 会议议程 / 工具准备”等任务
  ↓
执行 T1 → 检查 → 完成
执行 T2 → 检查发现缺少时间分配 → feedback → 重试 → 完成
执行 T3 → 检查 → 完成
  ↓
只拼装 T1、T2、T3 的 completed 结果
```

Plan 文书和最终会议方案是两份不同的内容：前者是用户确认的执行蓝图，后者是所有任务通过验收后由程序组装出来的交付结果。

## 四、代码结构

```text
academic_agent/
├── models/
│   └── state.py                    # AgentMode、ApplicationState
├── controllers/
│   ├── application.py              # 组合各 Controller 和 Repository
│   ├── agent.py                    # Agent 延迟初始化、重置、Provider 切换
│   ├── auth.py                     # 登录、登出和认证会话
│   ├── projects.py                 # 项目列表、激活、删除和会话隔离
│   └── sessions.py                 # Work/Chat 会话查询、加载和删除
├── agent/
│   ├── adapters/qwen.py             # Qwen-Agent 适配、主 Agent 与规划 Agent
│   ├── context.py                   # 用户文件、记忆、Plan、确认信息的上下文组装
│   ├── executor.py                  # 工具执行、工具观察和执行作用域
│   ├── runtime.py                   # 总运行时：Routing、Plan、任务板、收敛循环
│   ├── response.py                  # 回复文本、确认链接和结果处理
│   ├── session.py                   # 会话级确认信息、Plan、Trace、任务板
│   ├── types.py                     # AgentPlan、TaskBoard、TaskItem 等数据契约
│   ├── memory/
│   │   ├── coordinator.py           # 跨会话记忆召回和写入协调
│   │   └── manager.py               # 记忆管理
│   ├── orchestration/
│   │   ├── plan_loop.py             # 信息缺口框架、提问轮次、答案回填
│   │   └── verifier.py              # 当前任务的语义检查和证据收集
│   ├── planning/
│   │   ├── planner.py               # 本地能力路由和初始 Plan 上下文
│   │   └── configuration.py         # 文本挖掘/机器学习算法配置
│   ├── policies/tool_access.py      # 工具访问策略
│   ├── skills/catalog.py            # 路由到技能和工具的映射
│   └── tooling/
│       ├── registry.py              # 工具注册表
│       ├── builtins.py              # 工作区等内置能力
│       └── handlers/                # 数据、文本挖掘、ML、可视化、RAG 等处理器
├── application/services/
│   ├── data_service.py              # 数据读取、画像和数据类用例
│   ├── machine_learning_service.py  # 机器学习用例编排
│   ├── text_mining_service.py       # 文本挖掘用例编排
│   └── visualization_service.py     # 可视化用例编排
├── tools/                           # Agent 可调用的稳定工具入口
│   ├── data_analysis_tools.py
│   ├── ml_tools.py
│   ├── text_mining_tools.py
│   ├── viz_tools.py
│   ├── association_tools.py
│   ├── explainability_tools.py
│   └── source_video_adapters.py
├── algorithms/                      # 具体算法实现，不直接管理 Agent 状态
│   ├── machine_learning/
│   ├── preprocessing/
│   ├── text_mining/
│   └── visualization/
├── rag/                             # 文档解析、切分、混合检索、引用证据
│   ├── ingestion/
│   ├── retrieval/
│   ├── providers/
│   ├── storage/
│   └── analysis/
├── repositories/
│   ├── conversations.py              # 对话持久化仓储
│   ├── projects.py                   # 项目持久化仓储
│   └── sqlite_store.py               # SQLite 基础存储
├── infrastructure/
│   ├── workspace_manager.py          # 工作区访问、预览和待确认操作
│   ├── auth_store.py                 # 本地认证信息存储
│   ├── analysis_artifacts.py         # 分析产物记录
│   ├── document_generator.py        # 文档产物生成
│   ├── data_profiler.py              # 文件画像
│   ├── code_executor.py              # 受控代码执行
│   ├── dependency_manager.py        # 依赖检查和安装提示
│   ├── model_paths.py               # 预训练模型路径
│   ├── runtime_paths.py             # output 和运行时目录
│   ├── storage_config.py            # 持久化配置
│   └── persistence/                 # Chroma / 混合向量存储
├── views/
│   ├── main_window.py               # 主窗口生命周期和 Controller 绑定
│   ├── main_layout.py               # 主界面布局、模式和操作入口
│   ├── workers.py                   # Qt 后台线程和 Plan/信息确认回调
│   ├── chat_mixin.py                # 进度事件、任务状态和结果渲染
│   ├── auth.py                      # 登录窗口和账号提示
│   ├── file_mixin.py                # 文件面板与上传入口
│   ├── workspace_mixin.py           # 工作区操作
│   ├── model_mixin.py               # 模型选择
│   └── dialogs.py                   # 算法参数等必要确认窗口
├── integrations/video_text_adapter.py # 外部视频/文本适配
├── tests/
│   └── test_orchestration.py        # 三阶段编排和收敛循环测试
├── qt_client.py                     # PySide6 启动入口
└── scripts/start_qt.sh              # 本地启动脚本
```

核心入口有两个：

1. 客户端入口是 `qt_client.py`，负责启动 Qt、准备运行时目录、应用持久化配置和单实例保护，然后创建 `CodexChatWindow`。
2. Agent 核心入口是 `AgentRuntime`。上层只需要提供 `AgentRequest`、模型调用函数和 UI 回调；运行时负责串起计划生命周期、状态迁移、工具作用域、重试、重规划和最终汇总。

`ApplicationController` 是应用层的组合根：它把 `ApplicationState`、认证、项目、会话和 Agent 生命周期组合起来，但不参与具体任务的语义执行。`AgentService` 是模型适配门面：它同时维护主执行 Agent、Chat Agent 和规划 Agent，并把模型调用转换成 Runtime 能理解的回调。

## 五、模型和程序的职责边界

### 规划模型

`AgentService._get_planning_agent()` 创建一个 `function_list=[]` 的无工具 Agent。它负责：

- Routing 的语义重点提取；
- Plan Loop 的信息缺口判断；
- 生成自然语言 Plan；
- Plan 到任务板的语义拆分；
- 当前任务验收；
- Reflection。

它不读取文件、不写文件、不调用数据分析工具，也不直接执行用户任务。

### 主执行模型

主 Agent 只在真正的任务执行阶段使用注册工具。每次执行前，`ExecutionScope` 设置当前任务的 `allowed_tools`，工具调用必须同时通过任务级作用域和既有工具访问策略。

主 Agent 必须遵守以下边界：

- 只处理消息中的 `current_task`；
- 不提前完成后续任务；
- 不把未来现实动作描述成已经发生；
- 缺少必要信息时指出缺口，不用虚构内容填充；
- 文件生成、修改、删除仍沿用项目已有的预览和确认机制。

### 程序控制器

程序代码负责所有不应交给模型自由发挥的部分：

- 路由能力边界；
- 问题数量上限和去重；
- Plan 是否经过用户确认；
- 任务 ID、状态和依赖；
- 工具作用域；
- 尝试次数和重试上限；
- 结果是否进入 `completed`；
- 最终只汇总已完成结果。

## 六、主要数据契约

### 信息缺口

规划模型的缺口项形态如下：

```json
{
  "topic": "业务目标",
  "why_needed": "决定复盘会议重点和输出结构",
  "question": "这次复盘最希望解决什么业务问题？",
  "status": "missing"
}
```

程序补充：

```json
{
  "info_id": "R1-F1",
  "round": 1,
  "source": "planning_model"
}
```

用户回答后，状态变为 `filled`，并写入当前会话的 `confirmed_information`。

### 动态任务

```json
{
  "id": "T2",
  "title": "撰写会议议程",
  "task_goal": "把复盘过程拆解成可执行的时间段",
  "deliverable": "带时间分配的会议议程",
  "done_when": "必须包含具体时间分配",
  "status": "ready",
  "depends_on": ["T1"],
  "attempts": 0,
  "max_attempts": 2,
  "feedback": []
}
```

规划模型只提供标题、目标、交付物和验收标准；其余状态字段由 `TaskBoard` 创建和维护。

### 关键进度事件

运行时会向 Qt 发送结构化事件，界面不需要猜测模型当前做到了哪一步：

| 事件 | 用途 |
| --- | --- |
| `routing_decided` | 展示路由和语义重点 |
| `plan_created` | 展示初始能力步骤 |
| `plan_reviewed` | 展示本轮 Plan 检查结果 |
| `clarification_required` | 请求用户补充关键信息 |
| `information_filled` | 记录用户确认信息 |
| `plan_document_created` | Plan 文书生成完成 |
| `plan_review_required` | 等待用户确认 Plan |
| `task_board_created` | 展示初始任务板 |
| `task_started` | 当前任务开始 |
| `task_checked` | 当前任务验收结果 |
| `task_retry` | 记录反馈并重试当前任务 |
| `task_repair_started` / `task_repair_finished` | 聚类或情感结果进入规则修复路径，不重新调用 BERT/BGE |
| `task_completed` | 当前任务通过验收 |
| `task_blocked` | 当前任务达到重试上限 |
| `task_board_updated` | Reflection 增加任务 |
| `replan_requested` | Reflection 请求重新规划 |
| `replan_started` / `replan_completed` / `replan_failed` | 任务失败后的重新规划生命周期 |
| `convergence_completed` | 所有任务收敛 |
| `convergence_blocked` | 任务板因失败停止 |
| `response_started` / `completed` | 最终结果开始/结束渲染 |

客户端的“执行过程”在任务板建立前只展示规划阶段；收到 `task_board_created` 后会切换为动态任务清单，按 `task_started`、`task_completed`、`task_retry` 和最终任务板状态实时更新。每个任务完成时会短暂高亮后保留对号；重规划会清理旧任务行，避免重新编号的 `T1` 继承上一轮状态。

聚类和情感分析的修复不等于重新分析。首次分析失败后再次重试，或者重规划明确要求修复时，Runtime 会根据任务语义选择专用规则工具：聚类修复复用已有 `cluster_id` 和原始文本重建簇分布、代表文本与结果文件；情感修复复用已有情感标签和概率重建情感分布与工作簿。若原始分析没有留下可复用标签，修复工具会明确失败并要求重新发起首次分析，而不会偷偷重新加载 BERT/BGE。

## 七、会话、文件与安全边界

当前会话 `SessionContext` 保存：

- 用户上传文件及文件画像；
- 已确认的信息；
- 自然语言 Plan；
- Plan Loop 的 trace；
- 当前任务板；
- 已完成任务和产物路径。

文件权限仍沿用项目原有的两类边界：

1. 用户主动上传的文件是当前会话的授权输入，可以用于分析；上传文件的结果写入应用同级 `output`。
2. 项目工作区文件只能通过工作区工具访问。修改、生成和删除操作先创建预览，再由现有确认机制执行。

Plan 阶段使用无工具规划 Agent，是为了保证“提问、判断、写计划”本身不会产生文件或数据副作用。真正执行工具时又会叠加当前任务允许的工具集合，形成两层保护。

## 八、如何运行

### 启动桌面客户端

在已经配置好依赖和模型环境的情况下：

```bash
python qt_client.py
```

也可以使用项目启动脚本：

```bash
bash scripts/start_qt.sh
```

启动脚本默认使用 `/opt/miniconda3/envs/trancy_tool/bin/python`，可以通过 `PYTHON_BIN` 指向当前机器的 Python：

```bash
PYTHON_BIN=/path/to/python bash scripts/start_qt.sh
```

模型及服务配置沿用项目的 `.env`、`.env.example` 和 `academic_agent/agent/providers/config.py`。如果只想进行自然语言问答，可以在客户端切换到 Chat 模式；Chat 模式不执行工具，也不会进入复杂任务板。

### 运行编译检查

```bash
python -m compileall -q academic_agent tests
```

### 运行编排测试

如果环境已安装 pytest：

```bash
pytest tests/test_orchestration.py
pytest tests
```

`tests/test_orchestration.py` 覆盖：

- Plan Loop 每轮最多 3 个信息缺口；
- 问题编号、答案回填和重复问题过滤；
- 任务板稳定 ID、状态和顺序依赖；
- `passed/feedback` 验收契约；
- 一次执行一个任务、重试和最终结果组装。

## 九、如何扩展

### 增加一种能力路由

1. 在 `academic_agent/agent/types.py` 增加 `AgentRoute` 枚举项。
2. 在 `academic_agent/agent/skills/catalog.py` 注册该能力对应的工作流和工具。
3. 在 `academic_agent/agent/planning/planner.py` 增加确定性的路由识别和预期产物。
4. 在工具注册表中注册真实工具，并确保工具处理器遵守工作区边界。
5. 为该路由补充 Plan-to-Task 和 Convergence 的验收测试。

### 增加新的任务验收规则

优先让任务的 `done_when` 表达可由模型和工具观察共同检查的条件，例如“必须包含字段名”“必须产生一个存在的图表文件”“必须给出指标和限制”。不要把模糊的“看起来不错”作为验收标准。

如果规则是绝对的、可由程序可靠判断的，可以在 `TaskVerifier` 中加入确定性检查；如果规则依赖语义理解，则保留给无工具规划/检查模型，并把依据写入 `evidence` 或 `feedback`。

### 扩展 Reflection

新增 Reflection 动作时要保持“模型提议、程序验证、程序落状态”的顺序。建议的扩展顺序是：

1. 验证新增任务字段和依赖；
2. 限制新增任务数量；
3. 记录版本号和事件；
4. 让 UI 展示变化原因；
5. 如果是 `replan_plan`，暂停执行并重新进入 Plan Review，而不是在原 Plan 上静默改写。

## 十、当前版本的明确边界

为了让使用者知道当前实现到哪里，下面这些点是有意保留的边界：

- Plan Review 当前支持在对话内查看；执行方式通过输入区下拉框选择，人工模式在卡片中确认，全自动模式直接继续，尚未提供直接编辑整份 Plan 的能力。
- Plan 信息缺口当前使用模型生成的单选项，用户通过点击回答，不再自由输入。
- Reflection 已支持新增后续任务，以及任务失败后的有限次重新规划。
- 任务板已经支持依赖字段，但当前 Plan-to-Task 默认按顺序建立依赖，尚未做并行调度。
- 会话中的 Plan、trace 和任务板目前由 `SessionStore` 保存在运行中的进程内；应用重启后的长期持久化需要接入现有会话存储。
- Routing、任务检查和 Reflection 的语义输出不可用时，运行时仍有程序侧兜底；Plan Review 会先做安全等价归一化，再严格校验并自动修复一次，避免把内部格式问题伪装成用户问题。连续修复失败时会记录具体原因，并使用不新增用户约束的本地 Plan；坏结构不会写入任务状态。

这些边界不影响当前三阶段主链路，但在继续演进时应保持 Plan、Task Board、验证结果和最终产物之间的职责分离。

## 十一、推荐的开发验证顺序

每次修改 Agent 编排时，建议按以下顺序验证：

1. 先验证 `types.py` 的数据契约和状态迁移；
2. 再验证 `plan_loop.py` 的问题上限、去重和答案回填；
3. 使用无工具的语义模型 stub 验证 Runtime 的调用顺序；
4. 验证任务失败时只重试当前任务，不重复已经完成的任务；
5. 验证工具调用仍受 `ExecutionScope.allowed_tools` 限制；
6. 最后在 Qt 客户端验证 Plan Review、补充信息、任务进度和最终结果渲染。

这样可以把“模型输出不稳定”和“程序状态错误”分开定位，也能确保新增能力不会破坏现有的数据分析、可视化、RAG 和工作区操作功能。
