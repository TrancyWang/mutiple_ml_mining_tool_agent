"""Agent 服务 - 负责初始化和配置 Qwen-Agent 助手

集成文本挖掘和机器学习工具，支持 Function Calling。
"""

import os
import json
from typing import Optional, Dict, Any, List

from academic_agent.infrastructure.runtime_paths import configure_qwen_agent_workspace

# qwen-agent.settings 在导入时读取该变量，必须放在 qwen_agent 导入之前。
configure_qwen_agent_workspace()

QWEN_AGENT_IMPORT_ERROR = ""
try:
    from qwen_agent.agents import Assistant
    from qwen_agent.tools.base import BaseTool
    from qwen_agent.llm.base import register_llm
    from qwen_agent.llm.oai import TextChatAtOAI
    HAS_QWEN_AGENT = True

    @register_llm('gemini_oai')
    class GeminiChatAtOAI(TextChatAtOAI):
        """Gemini OpenAI-compatible adapter.

        qwen-agent 会自动注入随机 seed，但 Gemini API 不接受该字段。
        在发送前移除该参数，其余 OpenAI-compatible 参数保持不变。
        """

        def _chat_stream(self, messages, delta_stream, generate_cfg):
            gemini_cfg = dict(generate_cfg or {})
            gemini_cfg.pop('seed', None)
            return super()._chat_stream(messages, delta_stream, gemini_cfg)
except ImportError as exc:
    HAS_QWEN_AGENT = False
    QWEN_AGENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"⚠️  警告: qwen_agent 导入失败，将使用简化模式：{QWEN_AGENT_IMPORT_ERROR}")
    class BaseTool:  # type: ignore[no-redef]
        """Fallback so the non-Qwen mode can still expose local tools."""
        pass

from academic_agent.agent.tooling.registry import tool_registry
from academic_agent.agent.executor import tool_executor
from academic_agent.agent.runtime import AgentRuntime
from academic_agent.agent.types import AgentRequest
from academic_agent.agent.providers.config import (
    get_lightweight_text_config,
    get_llm_config_priority,
    get_qwen_fallback_config,
)


class TextMiningToolWrapper(BaseTool):
    """文本挖掘工具的 Qwen-Agent 包装器"""
    
    name = 'text_mining_tool'
    description = '文本挖掘工具集，支持数据加载、预处理、情感分析、聚类、关键词、实体识别和关系抽取'
    
    parameters = {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': '要执行的操作',
                'enum': ['load_data', 'preprocess', 'sentiment_analysis',
                        'clustering', 'extract_keywords', 'entity_recognition',
                        'relation_extraction', 'get_info']
            },
            'file_path': {
                'type': 'string',
                'description': '数据文件路径（load_data时需要）'
            },
            'encoding': {
                'type': 'string',
                'description': '文件编码',
                'default': 'UTF-8'
            },
            'text_column': {
                'type': 'string',
                'description': '要处理的文本列名'
            },
            'algorithm': {
                'type': 'string',
                'description': '算法类型',
                'enum': ['kmeans', 'agglomerative', 'dbscan'],
                'default': 'kmeans'
            },
            'n_clusters': {
                'type': 'integer',
                'description': '聚类数量',
                'default': 5
            },
            'eps': {
                'type': 'number',
                'description': 'DBSCAN 邻域半径',
                'default': 0.5
            },
            'min_samples': {
                'type': 'integer',
                'description': 'DBSCAN 最小样本数',
                'default': 5
            },
            'top_n': {
                'type': 'integer',
                'description': '关键词数量',
                'default': 10
            },
            'mode': {
                'type': 'string',
                'description': '情感模式：chinese 八分类或 general 五分类',
                'enum': ['chinese', 'general'],
                'default': 'chinese'
            },
            'batch_size': {
                'type': 'integer',
                'description': '大模型信息抽取每批文本数，默认 8',
                'default': 8
            },
            'max_texts': {
                'type': 'integer',
                'description': '大模型信息抽取最多处理文本数，默认 200',
                'default': 200
            },
            'provider': {
                'type': 'string',
                'description': '信息抽取模型服务：auto、gemini、qwen、qwen-beijing 或 ollama'
            },
            'model': {
                'type': 'string',
                'description': '可选的模型名称'
            },
            'base_url': {
                'type': 'string',
                'description': '可选的 OpenAI-compatible 服务地址'
            }
        },
        'required': ['action']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行工具调用
        
        Args:
            params: JSON 格式的参数
            kwargs: 其他参数
            
        Returns:
            执行结果（JSON字符串）
        """
        import json
        
        try:
            if isinstance(params, str):
                params_dict = json.loads(params)
            else:
                params_dict = params
            
            action = params_dict.pop('action', None)
            action_map = {
                'load_data': 'load_data',
                'preprocess': 'preprocess_text',
                'sentiment_analysis': 'sentiment_analysis',
                'clustering': 'text_clustering',
                'extract_keywords': 'extract_keywords',
                'entity_recognition': 'entity_recognition',
                'relation_extraction': 'relation_extraction',
                'get_info': 'get_data_info',
            }
            tool_name = action_map.get(action)
            result = (
                tool_executor.execute(tool_name, **params_dict)
                if tool_name else {"success": False, "error": f"未知操作: {action}"}
            )
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


class MLToolWrapper(BaseTool):
    """机器学习工具的 Qwen-Agent 包装器"""
    
    name = 'machine_learning_tool'
    description = '机器学习工具集，支持因果推断、回归分析、分类分析等功能'
    
    parameters = {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': '要执行的操作',
                'enum': ['causal_inference', 'regression', 'classification']
            },
            'treatment_var': {
                'type': 'string',
                'description': '处理变量（因果推断时使用）'
            },
            'outcome_var': {
                'type': 'string',
                'description': '结果变量'
            },
            'control_vars': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': '控制变量列表'
            },
            'feature_vars': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': '特征变量列表'
            },
            'target_var': {
                'type': 'string',
                'description': '目标变量'
            },
            'model_type': {
                'type': 'string',
                'description': '回归或分类模型。分类支持 svm、logistic、random_forest、extra_trees、gradient_boosting、knn、naive_bayes；回归支持 linear、ridge、random_forest、gbdt、adaboost、xgboost、lightgbm、catboost、elastic_net、huber、extra_trees、knn'
            },
            'method': {
                'type': 'string',
                'description': '因果推断方法：ols、logistic、linear_dml、psm、causal_forest'
            },
            'test_size': {
                'type': 'number',
                'description': '测试集比例，默认 0.2',
                'default': 0.2
            },
            'multiple_folds': {
                'type': 'integer',
                'description': '交叉训练次数，默认 5',
                'default': 5
            }
        },
        'required': ['action']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行工具调用"""
        import json
        
        try:
            if isinstance(params, str):
                params_dict = json.loads(params)
            else:
                params_dict = params
            
            action = params_dict.pop('action', None)
            if action == 'classification':
                params_dict.setdefault('model_type', 'svm')
            elif action == 'regression':
                params_dict.setdefault('model_type', 'linear')
                params_dict.setdefault('test_size', 0.2)
                params_dict.setdefault('multiple_folds', 5)
            elif action == 'causal_inference':
                params_dict.setdefault('method', 'ols')
            result = (
                tool_executor.execute(action, **params_dict)
                if action in {'causal_inference', 'regression', 'classification'}
                else {"success": False, "error": f"未知操作: {action}"}
            )
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


class VizToolWrapper(BaseTool):
    """可视化工具的 Qwen-Agent 包装器"""
    
    name = 'visualization_tool'
    description = '可视化工具集，支持生成词云、分布图、模型评估图等'
    
    parameters = {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': '要执行的操作',
                'enum': ['wordcloud', 'line_chart', 'cluster_plot', 'sentiment_plot', 
                        'regression_plot', 'feature_importance']
            },
            'text_column': {
                'type': 'string',
                'description': '文本列名（词云时使用）'
            },
            'cluster_column': {
                'type': 'string',
                'description': '聚类列名'
            },
            'sentiment_column': {
                'type': 'string',
                'description': '情感列名'
            },
            'title': {
                'type': 'string',
                'description': '图表标题'
            }
        },
        'required': ['action']
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行工具调用"""
        import json
        
        try:
            if isinstance(params, str):
                params_dict = json.loads(params)
            else:
                params_dict = params
            
            action = params_dict.pop('action', None)
            result = (
                tool_executor.execute(action, **params_dict)
                if action in {'wordcloud', 'cluster_plot', 'sentiment_plot', 'line_chart'}
                else {"success": False, "error": f"未知操作: {action}"}
            )
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


class DocumentRAGToolWrapper(BaseTool):
    """Qwen-Agent wrapper for the integrated structure-aware document RAG."""

    name = "document_rag"
    description = (
        "对用户上传或当前项目工作区内的 PDF、Markdown、TXT、DOCX、CSV、JSON 文档进行结构化解析、"
        "混合检索和引用溯源。需要查询文档内容时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要回答的文档问题"},
            "file_paths": {"type": "array", "items": {"type": "string"}, "description": "可选，限定文档路径"},
            "top_k": {"type": "integer", "description": "返回证据片段数，默认 6", "default": 6},
        },
        "required": ["query"],
    }

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else dict(params)
            result = tool_executor.execute("document_rag", **payload)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


class RegistryToolWrapper(BaseTool):
    """统一工具入口：自然语言 Agent 和 UI Tools 使用同一注册表。"""

    name = "analysis_tool"
    description = "统一工具入口。先选择 action，再提供 params；支持数据分析、可视化、当前项目工作区的 Glob/Grep/Read/Write/Edit/Delete。write_project_file 用于在当前工作区创建或覆盖代码文件，generate_project_file 用于把分析文档写入 output 目录；必要时可在临时工作区执行受控 Python 分析代码。调用工作区生成、写入、编辑或删除工具会先创建待确认预览，确认由 Qt 对话框中的可点击链接完成。"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [spec.name for spec in tool_registry.all()],
                "description": "要执行的注册工具名称",
            },
            "params": {
                "type": "object",
            "description": "工具参数，例如 text_column、n_clusters、target_var、feature_vars、path、query、pattern、content、operation_id",
            },
        },
        "required": ["action"],
    }

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else params
            action = payload.get("action")
            result = tool_executor.execute(action, **(payload.get("params") or {}))
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


class AgentService:
    """Qwen Agent 服务"""
    
    def __init__(self):
        """初始化 Agent 服务"""
        self.agent = None
        self.chat_agent = None
        self.planning_agent = None
        self.llm_config: Dict[str, Any] = {}
        # Qt 客户端不能直接把终端 traceback 展示给用户，因此保留最近一次
        # 初始化失败的可读原因，供桌面端提示和日志使用。
        self.last_init_error = ""
        self.runtime = AgentRuntime()
        self.tools = [
            RegistryToolWrapper(),
            DocumentRAGToolWrapper(),
            TextMiningToolWrapper(),
            MLToolWrapper(),
            VizToolWrapper()
        ]
        # 默认用户ID（单用户模式）
        self.default_user_id = "default_user"

    def _get_chat_agent(self) -> Any:
        """返回不注册任何工具的纯聊天 Agent。

        Work 模式和 Chat 模式共用模型配置，但 Chat 模式不应因为自然语言
        中出现“分析文件”等字样就意外执行项目工具。
        """
        if self.chat_agent is not None:
            return self.chat_agent
        if not HAS_QWEN_AGENT or not self.llm_config:
            return None
        self.chat_agent = Assistant(
            llm=self.llm_config,
            function_list=[],
            system_message=(
                "你是 Academic Agent 的 Chat 模式助手。只进行自然语言问答、解释和思路整理，"
                "不要读取、生成、修改或删除文件，不要调用任何工具。始终使用中文回答。"
            ),
            name="AcademicChatAssistant",
        )
        return self.chat_agent

    def _get_planning_agent(self) -> Any:
        """返回不注册工具的规划/检查 Agent，避免 Plan 阶段产生副作用。"""
        if self.planning_agent is not None:
            return self.planning_agent
        if not HAS_QWEN_AGENT or not self.llm_config:
            return None
        # 规划、检查和 JSON 契约不需要创造性，使用独立的低温配置，
        # 避免复用主 Agent 的高温度导致结构漂移。
        planning_llm = dict(self.llm_config)
        planning_generate_cfg = dict(self.llm_config.get("generate_cfg") or {})
        planning_generate_cfg.update({
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 2048,
        })
        planning_llm["generate_cfg"] = planning_generate_cfg
        self.planning_agent = Assistant(
            llm=planning_llm,
            function_list=[],
            system_message=(
                "你是 Academic Agent 的规划与质量检查模块。只处理语义规划、任务拆分、"
                "信息缺口判断和结果检查，不调用工具，不执行文件或数据操作。"
            ),
            name="AcademicPlanningAssistant",
        )
        return self.planning_agent
    
    def init_agent(self,
                   model_name: Optional[str] = None,
                   llm_config: Optional[Dict] = None) -> Any:
        """初始化文本挖掘助手 Agent
        
        Args:
            model_name: 模型名称
            llm_config: LLM 配置
            
        Returns:
            Assistant: Qwen-Agent 助手实例
        """
        self.last_init_error = ""
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "gemini-3.6-flash")

        if not HAS_QWEN_AGENT:
            self.last_init_error = (
                "打包应用中 qwen_agent 导入失败"
                f"（{QWEN_AGENT_IMPORT_ERROR or '模块不存在'}）。"
                "请重新构建应用，或检查构建环境是否使用 trancy_tool。"
            )
            print(f"❌ Agent 初始化失败: {self.last_init_error}")
            return None

        try:
            # 默认模式仍遵循 Gemini → Qwen（新加坡/北京）→ Ollama；显式选择的
            # provider 不静默改动，避免用户以为正在使用某个云模型。
            if llm_config is None:
                print(f"🔧 正在初始化 Agent，请求模型: {model_name}")
                llm_config = get_llm_config_priority(
                    model_name,
                    provider=os.getenv("LLM_PROVIDER", "auto"),
                )
                print(
                    f"✅ LLM 配置: {llm_config['model']} "
                    f"@ {llm_config.get('model_server', 'N/A')}"
                )
            self.llm_config = dict(llm_config)
        
            # 系统提示词
            system_prompt = """你是一个专业的文本挖掘和机器学习助手，具备以下能力：

【核心功能】
1. **数据处理**：加载CSV/Excel文件、文本预处理、分词、去停用词
2. **文本挖掘**：情感分析、文本聚类、关键词提取、实体识别、关系抽取
3. **机器学习**：因果推断、回归分析、分类建模（支持多种分类算法）
4. **可视化**：生成词云、分布图、模型评估图表
5. **文档 RAG**：对用户上传或当前工作区文档进行结构化解析、混合检索，并根据页码、标题和块类型返回引用证据
6. **项目代码开发**：可以在当前项目工作区读取、创建和修改 Python、JavaScript、TypeScript、HTML、CSS、Java、C/C++、Go、Rust 等源代码文件；禁止访问工作区之外的路径、密钥和隐藏配置
7. **分析产物生成**：可以在当前项目工作区的 output 目录生成 Markdown、TXT、JSON、CSV、Excel、Word、PDF 和 PPT 文件

【工作流程】
- Work 模式会由 Runtime 先完成 Routing、Plan 和任务板转换
- 执行任务时只处理提示中的 current_task，不要提前完成其他任务
- 每次工具操作后依据观察结果继续当前任务；不要自行修改任务状态
- 如果当前任务缺少关键信息，明确指出缺口，不要用虚构内容填补
- 每次操作后，向用户清晰地解释结果
- 如果用户没有提供必要的参数，请询问用户

【重要原则】
- 始终用中文回复
- 解释技术术语，让用户容易理解
- 提供实用的建议和下一步操作
- 如果出错，清楚地说明问题所在
- 涉及项目文件时，先使用项目工作区工具检索或读取，再创建或修改文件；生成代码文件使用 write_project_file，修改已有文件使用 edit_project_file，生成分析报告使用 generate_project_file
- 生成 .docx、.pdf、.pptx 或 .xlsx 时，将内容以清晰的 Markdown/纯文本结构传给 generate_project_file，由工具负责转换为对应二进制文档
- 只有当现有文本挖掘、机器学习和可视化工具无法完成任务时，才调用 execute_python_analysis；代码必须是短小、可复现的 Python 分析脚本，输入文件通过 input_files 显式声明
- 如果当前任务消息包含 scheduled_tool_observations，说明算法工具已经由 Runtime 调度执行；此时不要再次调用算法工具，只解释这些结构化观察结果
- execute_python_analysis 只在临时工作区运行；工作区文件的产物写入工作区 output，用户上传文件的产物写入应用同级 output；不要执行 shell、网络访问、系统管理或访问其他路径
- 项目代码文件的创建、编辑和删除必须先调用对应工作区工具创建预览；不要先向用户索要“确认生成”“同意”等文字，也不要自定义二次确认流程。Qt 客户端会根据 operation_id 自动显示“点击确认执行”链接，用户点击链接后系统会完成确认操作
- 对话回复中不要复制完整文件原文、代码、检索命中内容或差异内容，只汇报文件操作状态、数量、路径和结果摘要；用户需要查看原文时引导其使用右侧“打开文件”面板

你可以使用的工具：
- document_rag: 文档结构化解析、混合检索和引用溯源
- text_mining_tool: 文本挖掘相关操作
- machine_learning_tool: 机器学习建模
- visualization_tool: 图表生成
- analysis_tool: 当前项目工作区的文件检索、代码读取、代码创建、代码编辑和确认操作

请根据用户的需求，智能地选择合适的工具并执行。"""
        
            # 创建 Agent
            self.agent = Assistant(
                llm=llm_config,
                function_list=self.tools,
                system_message=system_prompt,
                name='TextMiningAssistant'
            )
            self.chat_agent = None
            self.planning_agent = None
            
            print(f"✅ Agent 初始化成功，使用模型: {llm_config['model']}")
            print(f"   服务器: {llm_config.get('model_server', 'N/A')}")
            return self.agent

        except Exception as e:
            self.agent = None
            safe_config = dict(self.llm_config or {})
            for secret_key in ("api_key", "key", "token"):
                if secret_key in safe_config:
                    safe_config[secret_key] = "***"
            self.last_init_error = (
                f"{type(e).__name__}: {e}；"
                f"模型={safe_config.get('model', '未设置')}；"
                f"服务地址={safe_config.get('model_server', '未设置')}"
            )
            print(f"❌ Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def chat(self, 
             messages: List[Dict[str, str]],
             user_id: Optional[str] = None,
             session_id: Optional[str] = None,
             workspace_path: Optional[str] = None,
             chat_only: bool = False,
             **kwargs) -> Any:
        """与 Agent 对话
        
        Args:
            messages: 消息历史列表
            user_id: 用户ID（可选，默认为 default_user）
            kwargs: 其他参数
            
        Returns:
            响应结果
        """
        if self.agent is None:
            return {"error": "Agent 未初始化"}

        owner = user_id or self.default_user_id
        runtime_agent = self._get_chat_agent() if chat_only else self.agent
        if runtime_agent is None:
            return {"error": "Chat Agent 未初始化"}
        request = AgentRequest(
            messages=[dict(message) for message in messages],
            user_id=owner,
            session_id=str(session_id or owner),
            workspace_path=workspace_path,
            chat_only=chat_only,
        )
        try:
            planning_agent = self._get_planning_agent()
            return self.runtime.run(
                request,
                lambda enhanced: runtime_agent.run(messages=enhanced, **kwargs),
                semantic_runner=(
                    (lambda enhanced: planning_agent.run(messages=enhanced, **kwargs))
                    if planning_agent is not None else None
                ),
            )
        except Exception as exc:
            return {"error": f"对话失败: {exc}"}

    def chat_stream(self, messages: List[Dict[str, str]], user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    workspace_path: Optional[str] = None,
                    chat_only: bool = False,
                    progress_callback=None,
                    plan_confirmation_callback=None,
                    plan_review_callback=None,
                    clarification_callback=None,
                    _allow_fallback: bool = True, **kwargs):
        """流式对话：模型产生增量内容后立即向 UI 返回。"""
        if self.agent is None:
            yield {"error": "Agent 未初始化"}
            return

        owner = user_id or self.default_user_id
        runtime_agent = self._get_chat_agent() if chat_only else self.agent
        if runtime_agent is None:
            yield {"error": "Chat Agent 未初始化"}
            return
        request = AgentRequest(
            messages=[dict(message) for message in messages],
            user_id=owner,
            session_id=str(session_id or owner),
            workspace_path=workspace_path,
            chat_only=chat_only,
        )
        has_output = False
        try:
            planning_agent = self._get_planning_agent()
            for response in self.runtime.stream(
                request,
                lambda enhanced: runtime_agent.run(messages=enhanced, **kwargs),
                event_callback=progress_callback,
                plan_confirmation_callback=plan_confirmation_callback,
                plan_review_callback=plan_review_callback,
                clarification_callback=clarification_callback,
                semantic_runner=(
                    (lambda enhanced: planning_agent.run(messages=enhanced, **kwargs))
                    if planning_agent is not None else None
                ),
            ):
                if isinstance(response, list) and response:
                    has_output = has_output or bool(response[-1].get("content"))
                yield response
            return
        except Exception as exc:
            if _allow_fallback and self.llm_config.get("model_type") == "gemini_oai" and not has_output:
                fallback_config = get_qwen_fallback_config()
                if fallback_config and self.init_agent(llm_config=fallback_config) is not None:
                    yield from self.chat_stream(
                        messages=messages,
                        user_id=owner,
                        session_id=session_id,
                        workspace_path=workspace_path,
                        chat_only=chat_only,
                        progress_callback=progress_callback,
                        plan_confirmation_callback=plan_confirmation_callback,
                        plan_review_callback=plan_review_callback,
                        clarification_callback=clarification_callback,
                        _allow_fallback=False,
                        **kwargs,
                    )
                    return
            yield {"error": f"对话失败 [{type(exc).__name__}]: {exc}"}
            return


# 创建全局 Agent 服务实例
agent_service = AgentService()
