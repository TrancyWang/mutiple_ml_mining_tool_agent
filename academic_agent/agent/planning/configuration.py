"""算法确认对话框使用的、与执行层共享的配置目录。"""

from __future__ import annotations

from typing import Any


def _parameter(
    name: str,
    label: str,
    kind: str,
    default: Any,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "label": label,
        "type": kind,
        "default": default,
    }
    if minimum is not None:
        value["min"] = minimum
    if maximum is not None:
        value["max"] = maximum
    if step is not None:
        value["step"] = step
    return value


def _option(
    key: str,
    label: str,
    description: str,
    parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "description": description,
        "parameters": parameters or [],
    }


def text_clustering_configuration() -> dict[str, Any]:
    return {
        "kind": "text_clustering",
        "title": "确认文本聚类方案",
        "description": "请选择文本向量聚类方法和参数。聚类前会先使用 BGE 文本向量。",
        "default_algorithm": "kmeans",
        "options": [
            _option(
                "kmeans",
                "KMeans（推荐）",
                "适合已大致知道主题数量、希望得到大小相对均衡的簇。",
                [_parameter("n_clusters", "聚类数量", "int", 5, 2, 50)],
            ),
            _option(
                "agglomerative",
                "层次聚类",
                "逐步合并相近文本，适合观察主题层次结构。",
                [_parameter("n_clusters", "聚类数量", "int", 5, 2, 50)],
            ),
            _option(
                "dbscan",
                "DBSCAN",
                "按密度发现簇，可识别离群文本；不需要预先指定簇数量。",
                [
                    _parameter("eps", "邻域半径 eps", "float", 0.5, 0.05, 10.0, 0.05),
                    _parameter("min_samples", "最小样本数", "int", 5, 2, 50),
                ],
            ),
        ],
    }


def sentiment_configuration() -> dict[str, Any]:
    return {
        "kind": "sentiment_analysis",
        "title": "确认情感分析方法",
        "description": "请选择中文情绪识别或通用情感分类模型。",
        "default_algorithm": "chinese",
        "options": [
            _option("chinese", "中文八分类情绪模型（推荐）", "识别中文文本的多种情绪及无情绪状态。"),
            _option("general", "通用五分类情感模型", "输出更通用的正面、负面等五类情感结果。"),
        ],
    }


def information_extraction_configuration(kind: str) -> dict[str, Any]:
    label = "实体识别" if kind == "entity_recognition" else "关系抽取"
    return {
        "kind": kind,
        "title": f"确认{label}方案",
        "description": f"使用可配置的 OpenAI-compatible 大模型完成{label}，结果会写入 Excel 工作簿。",
        "default_algorithm": "llm",
        "options": [
            _option(
                "llm",
                f"大模型{label}（推荐）",
                "输出结构化 JSON，并保留文本、实体/三元组和模型信息。",
                [
                    _parameter("batch_size", "每批文本数", "int", 8, 1, 32),
                    _parameter("max_texts", "最多处理文本数", "int", 200, 1, 5000),
                ],
            )
        ],
    }


def machine_learning_configuration(task: str) -> dict[str, Any]:
    common_parameters = [
        _parameter("test_size", "测试集比例", "float", 0.2, 0.1, 0.5, 0.05),
        _parameter("multiple_folds", "交叉训练次数", "int", 5, 2, 10),
    ]
    if task == "classification":
        options = [
            _option("svm", "SVM（支持向量机）", "当前项目已验证的分类实现，适合中小规模分类数据。", common_parameters),
            _option("logistic", "Logistic 回归", "可解释性较强，适合二分类或多分类基线模型。", common_parameters),
            _option("random_forest", "随机森林", "适合非线性关系和特征交互，抗过拟合能力较好。", common_parameters),
            _option("extra_trees", "ExtraTrees 极端随机树", "比随机森林随机性更强，适合快速建立非线性基线。", common_parameters),
            _option("gradient_boosting", "Gradient Boosting", "逐步拟合误差，适合中小规模结构化数据。", common_parameters),
            _option("knn", "KNN", "基于邻近样本进行分类，适合尺度统一后的中小规模数据。", common_parameters),
            _option("naive_bayes", "朴素贝叶斯", "训练速度快，适合作为概率分类基线。", common_parameters),
        ]
        title = "确认分类算法与参数"
        description = "请选择分类模型及评估参数。"
    elif task == "causal_inference":
        options = [
            _option("ols", "OLS 线性回归（推荐）", "估计连续结果变量的平均处理效应。"),
            _option("logistic", "Logistic 回归", "结果变量为二分类时使用。"),
            _option("linear_dml", "Linear DML 双重机器学习", "控制混杂并估计线性处理效应。"),
            _option("psm", "倾向得分匹配（PSM）", "按倾向得分寻找匹配样本后估计处理效应。"),
            _option("causal_forest", "因果森林", "估计个体处理效应异质性。"),
        ]
        title = "确认因果推断方法"
        description = "请选择因果推断方法；变量字段仍由 Agent 根据你的请求确认。"
    else:
        options = [
            _option("linear", "线性回归（推荐）", "适合目标变量与特征近似线性关系的回归任务。", common_parameters),
            _option("ridge", "Ridge 回归", "在线性模型中加入正则化，适合特征相关性较强的情况。", common_parameters),
            _option("random_forest", "随机森林回归", "适合非线性关系，并能提供特征重要性。", common_parameters),
            _option("gbdt", "GBDT 回归", "通过逐步拟合残差捕捉非线性关系。", common_parameters),
            _option("adaboost", "AdaBoost 回归", "通过组合弱学习器提升预测能力。", common_parameters),
            _option("xgboost", "XGBoost 回归", "适合结构化数据的高性能梯度提升模型。", common_parameters),
            _option("lightgbm", "LightGBM 回归", "适合较大规模结构化数据的梯度提升模型。", common_parameters),
            _option("catboost", "CatBoost 回归", "对类别特征较友好的梯度提升模型。", common_parameters),
            _option("elastic_net", "ElasticNet 回归", "结合 L1/L2 正则化，适合高维且存在共线性的特征。", common_parameters),
            _option("huber", "Huber 鲁棒回归", "对异常值不敏感，适合含有少量离群点的连续目标。", common_parameters),
            _option("extra_trees", "ExtraTrees 回归", "用极端随机树捕捉复杂非线性关系。", common_parameters),
            _option("knn", "KNN 回归", "根据邻近样本预测连续目标，适合局部规律明显的数据。", common_parameters),
        ]
        title = "确认回归算法与参数"
        description = "请选择回归模型及评估参数。"
    return {
        "kind": task,
        "title": title,
        "description": description,
        "default_algorithm": options[0]["key"],
        "options": options,
    }


def configuration_for_query(query: str, route: str) -> dict[str, Any] | None:
    """根据 Planner 已确定的路由识别需要用户确认的算法类型。"""
    text = str(query).lower()
    if route == "data":
        if any(word in text for word in ("实体识别", "命名实体", "ner", "named entity", "entity recognition")):
            return information_extraction_configuration("entity_recognition")
        if any(word in text for word in ("关系抽取", "关系识别", "relation extraction", "knowledge graph")):
            return information_extraction_configuration("relation_extraction")
        if "聚类" in text or "cluster" in text:
            return text_clustering_configuration()
        if "情感" in text or "情绪" in text or "sentiment" in text:
            return sentiment_configuration()
    if route == "machine_learning":
        if "分类" in text or "classification" in text:
            return machine_learning_configuration("classification")
        if "因果" in text or "causal" in text or "处理效应" in text:
            return machine_learning_configuration("causal_inference")
        if "回归" in text or "regression" in text or "预测" in text:
            return machine_learning_configuration("regression")
        # 用户只说“做机器学习”时仍给出明确入口，让 Agent 不必擅自替用户决定任务类型。
        config = machine_learning_configuration("regression")
        config["kind"] = "machine_learning"
        config["title"] = "确认机器学习任务与算法"
        config["description"] = "你的请求未明确分类、回归或因果推断，请先选择一个任务类型。"
        config["task_type_required"] = True
        config["task_options"] = [
            {"key": "regression", "label": "回归 / 数值预测"},
            {"key": "classification", "label": "分类 / 类别预测"},
            {"key": "causal_inference", "label": "因果推断"},
        ]
        return config
    return None
