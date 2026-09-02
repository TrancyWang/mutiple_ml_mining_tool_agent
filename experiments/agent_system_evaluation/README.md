# Agent system evaluation

这是一套针对当前项目第一篇系统论文的离线实验代码，评估重点是：

1. `TaskPlanner` 是否能把自然语言任务路由到正确的能力域；
2. 规划后的工具范围是否比“把所有工具直接暴露给模型”更紧凑；
3. `AgentPolicy` 是否能拦截聊天模式、无工作区和越权文件操作；
4. 工作区文件操作是否具备“预览—确认—执行”的可追溯流程。

## 运行

在项目根目录执行：

```bash
python experiments/agent_system_evaluation/run_experiments.py
```

也可以指定输出目录：

```bash
python experiments/agent_system_evaluation/run_experiments.py \
  --output-dir experiments/agent_system_evaluation/results
```

输出文件包括：

- `planner_results.csv`：每个任务的路由、工具覆盖和工具范围；
- `security_results.csv`：权限策略测试结果；
- `confirmation_results.json`：文件操作确认测试结果；
- `summary.json`：汇总指标；
- `report.md`：可直接作为论文实验结果初稿的 Markdown 报告。

## 当前实验边界

本实验不调用外部大模型 API，因此可以离线、稳定地重复运行。它验证的是系统中确定性的 Planner、工具注册和权限控制层，不等同于最终的端到端 LLM 任务成功率。

后续可以在同一任务集上增加模型适配器，记录实际模型的工具调用序列，再补充不同模型、不同提示词和多次重复运行的端到端结果。

## Gemini 端到端实验

项目 `.env` 中如果已配置 Gemini 凭据，可以执行：

```bash
python experiments/agent_system_evaluation/run_live_gemini.py --limit 6
```

该脚本会显式选择 Gemini，创建一个临时上传 CSV，运行小规模真实模型任务，并把工具调用序列写入 `results_live_gemini/live_results.json`。临时 CSV 会在实验结束后删除；项目文件生成任务只验证确认预览，不自动确认写入。

## 传统 Agent 对照实验

论文的核心比较使用：

```bash
python experiments/agent_system_evaluation/run_agent_comparison.py --limit 6 --repeats 1
```

`direct_agent` 和 `planned_agent` 使用同一个 Gemini 模型、同一批文本/机器学习数据、同一套工具执行器。前者不接收本地任务计划，后者接收当前 `TaskPlanner` 生成的路由和候选工具。结果记录任务成功率、干净执行率、工具调用次数、失败重试次数、耗时和产物生成率。

## 算法有效性主实验

论文主实验使用：

```bash
conda run --no-capture-output -n trancy_tool python -u \
  experiments/agent_system_evaluation/run_hybrid_vs_gemini.py --repeats 3
```

这里的 `gemini_only` 禁止调用工具、Python 和本地模型；`hybrid_bert` 使用项目中的本地情感模型；`hybrid_sklearn` 使用固定训练/测试划分上的 SVM 和线性回归。主指标是情感/分类的 Accuracy、Macro-F1，以及回归的 R²、RMSE、MAE。
