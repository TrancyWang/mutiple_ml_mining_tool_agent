# Academic Agent 用户使用文档

本文面向下载代码后在自己电脑上运行 Academic Agent 的用户，说明如何启动程序、首次登录、配置大模型服务、配置 BERT/BGE 本地模型，以及完成一次数据分析任务。

> Academic Agent 当前是 PySide6 桌面应用。账号、历史会话和部分配置默认保存在本机，不是已经部署好的网站，也没有云端账号服务器。

## 1. 使用前准备

### 1.1 获取代码并安装依赖

在项目根目录创建 Python 3.10 或 3.11 虚拟环境：

macOS / Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 不允许激活脚本运行，可以先在当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 1.2 启动客户端

在项目根目录执行：

```bash
python qt_client.py
```

macOS / Linux 也可以执行启动脚本：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/start_qt.sh
```

如果只想测试界面是否能打开，可以先不配置云端 API Key；真正发送需要模型服务可用。

## 2. 注册与登录

### 2.1 先理解“注册”是什么意思

当前版本没有单独的“注册账号”页面，也没有邮箱、手机号或验证码注册。

普通用户第一次登录时填写一个用户名，提交模型服务 API Key 后，程序会在本机 SQLite 的 `auth_accounts` 表中保存这条账户记录。这个过程可以理解为“本机首次创建账户”，但它不是云端注册：

- 用户名只用于区分本机账户和历史会话；
- 普通用户密码可以留空，当前代码不会把普通用户密码作为登录凭证；
- 每个普通用户需要提供自己有权限使用的 Gemini 或千问 API Key；
- 账户记录不会同步到其他电脑；
- 清空本机数据或更换运行目录后，可能需要重新登录。

### 2.2 普通用户首次登录

1. 启动程序，进入主界面。
2. 点击左下角的“登录账户”。如果尚未登录，第一次使用需要登录的功能时也会自动弹出登录窗口。
3. 在“用户名”中填写任意非空用户名，例如 `researcher01`。
4. “密码”可以留空。
5. 在“模型服务”中选择一个服务：
   - `Gemini`；
   - `千问（新加坡）`；
   - `千问（北京）`。
6. 在“API Key”中粘贴与所选服务对应的 API Key。
7. “记住 API Key”默认勾选。个人电脑可以勾选；共享电脑建议取消勾选。
8. 点击“登录”。

![登录窗口](docs/screenshots/login-dialog.png)

登录成功后，左下角会显示当前账户。此时可以使用数据分析、文本挖掘、机器学习、文档问答和工作区功能。

### 2.3 管理员/部署方登录

管理员模式用于读取部署方预先配置的 `.env`，不要求在登录窗口粘贴 API Key。管理员账号规则目前由 `academic_agent/controllers/auth.py` 中的部署配置决定。

公开发布前请注意：

- 不要把管理员用户名和密码写进 GitHub README、截图或安装包说明；
- 不要直接沿用源码中的演示凭据，应先修改为部署方自己的凭据；
- 管理员登录只适合单机或受控环境，不等同于生产级多租户认证；
- `.env` 必须放在本机受保护的位置，并且不能提交到 Git。

管理员登录后的模型服务配置来自 `.env`。源码运行时通常放在项目根目录；打包后可以放在应用旁边，具体搜索位置以当前版本启动提示为准。

### 2.4 切换账户与退出登录

- 切换账户：点击左下角账户入口，或打开“设置 → 账户 → 切换账户”。
- 退出登录：在账户菜单中选择退出登录。
- 退出登录只清理当前登录状态和云端凭据，不会删除历史会话、项目文件和分析产物。
- 如果勾选了“记住 API Key”，下次打开登录窗口时用户名、服务和 API Key 可能会自动填充。

本机登录信息默认保存在源码运行目录下的 `runtime/data/academic_agent.db`；也可以通过 `SQLITE_MEMORY_PATH` 指定其他 SQLite 文件。不要把这个数据库分享给其他人，因为其中可能包含被记住的 API Key 和本地账户记录。

## 3. 配置大模型服务

Academic Agent 有两类模型：

1. 对话/规划模型：负责理解问题、生成 Plan、拆分任务和解释结果；
2. 本地文本算法模型：BGE 负责文本向量，BERT 模型负责部分情感/情绪分析。

两者需要分别配置。登录窗口里的 API Key 只解决对话模型服务，不能代替本地 BERT/BGE 模型文件。

### 3.1 在界面中切换对话模型

登录后，点击输入框下方的“模型”按钮，可以选择：

- 自动：按 Gemini → 千问新加坡 → 千问北京 → Ollama 的顺序寻找可用服务；
- Gemini；
- Qwen 新加坡；
- Qwen 北京；
- Ollama 本地：`qwen3.5:2b` 或 `qwen3.5:0.8b`。

选择云端模型后，必须已经在登录窗口填写对应 API Key，或者由管理员 `.env` 提供对应凭据。显式选择某个服务时，如果该服务没有凭据，程序会提示配置错误，不会悄悄改用另一个云端服务。

### 3.2 使用 `.env` 配置云端服务

复制模板：

```bash
cp .env.example .env
```

`.env` 中只填写你自己的值，不要填写到 README 或提交到 GitHub：

```dotenv
# auto：Gemini → 千问新加坡 → 千问北京 → Ollama
LLM_PROVIDER=auto

# Gemini，二选一即可
GOOGLE_API_KEY=your_google_api_key
# GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-3.6-flash

# 千问新加坡，任选一个变量名
# ALIYUN_API_KEY=your_dashscope_api_key
# DASHSCOPE_API_KEY_SG=your_dashscope_api_key
# QWEN_API_KEY_SG=your_dashscope_api_key

# 千问北京，任选一个变量名
# DASHSCOPE_API_KEY_BJ=your_dashscope_beijing_api_key
# QWEN_API_KEY_BJ=your_dashscope_beijing_api_key
# QWEN_BEIJING_API_KEY=your_dashscope_beijing_api_key

QWEN_MODEL=qwen-plus
```

如果希望固定使用某个服务，可以改成：

```dotenv
LLM_PROVIDER=gemini
```

或：

```dotenv
LLM_PROVIDER=qwen
```

修改 `.env` 后请完全退出并重新启动客户端。普通用户登录时直接填写 API Key 后，程序会以本次登录选择为准；管理员登录时，程序会读取 `.env`。

### 3.3 使用本地 Ollama

Ollama 不需要云端 API Key，但需要先在本机安装并启动 Ollama：

```bash
ollama serve
ollama pull qwen3.5:2b
ollama list
```

`.env` 配置示例：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen3.5:2b
```

当前普通用户登录窗口要求填写 Gemini 或千问 API Key；因此“完全没有云端 Key、只使用 Ollama”的场景，需要由部署方使用管理员 `.env`，或在发布前按自己的认证需求调整 `academic_agent/controllers/auth.py`。登录后再从“模型”菜单选择 Ollama。

## 4. 配置本地 BERT / BGE 模型

### 4.1 必须自行下载模型

代码仓库不包含 BERT、Transformer 或 BGE 权重。程序不会在运行时自动替用户下载这些模型。用户必须：

1. 从原始模型发布方下载模型；
2. 解压为完整的 Transformers 模型目录；
3. 放到指定目录；
4. 在客户端或 `.env` 中配置模型库根目录。

不要只下载一个 `.bin` 或 `.safetensors` 文件。每个模型目录通常至少需要包含：

- `config.json`；
- tokenizer 配置和词表文件，例如 `tokenizer.json`、`tokenizer_config.json`、`vocab.txt` 或 `tokenizer.model`；
- 完整模型权重，例如 `*.safetensors` 或 `pytorch_model.bin`。

请同时确认模型的许可证允许你的研究、教学或商业部署场景。不要把未经许可的模型权重提交到公共 GitHub 仓库。

### 4.2 模型库的准确目录结构

建议把模型放在项目目录之外，例如 `/Users/yourname/models/pretrain_models` 或 `D:/models/pretrain_models`。

`PRETRAINED_MODELS_DIR` 必须指向下面的 `pretrain_models` 根目录：

```text
pretrain_models/
├── bge-cn/
│   ├── config.json
│   ├── tokenizer.json 或 tokenizer.model
│   └── 模型权重文件
├── multilingual-sentiment-analysis/
│   ├── config.json
│   ├── tokenizer 和词表文件
│   └── 模型权重文件
└── xuyuan-trial-sentiment-bert-chinese/
    ├── config.json
    ├── tokenizer 和词表文件
    └── 模型权重文件
```

三个子目录的用途：

| 目录 | 类型 | 主要用途 |
| --- | --- | --- |
| `bge-cn` | BGE 向量模型 | 文本向量、文本聚类和部分主题分析 |
| `multilingual-sentiment-analysis` | 通用五分类模型 | `general` 通用情感分析 |
| `xuyuan-trial-sentiment-bert-chinese` | 中文八分类 BERT | `chinese` 中文情感/情绪分析 |

其中 `bge-cn` 是文本向量模型，不是分类 BERT；文本聚类至少需要它。情感分析还需要对应的分类模型。

### 4.3 方式 A：在客户端选择模型库根目录

登录后有两种入口：

- 点击输入框下方“模型”按钮 → “设置文本挖掘模型…”；
- 打开设置 → “模型” → “设置模型库”。

然后按以下步骤操作：

1. 选择包含 `bge-cn`、`multilingual-sentiment-analysis` 和 `xuyuan-trial-sentiment-bert-chinese` 子目录的 `pretrain_models` 文件夹；
2. 查看窗口中的模型检查结果；
3. 确认 BGE 显示“已找到”；
4. 点击保存；
5. 如果只想单独指定中文八分类模型，在“设置 → 模型 → 中文八分类情绪模型”中选择具体的 `xuyuan-trial-sentiment-bert-chinese` 文件夹。

![模型配置页面](docs/screenshots/model-settings.png)

选择模型库时应选择根目录，不要选择 `bge-cn` 内部的某个权重文件。程序会自动检查子目录是否存在。首次运行文本聚类时，如果没有检测到 `bge-cn`，程序会再次提示选择模型库。

### 4.4 方式 B：在 `.env` 中配置模型路径

macOS / Linux：

```dotenv
PRETRAINED_MODELS_DIR=/Users/yourname/models/pretrain_models
SENTIMENT_MODEL_PATH=/Users/yourname/models/pretrain_models/xuyuan-trial-sentiment-bert-chinese
```

Windows：

```dotenv
PRETRAINED_MODELS_DIR=D:/models/pretrain_models
SENTIMENT_MODEL_PATH=D:/models/pretrain_models/xuyuan-trial-sentiment-bert-chinese
```

说明：

- `PRETRAINED_MODELS_DIR` 是模型库根目录；
- `SENTIMENT_MODEL_PATH` 是具体的中文八分类模型目录，可选；
- 如果不填写 `SENTIMENT_MODEL_PATH`，程序会优先使用模型库中的默认中文八分类目录；
- 修改 `.env` 后必须重启客户端；
- 界面中保存的路径会写入 Qt 本地设置，通常优先用于后续启动。

### 4.5 外部文本算法源项目

部分文本能力还复用了外部项目 `video_text_mutiplemodal_agent`，包括文本预处理、BGE 聚类、中文 BERT 情感分析和部分 KeyBERT 流程。推荐目录关系：

```text
parent/
├── mutiple_ml_mining_tool_agent/
└── video_text_mutiplemodal_agent/
    ├── src_codes/
    └── pretrain_models/
```

如果源项目不在默认的同级目录，需要在 `.env` 中配置：

```dotenv
VIDEO_AGENT_SOURCE_ROOT=/absolute/path/to/video_text_mutiplemodal_agent
PRETRAINED_MODELS_DIR=/absolute/path/to/pretrain_models
```

`VIDEO_AGENT_SOURCE_ROOT` 必须指向源项目根目录，不能直接指向 `src_codes` 或其中的某个子目录。

## 5. 完成第一次分析

### 5.1 选择模式

左上角可以选择：

- `Academic Agent · Work`：选择项目、上传文件、执行数据和项目工具；
- `Academic Agent · Chat`：只进行自然语言问答和解释，不执行项目文件工具。

### 5.2 Work 模式分析 CSV/Excel

1. 切换到 `Academic Agent · Work`。
2. 在左侧选择或管理当前项目。
3. 点击输入框左下方的“上传文件”，选择 CSV、Excel、JSON、TXT、PDF 或 DOCX 文件。
4. 等待文件出现在当前对话的上下文中。
5. 在输入框描述目标、文件和必要参数，例如：

```text
请查看当前文件的数据规模、字段类型、缺失值和重复值，并生成一份数据质量报告。
```

```text
请对当前文件的 content 列做文本聚类，使用 5 个主题，输出每个主题的数量、关键词和代表文本。
```

```text
请使用 sales 作为目标变量，使用 advertising_cost 和 price 作为特征做回归预测，并报告评估指标。
```

6. 根据界面提示补充缺少的信息，例如列名、聚类数、目标变量或特征列。
7. 检查 Plan 和任务清单后继续执行。
8. 结果完成后，在右侧“打开文件”区域查看 CSV、Excel、图片或报告。

### 5.3 审批方式

输入框下方的“执行方式”有两个选项：

- `请求批准`：Plan、算法参数和关键节点由用户确认，适合第一次使用或重要数据；
- `帮我批准`：Agent 自动完成这些确认节点，适合已经熟悉流程的重复任务。

无论选择哪种方式，涉及工作区文件生成、编辑和删除的操作仍应检查预览和确认提示。不要对不了解的删除操作直接确认。

### 5.4 查看结果

文本聚类通常会生成 CSV、Excel 或其他分析产物；结果默认写入应用的 `output/` 目录。用户上传文件产生的结果与项目工作区产物分开保存。

![文本聚类执行过程](docs/screenshots/cluster-progress.png)

![文本聚类结果](docs/screenshots/cluster-result.png)

常见结果包括：

- 预处理后的数据；
- 聚类标签和聚类统计；
- 情感分析结果；
- 关键词、词云和分布图；
- 回归/分类评估指标和图表；
- 文档问答的引用证据和分析报告。

## 6. 打包与发布

本项目使用 PyInstaller 打包桌面客户端。打包机只需要安装根目录下统一的 `requirements.txt`，不再区分 Windows、构建机和开发机依赖。

### 6.1 打包前检查

开始打包前，请确认：

1. 已获取完整代码，并在项目根目录执行命令；
2. 使用 Python 3.10 或 3.11；
3. 已激活项目自己的 `.venv`；
4. 已执行 `python -m pip install -r requirements.txt`；
5. `qwen-agent` 和 `PyInstaller` 可以被当前 Python 找到；
6. 已准备好 `.env.example`，但不要把真实 `.env`、API Key、SQLite 数据库、BERT/BGE 权重或用户数据放入发布包；
7. 如需发布文本预处理、BGE 聚类或中文 BERT 情感分析能力，已另外准备外部源项目和本地模型目录。

可以先检查打包工具：

```bash
python -c "import PyInstaller, qwen_agent; print('build dependencies: ok')"
```

### 6.2 macOS 打包

macOS 打包只能在 macOS 上执行，生成的 `.app` 不能直接替代 Windows 安装包。

先创建并准备环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS spec 默认需要找到外部项目 `video_text_mutiplemodal_agent` 中的：

```text
video_text_mutiplemodal_agent/
└── src_codes/main_process_agent/text_processor_subagent_stage_3/
```

如果它位于当前项目的同级目录，可以直接执行：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/build_macos.sh
```

如果源项目位于其他位置，显式指定路径：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
VIDEO_AGENT_SOURCE_ROOT=/absolute/path/to/video_text_mutiplemodal_agent \
bash scripts/build_macos.sh
```

脚本会自动完成：

1. 检查 Python 和 PyInstaller；
2. 检查外部文本算法源项目；
3. 清理旧的 `build/` 和 `dist/` 内容；
4. 设置 PyInstaller 和 Matplotlib 缓存目录；
5. 使用 `AcademicAgent.spec` 构建 `.app`；
6. 复制 `.env.example` 到产物目录；
7. 输出 `dist/AcademicAgent.app`。

构建完成后可以双击：

```text
dist/AcademicAgent.app
```

#### macOS 打包的密钥风险

当前 macOS spec 和构建脚本会在项目根目录存在 `.env` 时读取并复制它。`.env` 中如果有真实 API Key，打包后可能进入 `.app` 或 `dist/.env`，等同于把密钥发布给拿到安装包的人。

公开发布前请执行以下检查：

```bash
test ! -f .env && echo "safe: no real .env"
find dist -name .env -print
```

如果发现真实 `.env`，请先移出项目目录或改成只包含占位符的配置，再执行打包。发布包只应携带 `.env.example`，用户在安装后自行配置 API Key 或使用客户端登录窗口填写。

如确实需要在受控内网中发布预配置管理员环境，也不要把该包上传到公共 GitHub，并且应在发布前轮换不再使用的 API Key。

#### 是否携带本地模型

默认不把 BERT/BGE 权重放进 `.app`，推荐让用户在“模型”菜单中选择模型库目录。如果确实要随包携带外部源项目的模型，可以执行：

```bash
INCLUDE_LOCAL_MODELS=1 \
PYTHON_BIN="$PWD/.venv/bin/python" \
VIDEO_AGENT_SOURCE_ROOT=/absolute/path/to/video_text_mutiplemodal_agent \
bash scripts/build_macos.sh
```

这会显著增大安装包体积，并且需要确认模型许可证和分发权限。公共发布通常不建议这样做。

### 6.3 Windows 打包

Windows 打包需要在 Windows PowerShell 中执行。当前脚本会自动使用统一的 `requirements.txt` 安装依赖，并生成 Windows 图标。

在项目根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.\scripts\build_windows.ps1
```

也可以先激活虚拟环境，再直接执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
.\scripts\build_windows.ps1
```

脚本会自动完成：

1. 检查 `python` 是否在 PATH 中；
2. 更新 pip 并安装统一依赖；
3. 执行 `scripts/create_windows_icon.py`，生成 `resources/academic_agent_icon.ico`；
4. 清理旧的 `build/` 和 `dist/`；
5. 设置 PyInstaller 和 Matplotlib 缓存目录；
6. 使用 `AcademicAgent.windows.spec` 生成 onedir 应用；
7. 只复制 `.env.example`，不复制真实 `.env`；
8. 生成 `dist/AcademicAgent-windows.zip`。

Windows 发布产物为：

```text
dist/AcademicAgent-windows.zip
```

用户解压后运行其中的 `AcademicAgent.exe`。首次运行仍需配置：

- Gemini、千问或 Ollama 模型服务；
- BERT/BGE 本地模型目录；
- 外部文本算法源项目（如果使用相关文本能力）。

Windows spec 默认不把外部 `video_text_mutiplemodal_agent` 和本地模型权重打进安装包，因此只下载 Windows 压缩包并不等于已经准备好全部文本算法依赖。

### 6.4 GitHub Actions 自动构建

仓库中的 `.github/workflows/build-windows.yml` 支持：

- 手动触发 `workflow_dispatch`；
- 推送 `v*` 格式 tag 时自动构建。

Workflow 使用 Windows Runner、Python 3.10 和统一 `requirements.txt`，最终上传 `AcademicAgent-windows.zip`。如果只是本地运行源码，不需要执行 GitHub Actions，也不需要修改 `.spec` 文件。

### 6.5 发布前验收

打包后不要只检查压缩包是否生成，还应在一台没有开发环境的干净电脑上验证：

1. 解压或复制安装包到新的目录；
2. 确认包内没有真实 `.env`、API Key 和开发机绝对路径；
3. 启动应用并打开登录窗口；
4. 使用测试账号和测试 API Key 登录，或配置本地 Ollama；
5. 在“模型”设置中选择外部模型库；
6. 上传一个小型 CSV；
7. 测试数据概览：

```text
请查看当前文件的数据规模、字段类型、缺失值和重复值。
```

8. 如果已配置 `bge-cn`，再测试文本聚类：

```text
请对当前文件的 content 列做文本聚类，使用 5 个主题，并输出每个主题的数量和代表文本。
```

9. 确认结果文件可以在右侧文件面板打开，且输出目录具有写入权限。

### 6.6 只运行源码和制作安装包的区别

| 场景 | 是否需要 `.spec` | 是否需要 PyInstaller | 是否需要清理密钥和模型 |
| --- | --- | --- | --- |
| 用户下载源码运行 | 不需要 | 不需要 | 仍然需要保护 `.env` 和模型目录 |
| 开发者本机调试 | 不需要 | 不需要 | 不要把本地配置提交到 Git |
| macOS 制作 `.app` | 需要 `AcademicAgent.spec` | 需要 | 必须，尤其是 macOS `.env` 封装风险 |
| Windows 制作压缩包 | 需要 `AcademicAgent.windows.spec` | 需要 | 必须，Windows 脚本只复制 `.env.example` |

## 7. 常见问题

### 登录后提示普通用户必须填写 API Key

请确认：

1. 用户名不是部署方的管理员用户名；
2. 已选择 Gemini 或千问服务；
3. API Key 与服务节点对应；
4. 没有把本地 Ollama 误当成普通用户的云端登录方式。

如果只想使用 Ollama，请参考“3.3 使用本地 Ollama”。

### 提示找不到 BGE 向量模型

请检查 `PRETRAINED_MODELS_DIR` 是否指向模型库根目录，并确认存在：

```text
<PRETRAINED_MODELS_DIR>/bge-cn/
```

不要把路径配置成 `bge-cn/config.json` 或某个权重文件路径。还要确认模型目录中包含完整 tokenizer 和权重文件。

### 情感分析提示找不到中文模型

确认存在：

```text
<PRETRAINED_MODELS_DIR>/xuyuan-trial-sentiment-bert-chinese/
```

或者在“设置 → 模型 → 中文八分类情绪模型”中直接选择具体模型目录。该目录至少应包含 `config.json`。

### Ollama 连接失败

执行：

```bash
ollama list
```

确认模型名称与 `OLLAMA_MODEL` 一致，再检查 Ollama 是否正在运行、端口是否为 `11434`，以及 `OLLAMA_HOST` 是否设置为 `127.0.0.1`。

### 提示缺少 `qwen_agent`

在已激活的虚拟环境中执行：

```bash
python -m pip install "qwen-agent>=0.0.34"
```

然后重新启动客户端。

### 结果没有写入文件

检查 `output/` 或 `ACADEMIC_AGENT_OUTPUT_DIR` 是否可写。打包应用不要把输出目录放到只读的 `.app` 内部；可以把它配置到用户有权限的目录。

## 8. 安全注意事项

- 不要把真实 API Key 写入源代码、README、截图、测试文件或 Git 提交。
- 不要提交 `.env`、`dist/.env`、SQLite 数据库、`runtime/`、`output/`、模型权重和用户上传文件。
- 如果 API Key 曾经误提交到 Git，不能只删除当前文件，还需要在服务商后台撤销/轮换，并清理 Git 历史。
- 共享电脑不要勾选“记住 API Key”。
- 模型权重可能受单独许可证约束，发布前请检查来源和使用范围。
- 当前登录模块是本地桌面认证，不适合作为公网多用户账号系统。若要在线部署，需要另外设计服务端鉴权、用户隔离、密钥管理、文件权限和资源限制。

更多安装、构建和开发说明请查看项目根目录的 [README.md](README.md)。
