# 医疗增强型 GraphRAG 项目

这是一个基于“**大模型 + 知识图谱**”的医疗问答课程项目实现。系统融合了：

- `ChromaDB` 文本向量检索
- `本地 JSON 图谱 + Python 多跳检索`
- `硅基流动 / 智谱 / 通义` 等 OpenAI 兼容 API 的知识抽取与问答生成
- `LangChain` 问答链路编排
- `Streamlit + streamlit-agraph` 的可视化溯源前端

项目重点支持多跳问题，例如：

- `肺炎有哪些常见症状？`
- `肺炎的并发症脓毒症有哪些适用抗生素？`
- `糖尿病需要做哪些检查？`

## 目录结构

```text
.
├── app.py
├── data/
│   └── medical_texts/
├── docs/
├── scripts/
│   ├── build_demo.py
│   ├── generate_medical_corpus.py
│   └── verify_pipeline.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── kg_builder.py
│   ├── llm_client.py
│   ├── prompts.py
│   ├── qa_chain.py
│   └── retriever.py
└── requirements.txt
```

## 1. 环境准备

1. 安装 Python 3.10 及以上。
2. 复制环境变量模板：

```bash
copy .env.example .env
```

3. 修改 `.env` 中的 LLM 配置。

推荐配置：

- 硅基流动：`LLM_PROVIDER=siliconflow`，`LLM_MODEL=deepseek-ai/DeepSeek-V3`
- 智谱：`LLM_PROVIDER=zhipu`，`LLM_MODEL=glm-4-flash`
- 通义：`LLM_PROVIDER=tongyi`，`LLM_MODEL=qwen-plus`

如果暂时没有 API Key，项目仍可运行，但会退化为**规则型抽取 + 规则型回答兜底模式**，便于先打通全流程。

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 生成 500+ 条医疗文本

```bash
python scripts/generate_medical_corpus.py
```

执行后会在 `data/medical_texts/medical_corpus.jsonl` 生成 512 条结构化医疗文本，覆盖疾病、症状、检查、并发症、药物、抗生素等信息。

## 4. 构建图谱和向量库

```bash
python scripts/build_demo.py
```

如需只构建图谱，可执行：

```bash
python -m src.kg_builder
```

## 5. 启动前端

```bash
streamlit run app.py
```

启动后网页可完成：

- 对话式提问
- 纯向量召回结果展示
- 图谱三元组命中展示
- 触发子图可视化溯源

## 6. 核心模块说明

### `src/kg_builder.py`

- 读取 `data/medical_texts/` 的 JSON / JSONL 文本
- 做去噪、规则抽取或大模型抽取
- 输出标准三元组
- 写入本地 `data/graph_store.json`，并完成实体别名对齐

### `src/retriever.py`

- 基于 `BAAI/bge-small-zh-v1.5` 构建文本向量
- 在 ChromaDB 中完成 Top-K 检索
- 对问题进行实体链接
- 生成一跳到多跳 Python 图遍历结果
- 将图谱关系与文本片段合并为混合上下文

### `src/qa_chain.py`

- 使用 LangChain 组织系统提示词与问答链
- 强化“只能依据当前知识库回答”的安全边界
- 当上下文不足时返回“无法根据当前知识库作答”

### `app.py`

- 提供 Streamlit 问答页面
- 展示图谱命中实体、三元组、纯向量片段与混合上下文
- 使用 `streamlit-agraph` 渲染当前查询触发的知识网络子图

## 7. 验证方式

### 底座验证

- 构建完成后检查本地图谱文件是否生成：

```bash
python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("data/graph_store.json").read_text(encoding="utf-8"))
print("nodes =", len(payload["nodes"]))
print("edges =", len(payload["edges"]))
print("documents =", len(payload["documents"]))
PY
```

预期应看到上百个节点与大量关系，并可观察到类似：

- `肺炎 -[:HAS_COMPLICATION]-> 脓毒症`
- `脓毒症 -[:RECOMMENDED_DRUG]-> 美罗培南`
- `美罗培南 -[:BELONGS_TO_DRUG_CLASS]-> 抗生素`

### 召回对比验证

```bash
python scripts/verify_pipeline.py
```

脚本会分别打印：

- 纯向量检索返回
- 混合检索命中的图谱关系
- 最终回答

### 典型案例

1. 简单常识查询：`肺炎有哪些常见症状？`
2. 多跳推理查询：`肺炎的并发症脓毒症有哪些适用抗生素？`
3. 越界问题：`请告诉我协和医院某位医生的手机号。`

第三类问题应返回“无法根据当前知识库作答”，并且页面只展示有限的已知图谱节点或空结果。
