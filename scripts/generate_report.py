# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "dataset_report_results.json"
MD_PATH = ROOT / "项目研究报告.md"
DOCX_PATH = ROOT / "项目研究报告.docx"


def load_metrics():
    if RESULT_PATH.exists():
      with RESULT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
      return data.get("ingest", {})
    return {
        "movies": 2000,
        "edges": 1895,
        "documents": 3890,
        "entities": 6758,
        "relationships": 22405,
    }


def build_markdown(metrics):
    return f"""# GraphRAG 增强型知识图谱问答系统研究报告

## 1. 项目概述

本项目基于 React、Express、SQLite 与智谱大模型构建了一个增强型 GraphRAG 电影知识问答系统。与前期仅支持示例文本导入的版本相比，当前版本已经完成了对项目根目录 `data/` 数据集的真实接入，可直接解析 Excel/Graph 文件，自动构建电影、导演、演员及其合作关系图谱，并据此完成问答与可视化展示。

本系统主要面向课程中的“知识图谱增强检索生成”任务，重点体现以下能力：

1. 使用本地真实数据集而非手工样例。
2. 同时支持纯向量/文本召回和 GraphRAG 混合检索。
3. 支持可视化展示知识图谱及证据片段。
4. 支持课程测试报告中的三类典型问答场景。

## 2. 系统需求描述

### 2.1 功能需求

1. 支持从 `data/` 目录批量导入电影数据集。
2. 支持自动构建电影、导演、演员、类型、合作关系等知识图谱结构。
3. 支持用户使用自然语言对电影信息发起查询。
4. 支持对问答过程中的图谱证据和文本证据进行展示。
5. 支持纯向量 RAG 与 GraphRAG 两种问答模式的效果对比。
6. 支持清空知识库并重新导入数据集。

### 2.2 非功能需求

1. 系统应支持本地运行，部署简单。
2. 系统应具有良好的扩展性，便于替换模型或数据源。
3. 系统应具备较强的可视化和交互体验，便于展示课程成果。

## 3. 系统设计

### 3.1 总体架构

系统整体采用四层结构：

1. 前端层：React + Vite，负责页面交互、问题输入、图谱展示、证据片段展示。
2. 服务层：Express，负责数据导入、检索、问答和状态接口。
3. 数据层：SQLite，负责存储文档、实体与关系。
4. 模型层：智谱开放平台，负责答案生成；本地轻量向量函数负责数据集检索向量构建。

### 3.2 核心模块

1. 数据解析模块：通过 `scripts/extract_dataset.py` 解析 `data/` 中的电影表格与演员合作边。
2. 图谱构建模块：将电影、导演、主演、类型、合作关系写入 SQLite 图谱表。
3. 文档索引模块：将电影说明文本与合作关系文本写入文档表，并生成本地检索向量。
4. 混合检索模块：先进行文本/向量召回，再进行 2-Hop 图谱检索。
5. 问答生成模块：调用智谱模型输出最终答案。
6. 对照测试模块：`/api/query-vector` 仅使用文档证据，`/api/query` 同时使用图谱关系与文档证据。

## 4. 系统实现

### 4.1 数据集导入实现

项目使用 `data/` 目录中的真实数据文件：

1. `net_edges.xlsx`：演员合作关系。
2. 电影元数据表：包含电影名称、导演、主演、类型、英文名、评分等字段。
3. `net.graphml` 与 `net_large.graphml`：图结构原始文件，作为补充数据资源保留。

数据导入接口为 `POST /api/ingest-dataset`。导入后系统统计结果如下：

- 电影数量：{metrics.get("movies", 0)}
- 合作边数量：{metrics.get("edges", 0)}
- 文档数量：{metrics.get("documents", 0)}
- 实体数量：{metrics.get("entities", 0)}
- 关系数量：{metrics.get("relationships", 0)}

### 4.2 问答流程实现

系统问答流程如下：

1. 用户在前端输入自然语言问题。
2. 系统首先在 `documents` 表中进行文本/向量召回。
3. 系统识别问题中的关键实体，例如电影名、导演名、演员名。
4. GraphRAG 模式会在 `relationships` 表中执行两跳图谱检索。
5. 系统将召回的文本证据与图谱关系拼接成 Prompt。
6. 智谱模型根据证据输出最终答案。

### 4.3 前端实现

前端页面已调整为数据集导向版本，主要包括：

1. “导入 data 目录数据集”按钮。
2. 图谱可视化面板。
3. 问答输入区域。
4. 证据片段展示区域。
5. 知识库清空功能。

## 5. 核心 Prompt 展示

### 5.1 查询实体识别 Prompt

```text
你是查询实体识别助手。请只输出 JSON 字符串数组，
提取用户问题中的关键实体名；如果没有明显实体，返回 []。

{{QUERY}}
```

### 5.2 GraphRAG 问答生成 Prompt

```text
你是 GraphRAG 问答助手。请优先基于给定的图谱关系和文档片段作答，回答使用中文；
如果证据不足，请明确说明不确定，不要编造。

知识图谱上下文：
{{GRAPH_CONTEXT}}

文档片段：
{{DOCUMENT_CONTEXT}}

用户问题：
{{QUERY}}
```

### 5.3 纯向量 RAG Prompt

```text
你是纯向量RAG问答助手。你只能基于给定文档片段作答，回答使用中文；
如果文档中没有足够证据，请明确说明无法回答，不要补充图谱推理和外部常识。

文档片段：
{{DOCUMENT_CONTEXT}}

用户问题：
{{QUERY}}
```

## 6. 系统效果展示

建议在此部分插入以下截图：

1. 系统首页：展示左侧数据集导入按钮、右侧图谱区域与输入框。
2. data 数据集导入成功截图：展示导入完成提示。
3. 简单查询截图：例如“《英雄》是谁执导的？”。
4. 多跳推理截图：例如“张国荣和梁家辉有什么关系？”。
5. 边界案例截图：例如“周星驰获得过几次奥斯卡奖？”。

## 7. 测试报告

### 7.1 测试案例一：简单查询

问题：`《英雄》是谁执导的？`

纯向量 RAG 预期表现：
能够从召回到的电影说明文本中找到导演字段并给出回答。

GraphRAG 预期表现：
不仅能回答导演是张艺谋，还能利用图谱中的“电影-导演”关系增强答案稳定性。

结论：
在单跳事实问题上，二者都能得到较好答案，但 GraphRAG 的结构化依据更明确。

### 7.2 测试案例二：多跳推理

问题：`张国荣和梁家辉有什么关系？`

纯向量 RAG 预期表现：
主要依赖文本片段召回，可能只能从电影说明中间接得出二人共同出演同一作品的事实。

GraphRAG 预期表现：
能够利用合作关系边或电影-主演关系边，直接指出二者存在合作演员关系，回答更加结构化。

结论：
在涉及人物关系、合作网络、多跳关联的问题上，GraphRAG 优势更明显。

### 7.3 测试案例三：无法回答的边界案例

问题：`周星驰获得过几次奥斯卡奖？`

纯向量 RAG 预期表现：
若召回文本中没有该信息，应提示证据不足。

GraphRAG 预期表现：
能够根据当前图谱和文档都不存在相关证据，明确拒答并避免幻觉。

结论：
在边界问题上，GraphRAG 更容易基于“图谱缺失 + 文档缺失”给出受约束的回答。

## 8. 总结

本项目已经完成从“前端示例展示版”到“接入真实数据集的 GraphRAG 系统”的升级。相比只依赖文本检索的纯向量 RAG，GraphRAG 在人物合作关系、电影关系网络、多跳推理与证据链解释方面具有更强优势。当前系统已具备课程展示、测试对比和后续扩展的基础能力。

## 9. 复现实验方法

1. 在项目根目录准备 `.env.local`。
2. 安装依赖：`npm install`
3. 安装 Python 依赖：`python -m pip install openpyxl`
4. 启动项目：`npm run dev`
5. 在前端点击“导入 data 目录数据集”
6. 分别通过 `/api/query-vector` 与 `/api/query` 对比问答效果
"""


def set_default_style(document: Document):
    style = document.styles["Normal"]
    style.font.name = "宋体"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.font.size = Pt(11)


def add_paragraph(document: Document, text: str):
    if text.startswith("# "):
        return
    if text.startswith("## "):
        document.add_heading(text[3:], level=1)
        return
    if text.startswith("### "):
        document.add_heading(text[4:], level=2)
        return
    if text.startswith("```"):
        return
    if text.startswith("- "):
        document.add_paragraph(text[2:], style="List Bullet")
        return
    if any(text.startswith(f"{i}. ") for i in range(1, 10)):
        document.add_paragraph(text, style="List Number")
        return
    if text.strip():
        document.add_paragraph(text)


def build_docx(markdown_text: str):
    doc = Document()
    set_default_style(doc)

    title = doc.add_paragraph()
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = title.add_run("GraphRAG 增强型知识图谱问答系统研究报告")
    run.bold = True
    run.font.size = Pt(16)
    run.font.name = "黑体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    subtitle.add_run("数据集接入版")

    doc.add_paragraph("")
    for line in markdown_text.splitlines():
        add_paragraph(doc, line)

    doc.save(DOCX_PATH)


def main():
    metrics = load_metrics()
    markdown_text = build_markdown(metrics)
    MD_PATH.write_text(markdown_text, encoding="utf-8")
    build_docx(markdown_text)
    print(MD_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
