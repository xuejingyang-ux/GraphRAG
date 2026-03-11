# GraphRAG 增强型知识图谱问答系统

一个基于 React + Express + SQLite 的本地 GraphRAG 演示项目。系统现在默认使用项目根目录 `data/` 中的电影数据集构建知识图谱与检索索引，支持围绕电影、导演、演员和合作关系进行问答。

## 当前能力

- 数据集导入：解析 `data/` 目录中的 Excel/Graph 数据文件并批量入库
- 图谱构建：自动生成电影、导演、演员、类型、合作关系等实体与边
- 检索增强：结合本地快速向量表示、文本匹配和 2-Hop 图谱检索
- 问答生成：基于命中的图谱关系和证据片段调用智谱生成答案
- 对照测试：支持 `GraphRAG` 与 `纯向量 RAG` 两种问答接口
- 图谱管理：支持清空数据库、获取完整图谱结构

## 技术栈

- 前端：React 19、Vite、TypeScript、D3
- 后端：Express、better-sqlite3、Node.js
- 模型服务：智谱开放平台
- 数据存储：SQLite
- 数据解析：Python + openpyxl

## 项目结构

```text
.
├─ data/                课程数据集目录
├─ scripts/             数据解析、评测与报告脚本
├─ src/                 前端页面
├─ server.mjs           本地开发服务端与 API
├─ graph_rag.db         运行后生成的 SQLite 数据库
├─ package.json         项目依赖与脚本
└─ README.md            项目说明
```

## 环境要求

- Node.js 18+
- Python 3（需可运行 `python` 命令）
- 已安装 `openpyxl`
- 可用的智谱 API Key

## 安装与启动

1. 安装前端/后端依赖

```bash
npm install
```

2. 确认 Python 依赖已安装

```bash
python -m pip install openpyxl
```

3. 在项目根目录创建 `.env.local`

```env
ZHIPU_API_KEY=你的智谱APIKey
ZHIPU_CHAT_MODEL=glm-4-flash
ZHIPU_REASONING_MODEL=glm-4-flash
ZHIPU_EMBEDDING_MODEL=embedding-3
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
PORT=3000
```

4. 启动项目

```bash
npm run dev
```

启动后访问 [http://localhost:3000](http://localhost:3000)，点击左侧“导入 data 目录数据集”即可完成数据入库。

## data 数据集说明

当前项目使用以下文件：

- `data/net_edges.xlsx`：演员合作关系边
- `data/*.xlsx`（除 `net_edges.xlsx` 外）：电影元数据表
- `data/net.graphml`、`data/net_large.graphml`：原始图文件，可作为课程附加材料保留

系统导入时会将数据整理为以下结构：

- 电影实体
- 导演实体
- 演员实体
- 类型实体
- 电影-导演关系
- 电影-主演关系
- 演员-参演关系
- 演员-合作演员关系
- 电影说明文本与合作关系文本

## 后端接口

### `POST /api/ingest-dataset`

从 `data/` 目录读取真实数据集并完成批量导入。

返回示例：

```json
{
  "success": true,
  "movies": 2000,
  "edges": 1895,
  "documents": 3895,
  "entities": 3000,
  "relationships": 8000
}
```

### `POST /api/query`

执行 GraphRAG 混合检索，结合图谱与证据片段回答问题。

请求示例：

```json
{
  "query": "张国荣和梁家辉有什么关系？"
}
```

### `POST /api/query-vector`

执行纯向量/文本检索对照问答，不使用图谱关系。

### `GET /api/graph`

返回完整图谱结构，供前端可视化使用。

### `GET /api/dataset-status`

返回 `data/` 目录状态与当前数据库中的实体、关系、文档数量。

### `POST /api/clear`

清空当前 SQLite 中的文档、实体和关系数据。

## 数据流程

1. 点击前端按钮或调用 `/api/ingest-dataset`
2. 后端调用 `scripts/extract_dataset.py` 解析 Excel 数据
3. 系统批量写入电影、导演、演员、类型与合作关系
4. 系统为电影描述与合作描述生成本地检索向量
5. 用户提问时，系统先做文本/向量召回，再做图谱两跳检索
6. 系统将图谱关系和文档证据交给智谱生成最终答案

## 开发说明

- `npm run dev`：启动 Express + Vite 开发服务
- `npm run build`：构建前端静态资源
- `npm run lint`：执行 TypeScript 类型检查
- `python scripts/run_eval.py`：生成测试对照结果 `test_results.json`
- `python scripts/generate_report.py`：基于测试结果生成课程报告文档

## 注意事项

- 没有配置 `ZHIPU_API_KEY` 时，问答接口会直接返回错误
- 数据集导入依赖本地 `python` 和 `openpyxl`
- `graph_rag.db` 是本地数据库文件，清空接口会删除其中所有图谱和检索数据
- `.env.local`、数据库文件、评测结果和课程文档默认不应提交到仓库

## 推荐提问示例

- `《英雄》是谁执导的？`
- `姜文主演过什么电影？`
- `张国荣和梁家辉有什么关系？`
- `紫霞仙子出现在哪部电影里？`
- `冯小刚执导过哪些电影？`
