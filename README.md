# GraphRAG 增强型知识图谱问答系统

一个基于 React + Express + SQLite 的本地 GraphRAG 演示项目。前端负责对话和图谱可视化，后端负责文档入库、实体关系抽取、向量检索、两跳图查询，以及通过智谱大模型生成最终回答。

## 当前能力

- 文本入库：调用智谱 Embedding 接口生成向量并写入 SQLite
- 图谱构建：调用智谱对文本抽取三元组并保存实体、关系
- 实体对齐：在导入阶段尝试将别名对齐到已存在实体
- 混合检索：组合向量相似度检索和 2-Hop 图谱关系检索
- 问答生成：基于命中的图谱和文档片段调用智谱生成答案
- 图谱管理：支持清空数据库、获取完整图谱结构

## 技术栈

- 前端：React 19、Vite、TypeScript、D3
- 后端：Express、better-sqlite3、Node.js
- 模型服务：智谱开放平台
- 数据存储：SQLite

## 项目结构

```text
.
├─ src/                 前端页面
├─ scripts/             评测与报告辅助脚本
├─ server.mjs           本地开发服务端与 API
├─ vite.config.ts       Vite 配置
├─ package.json         项目依赖与脚本
└─ README.md            项目说明
```

## 环境要求

- Node.js 18+
- 可用的智谱 API Key

## 安装与启动

1. 安装依赖

```bash
npm install
```

2. 在项目根目录创建 `.env.local`，写入至少以下内容

```env
ZHIPU_API_KEY=你的智谱APIKey
```

3. 可选模型配置

```env
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

启动后访问 [http://localhost:3000](http://localhost:3000)

## 后端接口

### `POST /api/ingest`

导入一段文本，自动完成：

- 向量化
- 三元组抽取
- 实体对齐
- 文档与图谱落库

请求示例：

```json
{
  "text": "周星驰执导了电影《功夫》。"
}
```

### `POST /api/query`

根据用户问题执行混合检索并返回答案、命中的图谱关系和参考片段。

请求示例：

```json
{
  "query": "周星驰和功夫有什么关系？"
}
```

### `POST /api/query-vector`

执行纯向量 RAG 检索，仅使用召回文档片段生成答案，不使用图谱关系。

请求示例：

```json
{
  "query": "周星驰和功夫有什么关系？"
}
```

### `GET /api/graph`

返回完整图谱结构，供前端可视化使用。

### `POST /api/clear`

清空当前 SQLite 中的文档、实体和关系数据。

## 数据流程

1. 前端提交文本到 `/api/ingest`
2. 后端调用智谱 Embedding 生成文档向量
3. 后端调用智谱抽取实体关系三元组
4. 后端执行实体对齐后写入 SQLite
5. 用户提问时，后端先做向量召回，再做图谱两跳检索
6. `/api/query-vector` 仅基于召回文档生成答案，作为纯向量 RAG 对照
7. `/api/query` 将图谱证据与文档证据一起交给模型生成最终答案

## 开发说明

- `npm run dev`：启动 Express + Vite 开发服务
- `npm run build`：构建前端静态资源
- `npm run lint`：执行 TypeScript 类型检查
- `python scripts/run_eval.py`：生成测试对照结果 `test_results.json`
- `python scripts/generate_report.py`：基于测试结果生成课程报告文档

## 注意事项

- 没有配置 `ZHIPU_API_KEY` 时，后端接口会直接返回错误
- 当前实体抽取和实体对齐依赖模型输出，建议先导入较干净的文本样本
- `graph_rag.db` 是本地数据库文件，清空接口会删除其中所有图谱和向量数据
- `.env.local`、数据库文件、评测结果和课程报告默认不应提交到仓库

## 后续可扩展方向

- 支持批量文档上传
- 支持更稳定的结构化输出校验
- 为图谱关系增加去重和置信度
- 接入用户自定义知识库文件
