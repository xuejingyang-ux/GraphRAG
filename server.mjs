import express from "express";
import { createServer as createViteServer } from "vite";
import Database from "better-sqlite3";
import dotenv from "dotenv";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { existsSync } from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

dotenv.config();
dotenv.config({ path: ".env.local" });

const PORT = Number(process.env.PORT || 3000);
const ZHIPU_API_KEY = process.env.ZHIPU_API_KEY || "";
const ZHIPU_BASE_URL = process.env.ZHIPU_BASE_URL || "https://open.bigmodel.cn/api/paas/v4";
const ZHIPU_CHAT_MODEL = process.env.ZHIPU_CHAT_MODEL || "glm-4-flash";
const ZHIPU_REASONING_MODEL = process.env.ZHIPU_REASONING_MODEL || ZHIPU_CHAT_MODEL;
const ZHIPU_EMBEDDING_MODEL = process.env.ZHIPU_EMBEDDING_MODEL || "embedding-3";
const LOCAL_EMBEDDING_DIM = 256;
const DATA_DIR = path.join(process.cwd(), "data");
const DATASET_SCRIPT = path.join(process.cwd(), "scripts", "extract_dataset.py");

const db = new Database("graph_rag.db");

db.exec(`
  CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
  );

  CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT,
    description TEXT
  );

  CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES entities(id),
    FOREIGN KEY(target_id) REFERENCES entities(id)
  );

  CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
  CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
  CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
`);

function ensureApiKey() {
  if (!ZHIPU_API_KEY) {
    throw new Error("Missing ZHIPU_API_KEY environment variable.");
  }
}

function safeName(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function tokenizeText(text) {
  const normalized = safeName(text).toLowerCase();
  const chunks = normalized.match(/[\p{Script=Han}]{2,}|[\p{L}\p{N}]+/gu) || [];
  const tokens = [...chunks];
  for (const chunk of chunks) {
    if (/^[\p{Script=Han}]+$/u.test(chunk)) {
      for (let i = 0; i < chunk.length - 1; i += 1) {
        tokens.push(chunk.slice(i, i + 2));
      }
    }
  }
  return Array.from(new Set(tokens.filter(Boolean)));
}

function buildLocalEmbedding(text, dimension = LOCAL_EMBEDDING_DIM) {
  const vector = new Array(dimension).fill(0);
  const tokens = tokenizeText(text);

  for (const token of tokens) {
    let hash = 0;
    for (let i = 0; i < token.length; i += 1) {
      hash = (hash * 31 + token.charCodeAt(i)) >>> 0;
    }
    const index = hash % dimension;
    vector[index] += 1;
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!norm) {
    return vector;
  }

  return vector.map((value) => value / norm);
}

function cosineSimilarity(vecA, vecB) {
  if (!vecA.length || !vecB.length || vecA.length !== vecB.length) {
    return 0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i += 1) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (!normA || !normB) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function keywordScore(query, content) {
  const terms = tokenizeText(query);
  if (!terms.length) {
    return 0;
  }
  const haystack = safeName(content).toLowerCase();
  let hits = 0;
  for (const term of terms) {
    if (term.length >= 2 && haystack.includes(term)) {
      hits += 1;
    }
  }
  return hits / terms.length;
}

function decodeEmbedding(buffer) {
  return Array.from(
    new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Float32Array.BYTES_PER_ELEMENT),
  );
}

function encodeEmbedding(vector) {
  return Buffer.from(new Float32Array(vector).buffer);
}

function scoreDocument(docVector, queryEmbeddings, content, query) {
  const lexical = keywordScore(query, content);

  if (docVector.length === queryEmbeddings.remote.length && docVector.length > 0) {
    return cosineSimilarity(queryEmbeddings.remote, docVector) + lexical * 0.15;
  }

  if (docVector.length === queryEmbeddings.local.length && docVector.length > 0) {
    return cosineSimilarity(queryEmbeddings.local, docVector) + lexical * 0.25;
  }

  return lexical;
}

function extractJsonBlock(text) {
  const trimmed = text.trim();

  if (trimmed.startsWith("```")) {
    return trimmed.replace(/^```(?:json)?/i, "").replace(/```$/i, "").trim();
  }

  const arrayStart = trimmed.indexOf("[");
  const objectStart = trimmed.indexOf("{");
  const start =
    arrayStart === -1
      ? objectStart
      : objectStart === -1
        ? arrayStart
        : Math.min(arrayStart, objectStart);

  if (start === -1) {
    return trimmed;
  }

  const arrayEnd = trimmed.lastIndexOf("]");
  const objectEnd = trimmed.lastIndexOf("}");
  const end = Math.max(arrayEnd, objectEnd);

  return end >= start ? trimmed.slice(start, end + 1) : trimmed.slice(start);
}

function parseJsonResponse(text, fallback) {
  try {
    return JSON.parse(extractJsonBlock(text));
  } catch {
    return fallback;
  }
}

function normalizeMessageContent(content) {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && "text" in item) {
          return String(item.text || "");
        }
        return "";
      })
      .join("\n")
      .trim();
  }

  return "";
}

async function zhipuRequest(pathname, payload) {
  ensureApiKey();

  const response = await fetch(`${ZHIPU_BASE_URL}${pathname}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ZHIPU_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const rawText = await response.text();
  let parsed = null;

  try {
    parsed = rawText ? JSON.parse(rawText) : null;
  } catch {
    parsed = null;
  }

  if (!response.ok) {
    const message = parsed?.error?.message || parsed?.message || rawText || `Zhipu API request failed with status ${response.status}`;
    throw new Error(message);
  }

  return parsed;
}

async function createRemoteEmbedding(text) {
  const result = await zhipuRequest("/embeddings", {
    model: ZHIPU_EMBEDDING_MODEL,
    input: text,
  });

  const embedding = result?.data?.[0]?.embedding;
  if (!Array.isArray(embedding)) {
    throw new Error("Zhipu embeddings response is missing embedding data.");
  }

  return embedding;
}

async function createEmbedding(text, options = {}) {
  if (options.forceLocal) {
    return buildLocalEmbedding(text);
  }

  try {
    return await createRemoteEmbedding(text);
  } catch {
    return buildLocalEmbedding(text);
  }
}

async function createQueryEmbeddings(text) {
  let remote = [];
  try {
    remote = await createRemoteEmbedding(text);
  } catch {
    remote = [];
  }

  return {
    remote,
    local: buildLocalEmbedding(text),
  };
}

async function createChatCompletion(messages, model = ZHIPU_CHAT_MODEL) {
  const result = await zhipuRequest("/chat/completions", {
    model,
    temperature: 0.2,
    messages,
  });

  return normalizeMessageContent(result?.choices?.[0]?.message?.content);
}

function uniqueTriplets(triplets) {
  return Array.from(new Map(triplets.map((item) => [`${item.s}__${item.r}__${item.o}`, item])).values());
}

function extractTripletsByRules(text) {
  const triplets = [];
  const trimmed = safeName(text);

  let match = trimmed.match(/^《([^》]+)》由([^，。]+)执导并主演/);
  if (match) {
    triplets.push({ s: `《${match[1]}》`, r: "导演", o: match[2] });
    triplets.push({ s: `《${match[1]}》`, r: "主演", o: match[2] });
  }

  match = trimmed.match(/^《([^》]+)》由([^，。]+)自编自导自演/);
  if (match) {
    triplets.push({ s: `《${match[1]}》`, r: "导演", o: match[2] });
    triplets.push({ s: `《${match[1]}》`, r: "主演", o: match[2] });
  }

  match = trimmed.match(/^《([^》]+)》由([^，。]+)执导/);
  if (match) {
    triplets.push({ s: `《${match[1]}》`, r: "导演", o: match[2] });
  }

  match = trimmed.match(/^([^，。]+)通常被认为是([^，。]+)的常见别称/);
  if (match) {
    triplets.push({ s: match[1], r: "别称", o: match[2] });
  }

  match = trimmed.match(/^([^，。]+)在《([^》]+)》中饰演([^，。]+)/);
  if (match) {
    triplets.push({ s: match[1], r: "饰演", o: match[3] });
    triplets.push({ s: match[3], r: "出现于", o: `《${match[2]}》` });
  }

  match = trimmed.match(/^([^，。]+)是([^，。]+)的重要合作伙伴/);
  if (match) {
    triplets.push({ s: match[1], r: "合作伙伴", o: match[2] });
    triplets.push({ s: match[2], r: "合作伙伴", o: match[1] });
  }

  match = trimmed.match(/^([^，。]+)后来更名为([^，。]+)，简称([^，。]+)/);
  if (match) {
    triplets.push({ s: match[1], r: "更名为", o: match[2] });
    triplets.push({ s: match[2], r: "简称", o: match[3] });
  }

  match = trimmed.match(/^([^，。]+)成立于(\d{4}年)/);
  if (match) {
    triplets.push({ s: match[1], r: "成立于", o: match[2] });
  }

  match = trimmed.match(/^([^，。]+)，\d{4}年\d{1,2}月\d{1,2}日出生于([^，。]+)/);
  if (match) {
    triplets.push({ s: match[1], r: "出生地", o: match[2] });
  }

  return uniqueTriplets(triplets);
}

async function extractTriplets(text) {
  const content = await createChatCompletion([
    {
      role: "system",
      content: "你是知识图谱抽取助手。请只输出 JSON 数组，不要输出解释。每个元素必须是 {\"s\":\"主语\",\"r\":\"关系\",\"o\":\"宾语\"}。如果没有明确关系，返回 []。",
    },
    {
      role: "user",
      content: `请从下面文本中抽取实体关系三元组：\n${text}`,
    },
  ]);

  const modelTriplets = parseJsonResponse(content, []).filter(
    (item) => item && typeof item.s === "string" && typeof item.r === "string" && typeof item.o === "string" && item.s.trim() && item.r.trim() && item.o.trim(),
  );

  return uniqueTriplets([...modelTriplets, ...extractTripletsByRules(text)]);
}

async function alignEntities(newEntities, existingEntities) {
  if (!newEntities.length || !existingEntities.length) {
    return {};
  }

  const content = await createChatCompletion([
    {
      role: "system",
      content: "你是实体对齐助手。请只输出 JSON 对象，key 是新实体名，value 是已存在实体名。只有在两个名称明确指向同一实体时才映射，否则不要包含该 key。",
    },
    {
      role: "user",
      content: `新实体：${JSON.stringify(newEntities)}\n已存在实体：${JSON.stringify(existingEntities)}`,
    },
  ]);

  const mapping = parseJsonResponse(content, {});
  return Object.fromEntries(Object.entries(mapping).filter(([key, value]) => newEntities.includes(key) && existingEntities.includes(value)));
}

function extractQueryEntitiesByRules(query) {
  const entities = [];
  const movieMatches = safeName(query).match(/《[^》]+》/g) || [];
  entities.push(...movieMatches);

  const knownNames = ["周星驰", "星爷", "刘镇伟", "朱茵", "吴孟达", "紫霞仙子", "亚洲电视", "丽的电视", "ATV"];
  for (const name of knownNames) {
    if (query.includes(name)) {
      entities.push(name);
    }
  }

  return Array.from(new Set(entities));
}

async function extractQueryEntities(query) {
  const content = await createChatCompletion([
    {
      role: "system",
      content: "你是查询实体识别助手。请只输出 JSON 字符串数组，提取用户问题中的关键实体名；如果没有明显实体，返回 []。",
    },
    { role: "user", content: query },
  ]);

  const entities = parseJsonResponse(content, []).filter((item) => typeof item === "string" && item.trim());
  return Array.from(new Set([...entities, ...extractQueryEntitiesByRules(query)]));
}

async function answerQuestion(query, graphContext, documents) {
  const graphText = graphContext.length
    ? graphContext.map((item) => `(${item.s}) -[${item.r}]-> (${item.o})`).join("\n")
    : "暂无命中的图谱关系。";
  const documentText = documents.length ? documents.join("\n---\n") : "暂无命中的文档片段。";

  return createChatCompletion(
    [
      {
        role: "system",
        content: "你是 GraphRAG 问答助手。请优先基于给定的图谱关系和文档片段作答，回答使用中文；如果证据不足，请明确说明不确定，不要编造。",
      },
      {
        role: "user",
        content: `知识图谱上下文：\n${graphText}\n\n文档片段：\n${documentText}\n\n用户问题：${query}`,
      },
    ],
    ZHIPU_REASONING_MODEL,
  );
}

async function answerVectorOnlyQuestion(query, documents) {
  const documentText = documents.length ? documents.join("\n---\n") : "暂无命中的文档片段。";

  return createChatCompletion(
    [
      {
        role: "system",
        content: "你是纯向量RAG问答助手。你只能基于给定文档片段作答，回答使用中文；如果文档中没有足够证据，请明确说明无法回答，不要补充图谱推理和外部常识。",
      },
      {
        role: "user",
        content: `文档片段：\n${documentText}\n\n用户问题：${query}`,
      },
    ],
    ZHIPU_REASONING_MODEL,
  );
}

function clearDatabase() {
  db.prepare("DELETE FROM relationships").run();
  db.prepare("DELETE FROM entities").run();
  db.prepare("DELETE FROM documents").run();
}

function runDatasetExtractor() {
  if (!existsSync(DATA_DIR)) {
    throw new Error("data 目录不存在，请先将数据集放入项目根目录的 data 文件夹。");
  }
  if (!existsSync(DATASET_SCRIPT)) {
    throw new Error("缺少数据集解析脚本 scripts/extract_dataset.py。");
  }

  const raw = execFileSync("python", [DATASET_SCRIPT, DATA_DIR], {
    cwd: process.cwd(),
    encoding: "utf-8",
    maxBuffer: 20 * 1024 * 1024,
  });
  return JSON.parse(raw.replace(/^\uFEFF/, ""));
}

function importDataset(dataset) {
  const insertEntity = db.prepare("INSERT OR IGNORE INTO entities (name, type) VALUES (?, ?)");
  const getEntityId = db.prepare("SELECT id FROM entities WHERE name = ?");
  const insertRelation = db.prepare("INSERT INTO relationships (source_id, target_id, relation) VALUES (?, ?, ?)");
  const insertDocument = db.prepare("INSERT INTO documents (content, embedding) VALUES (?, ?)");

  const relationSeen = new Set();
  const documentSeen = new Set();

  const saveRelation = (source, relation, target, sourceType = "entity", targetType = "entity") => {
    const s = safeName(source);
    const r = safeName(relation);
    const t = safeName(target);
    if (!s || !r || !t) {
      return;
    }

    insertEntity.run(s, sourceType);
    insertEntity.run(t, targetType);
    const sourceRow = getEntityId.get(s);
    const targetRow = getEntityId.get(t);
    if (!sourceRow || !targetRow) {
      return;
    }

    const key = `${sourceRow.id}__${r}__${targetRow.id}`;
    if (relationSeen.has(key)) {
      return;
    }
    relationSeen.add(key);
    insertRelation.run(sourceRow.id, targetRow.id, r);
  };

  const saveDocument = (content) => {
    const normalized = safeName(content);
    if (!normalized || documentSeen.has(normalized)) {
      return;
    }
    documentSeen.add(normalized);
    insertDocument.run(normalized, encodeEmbedding(buildLocalEmbedding(normalized)));
  };

  const importAll = db.transaction(() => {
    clearDatabase();

    for (const movie of dataset.movies || []) {
      const title = safeName(movie.title);
      const movieNode = title.startsWith("《") ? title : `《${title}》`;
      const director = safeName(movie.director);
      const year = safeName(movie.year);
      const englishTitle = safeName(movie.english_title);
      const genres = Array.isArray(movie.genres) ? movie.genres.map(safeName).filter(Boolean) : [];
      const actors = Array.isArray(movie.actors) ? movie.actors.map(safeName).filter(Boolean) : [];
      const score = safeName(movie.score);

      insertEntity.run(movieNode, "movie");
      if (director) {
        saveRelation(movieNode, "导演", director, "movie", "person");
        saveRelation(director, "执导", movieNode, "person", "movie");
      }
      if (year) {
        saveRelation(movieNode, "上映年份", year, "movie", "year");
      }
      if (englishTitle) {
        saveRelation(movieNode, "英文名", englishTitle, "movie", "title");
      }
      for (const genre of genres) {
        saveRelation(movieNode, "类型", genre, "movie", "genre");
      }
      for (const actor of actors) {
        saveRelation(movieNode, "主演", actor, "movie", "person");
        saveRelation(actor, "参演", movieNode, "person", "movie");
      }

      const movieText = [
        `电影${movieNode}`,
        year ? `上映于${year}年` : "",
        director ? `导演是${director}` : "",
        actors.length ? `主演包括${actors.join("、")}` : "",
        genres.length ? `类型为${genres.join("、")}` : "",
        englishTitle ? `英文名为${englishTitle}` : "",
        score ? `时光网评分为${score}` : "",
      ].filter(Boolean).join("，") + "。";
      saveDocument(movieText);
    }

    for (const edge of dataset.edges || []) {
      const source = safeName(edge.source);
      const target = safeName(edge.target);
      const year = safeName(edge.year);
      saveRelation(source, "合作演员", target, "person", "person");
      saveRelation(target, "合作演员", source, "person", "person");
      const edgeText = year
        ? `${source} 与 ${target} 在 ${year} 年存在电影合作关系。`
        : `${source} 与 ${target} 存在电影合作关系。`;
      saveDocument(edgeText);
    }
  });

  importAll();

  return {
    movies: dataset.movies?.length || 0,
    edges: dataset.edges?.length || 0,
    documents: db.prepare("SELECT COUNT(*) AS count FROM documents").get().count,
    entities: db.prepare("SELECT COUNT(*) AS count FROM entities").get().count,
    relationships: db.prepare("SELECT COUNT(*) AS count FROM relationships").get().count,
  };
}

function getTopDocuments(query, limit = 3) {
  const queryPromise = createQueryEmbeddings(query);
  return queryPromise.then((queryEmbeddings) => {
    const allDocs = db.prepare("SELECT id, content, embedding FROM documents").all();
    return allDocs
      .map((doc) => ({
        content: doc.content,
        score: scoreDocument(decodeEmbedding(doc.embedding), queryEmbeddings, doc.content, query),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  });
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "50mb" }));

  app.post("/api/ingest", async (req, res) => {
    const { text } = req.body;
    if (!text?.trim()) {
      return res.status(400).json({ error: "Text is required" });
    }

    try {
      const embedding = await createEmbedding(text);
      const triplets = await extractTriplets(text);
      const newEntities = [...new Set(triplets.flatMap((item) => [item.s, item.o]).map(safeName).filter(Boolean))];
      const existingEntities = db.prepare("SELECT name FROM entities").all().map((item) => item.name);
      const alignmentMap = await alignEntities(newEntities, existingEntities);

      const insertEntity = db.prepare("INSERT OR IGNORE INTO entities (name) VALUES (?)");
      const getEntityId = db.prepare("SELECT id FROM entities WHERE name = ?");
      const insertRelation = db.prepare("INSERT INTO relationships (source_id, target_id, relation) VALUES (?, ?, ?)");
      const insertDocument = db.prepare("INSERT INTO documents (content, embedding) VALUES (?, ?)");

      db.transaction(() => {
        for (const triplet of triplets) {
          const subject = alignmentMap[triplet.s] || triplet.s;
          const object = alignmentMap[triplet.o] || triplet.o;
          insertEntity.run(subject);
          insertEntity.run(object);
          const subjectRow = getEntityId.get(subject);
          const objectRow = getEntityId.get(object);
          if (!subjectRow || !objectRow) {
            continue;
          }
          insertRelation.run(subjectRow.id, objectRow.id, triplet.r);
        }
        insertDocument.run(safeName(text), encodeEmbedding(embedding));
      })();

      return res.json({ success: true, tripletsCount: triplets.length, alignedEntities: alignmentMap });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: error instanceof Error ? error.message : "Failed to ingest data" });
    }
  });

  app.post("/api/ingest-dataset", async (_req, res) => {
    try {
      const dataset = runDatasetExtractor();
      const summary = importDataset(dataset);
      return res.json({ success: true, ...summary });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: error instanceof Error ? error.message : "Dataset ingest failed" });
    }
  });

  app.post("/api/query", async (req, res) => {
    const { query } = req.body;
    if (!query?.trim()) {
      return res.status(400).json({ error: "Query is required" });
    }

    try {
      const scoredDocs = await getTopDocuments(query);
      const queryEntities = await extractQueryEntities(query);
      const graphContext = [];
      const visitedRelationIds = new Set();
      const findEntity = db.prepare("SELECT id, name FROM entities WHERE name LIKE ? LIMIT 1");
      const graphQuery = db.prepare(`
        WITH RECURSIVE neighbors(id, depth) AS (
          SELECT ? as id, 0 as depth
          UNION
          SELECT CASE WHEN r.source_id = n.id THEN r.target_id ELSE r.source_id END, n.depth + 1
          FROM relationships r
          JOIN neighbors n ON r.source_id = n.id OR r.target_id = n.id
          WHERE n.depth < 2
        )
        SELECT DISTINCT e1.name as s, r.relation as r, e2.name as o, r.id as relId, e1.id as sId, e2.id as oId
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        JOIN neighbors n ON r.source_id = n.id OR r.target_id = n.id
      `);

      for (const entityName of queryEntities) {
        const entity = findEntity.get(`%${entityName}%`);
        if (!entity) {
          continue;
        }

        for (const relation of graphQuery.all(entity.id)) {
          if (visitedRelationIds.has(relation.relId)) {
            continue;
          }
          visitedRelationIds.add(relation.relId);
          graphContext.push(relation);
        }
      }

      const answer = await answerQuestion(query, graphContext, scoredDocs.map((item) => item.content));
      return res.json({ answer, graph: graphContext, sources: scoredDocs.map((item) => item.content), entities: queryEntities });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: error instanceof Error ? error.message : "Query failed" });
    }
  });

  app.post("/api/query-vector", async (req, res) => {
    const { query } = req.body;
    if (!query?.trim()) {
      return res.status(400).json({ error: "Query is required" });
    }

    try {
      const scoredDocs = await getTopDocuments(query);
      const answer = await answerVectorOnlyQuestion(query, scoredDocs.map((item) => item.content));
      return res.json({ answer, graph: [], sources: scoredDocs.map((item) => item.content), mode: "vector-only" });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: error instanceof Error ? error.message : "Vector query failed" });
    }
  });

  app.get("/api/graph", (_req, res) => {
    const nodes = db.prepare("SELECT id, name, COALESCE(type, 'entity') as type FROM entities").all();
    const links = db.prepare("SELECT source_id as source, target_id as target, relation as label FROM relationships").all();
    res.json({ nodes, links });
  });

  app.get("/api/dataset-status", (_req, res) => {
    res.json({
      dataDirExists: existsSync(DATA_DIR),
      entityCount: db.prepare("SELECT COUNT(*) AS count FROM entities").get().count,
      documentCount: db.prepare("SELECT COUNT(*) AS count FROM documents").get().count,
      relationshipCount: db.prepare("SELECT COUNT(*) AS count FROM relationships").get().count,
    });
  });

  app.post("/api/clear", (_req, res) => {
    try {
      clearDatabase();
      return res.json({ success: true });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Failed to clear database" });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      configFile: false,
      plugins: [react(), tailwindcss()],
      define: {
        "process.env.ZHIPU_API_KEY": JSON.stringify(process.env.ZHIPU_API_KEY || ""),
      },
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
