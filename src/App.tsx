import React, { useEffect, useRef, useState } from 'react';
import { Send, Database, Search, Loader2, Info, ChevronRight, Network, BookOpen, Trash2 } from 'lucide-react';
import * as d3 from 'd3';
import { motion } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface GraphRelation {
  s: string;
  r: string;
  o: string;
  sId: number;
  oId: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  graph?: GraphRelation[];
  sources?: string[];
}

interface Node extends d3.SimulationNodeDatum {
  id: number;
  name: string;
  type: string;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: number | Node;
  target: number | Node;
  label: string;
}

const SAMPLE_TEXTS = [
  '周星驰，1962年6月22日出生于中国香港，祖籍浙江宁波，是演员、导演、编剧、制作人。',
  '《大话西游》由刘镇伟执导，周星驰、朱茵、吴孟达等主演，是华语经典奇幻喜剧电影。',
  '刘镇伟，英文名 Jeffrey Lau，是中国香港导演、编剧、制作人。',
  '《功夫》由周星驰执导并主演，是一部融合动作与喜剧风格的电影。',
  '星爷通常被认为是周星驰的常见别称。',
  '《少林足球》由周星驰自编自导自演，将功夫元素与足球运动结合。',
  '吴孟达是周星驰的重要合作伙伴，两人共同出演过多部经典电影。',
  '朱茵在《大话西游》中饰演紫霞仙子，这一角色广受欢迎。',
  '西安电影制片厂成立于1958年，是中国重要的电影制片机构之一。',
  '丽的电视后来更名为亚洲电视，简称 ATV。',
];

const INITIAL_STEPS = [
  '正在识别问题中的核心实体',
  '正在检索两跳图谱关系',
  '正在执行向量相似度召回',
  '正在汇总证据并生成回答',
];

const GraphView = ({ data }: { data: { nodes: Node[]; links: Link[] } }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) {
      return;
    }

    const width = svgRef.current.clientWidth || 400;
    const height = svgRef.current.clientHeight || 300;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.append('g');
    const simulation = d3
      .forceSimulation<Node>(data.nodes)
      .force('link', d3.forceLink<Node, Link>(data.links).id((d) => d.id).distance(110))
      .force('charge', d3.forceManyBody().strength(-260))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const link = container
      .append('g')
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('stroke', '#94a3b8')
      .attr('stroke-opacity', 0.75)
      .attr('stroke-width', 1.5);

    const linkText = container
      .append('g')
      .selectAll('text')
      .data(data.links)
      .join('text')
      .attr('font-size', '10px')
      .attr('fill', '#64748b')
      .attr('text-anchor', 'middle')
      .text((d) => d.label);

    const node = container
      .append('g')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', 9)
      .attr('fill', '#10b981')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .call(
        d3
          .drag<SVGCircleElement, Node>()
          .on('start', (event) => {
            if (!event.active) {
              simulation.alphaTarget(0.3).restart();
            }
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
          })
          .on('drag', (event) => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
          })
          .on('end', (event) => {
            if (!event.active) {
              simulation.alphaTarget(0);
            }
            event.subject.fx = null;
            event.subject.fy = null;
          }) as any,
      );

    const label = container
      .append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .attr('font-size', '12px')
      .attr('fill', '#0f172a')
      .attr('font-weight', '600')
      .attr('dx', 12)
      .attr('dy', 4)
      .text((d) => d.name);

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as Node).x ?? 0)
        .attr('y1', (d) => (d.source as Node).y ?? 0)
        .attr('x2', (d) => (d.target as Node).x ?? 0)
        .attr('y2', (d) => (d.target as Node).y ?? 0);

      linkText
        .attr('x', (d) => (((d.source as Node).x ?? 0) + ((d.target as Node).x ?? 0)) / 2)
        .attr('y', (d) => (((d.source as Node).y ?? 0) + ((d.target as Node).y ?? 0)) / 2);

      node.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0);
      label.attr('x', (d) => d.x ?? 0).attr('y', (d) => d.y ?? 0);
    });

    svg.call(
      d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.4, 4]).on('zoom', (event) => {
        container.attr('transform', event.transform);
      }),
    );

    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-inner">
      <div className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-3 py-1.5 shadow-sm backdrop-blur-md">
        <Network className="h-4 w-4 text-emerald-600" />
        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">知识图谱视图</span>
      </div>
      <svg ref={svgRef} className="h-full w-full cursor-move" />
    </div>
  );
};

function buildGraph(graph: GraphRelation[]) {
  const nodes = Array.from(
    new Map(
      graph
        .flatMap((item) => [
          [item.sId, { id: item.sId, name: item.s, type: 'entity' }],
          [item.oId, { id: item.oId, name: item.o, type: 'entity' }],
        ])
        .map(([id, node]) => [id, node]),
    ).values(),
  ) as Node[];

  const links = graph.map((item) => ({
    source: item.sId,
    target: item.oId,
    label: item.r,
  })) as Link[];

  return { nodes, links };
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: '你好，我是 GraphRAG 问答助手。你可以先导入示例知识，再询问人物、作品和关系问题。',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [thinkingSteps, setThinkingSteps] = useState<string[]>([]);
  const [activeGraph, setActiveGraph] = useState<{ nodes: Node[]; links: Link[] }>({ nodes: [], links: [] });
  const [activeSources, setActiveSources] = useState<string[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!input.trim() || loading) {
      return;
    }

    const userMsg = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);
    setThinkingSteps(INITIAL_STEPS);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || '查询失败');
      }

      const graph = Array.isArray(data.graph) ? (data.graph as GraphRelation[]) : [];
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || '没有生成回答。',
          graph,
          sources: data.sources || [],
        },
      ]);
      setActiveGraph(buildGraph(graph));
      setActiveSources(data.sources || []);
    } catch (error) {
      const message = error instanceof Error ? error.message : '查询过程中出现错误';
      setMessages((prev) => [...prev, { role: 'assistant', content: `抱歉，${message}` }]);
    } finally {
      setLoading(false);
      setThinkingSteps([]);
    }
  };

  const handleIngestSample = async () => {
    if (loading) {
      return;
    }

    setLoading(true);
    setThinkingSteps(['正在导入示例数据', '正在抽取实体关系', '正在写入向量与图谱']);

    try {
      for (const text of SAMPLE_TEXTS) {
        const response = await fetch('/api/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || '示例导入失败');
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '示例知识已导入完成。你现在可以提问，例如“周星驰和功夫有什么关系？”',
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : '示例导入失败';
      setMessages((prev) => [...prev, { role: 'assistant', content: `导入失败：${message}` }]);
    } finally {
      setLoading(false);
      setThinkingSteps([]);
    }
  };

  const handleClearDB = async () => {
    if (loading) {
      return;
    }

    const confirmed = window.confirm('确定要清空已经构建的知识图谱和向量数据吗？该操作不可恢复。');
    if (!confirmed) {
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/clear', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || '清空数据库失败');
      }

      setMessages([
        {
          role: 'assistant',
          content: '知识库已经清空。你可以重新导入样本或开始新一轮构建。',
        },
      ]);
      setActiveGraph({ nodes: [], links: [] });
      setActiveSources([]);
    } catch (error) {
      const message = error instanceof Error ? error.message : '清空数据库失败';
      setMessages((prev) => [...prev, { role: 'assistant', content: `清空失败：${message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 selection:bg-emerald-100 selection:text-emerald-900">
      <aside className="flex w-80 flex-col border-r border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-100 p-6">
          <div className="mb-2 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 shadow-lg shadow-emerald-600/20">
              <Database className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-800">GraphRAG</h1>
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">增强型知识图谱问答</p>
            </div>
          </div>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          <button
            onClick={handleIngestSample}
            className="group w-full rounded-xl border border-slate-200 bg-slate-50 p-4 text-left transition-all hover:border-emerald-200 hover:bg-slate-100"
          >
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">系统初始化</span>
              <ChevronRight className="h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-1" />
            </div>
            <p className="text-sm font-semibold text-slate-700">导入示例知识库</p>
            <p className="mt-1 text-xs text-slate-500">加载样本文本并自动构建向量与图谱。</p>
          </button>

          <button
            onClick={handleClearDB}
            className="group w-full rounded-xl border border-slate-200 bg-slate-50 p-4 text-left transition-all hover:border-red-200 hover:bg-red-50"
          >
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-red-600">数据管理</span>
              <Trash2 className="h-4 w-4 text-red-400" />
            </div>
            <p className="text-sm font-semibold text-slate-700">清空知识库</p>
            <p className="mt-1 text-xs text-slate-500">重置全部文档、实体关系与向量索引。</p>
          </button>

          <div className="pt-4">
            <h3 className="mb-3 px-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">系统状态</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <span className="text-xs text-slate-600">向量索引（智谱）</span>
                <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-600">已接入</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <span className="text-xs text-slate-600">图谱检索（2-Hop）</span>
                <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-600">可用</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <span className="text-xs text-slate-600">实体对齐</span>
                <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-600">启用中</span>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-100 p-6">
          <div className="flex items-center gap-3 text-slate-400">
            <Info className="h-4 w-4" />
            <span className="text-xs">版本 v1.1.0</span>
          </div>
        </div>
      </aside>

      <main className="relative flex flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-20 items-center justify-between border-b border-slate-200 bg-white/80 px-8 shadow-sm backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <div className="flex -space-x-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-white bg-emerald-600 text-[10px] font-bold text-white shadow-sm">AI</div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-white bg-indigo-600 text-[10px] font-bold text-white shadow-sm">KG</div>
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-800">混合检索推理引擎</h2>
              <p className="text-[10px] font-medium text-slate-500">结合向量召回、实体对齐与图谱路径推理</p>
            </div>
          </div>
          <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-500 shadow-sm">
            智谱 GLM + SQLite
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <section className="flex flex-1 flex-col">
            <div className="flex-1 space-y-8 overflow-y-auto p-8">
              {messages.map((msg, index) => (
                <motion.div
                  key={`${msg.role}-${index}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn('flex max-w-3xl gap-4', msg.role === 'user' ? 'ml-auto flex-row-reverse' : '')}
                >
                  <div
                    className={cn(
                      'mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg shadow-sm',
                      msg.role === 'user'
                        ? 'bg-slate-800 text-white'
                        : 'border border-emerald-200 bg-emerald-100 text-emerald-700',
                    )}
                  >
                    {msg.role === 'user' ? <span className="text-[10px] font-bold">用户</span> : <Database className="h-4 w-4" />}
                  </div>
                  <div className={cn('space-y-4', msg.role === 'user' ? 'text-right' : '')}>
                    <div
                      className={cn(
                        'rounded-2xl border p-4 text-sm leading-relaxed shadow-sm',
                        msg.role === 'user'
                          ? 'rounded-tr-none border-emerald-500 bg-emerald-600 text-white'
                          : 'rounded-tl-none border-slate-200 bg-white text-slate-700',
                      )}
                    >
                      {msg.content}
                    </div>

                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.map((_, sourceIndex) => (
                          <div
                            key={sourceIndex}
                            className="flex items-center gap-2 rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-[10px] text-slate-500"
                          >
                            <BookOpen className="h-3 w-3" />
                            参考片段 {sourceIndex + 1}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {loading && (
                <div className="flex max-w-3xl gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100">
                    <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
                  </div>
                  <div className="min-w-[220px] rounded-2xl rounded-tl-none border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="mb-3 flex items-center gap-2">
                      <span className="text-xs font-medium text-slate-500">正在处理中...</span>
                    </div>
                    <div className="space-y-1.5">
                      {thinkingSteps.map((step, index) => (
                        <motion.div
                          key={step}
                          initial={{ opacity: 0, x: -5 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.12 }}
                          className="flex items-center gap-2 text-[11px] text-slate-400"
                        >
                          <div className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                          {step}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="p-8 pt-0">
              <form onSubmit={handleSubmit} className="group relative">
                <div className="absolute inset-0 opacity-0 blur-2xl transition-opacity group-focus-within:opacity-100 bg-emerald-500/10" />
                <div className="relative flex items-center rounded-2xl border border-slate-200 bg-white p-2 shadow-xl transition-all focus-within:border-emerald-500 focus-within:ring-4 focus-within:ring-emerald-500/5">
                  <input
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="例如：周星驰和《功夫》是什么关系？"
                    className="flex-1 border-none bg-transparent px-4 py-2 text-sm text-slate-700 placeholder:text-slate-400 focus:ring-0"
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-lg shadow-emerald-600/20 transition-all hover:bg-emerald-500 disabled:bg-slate-100 disabled:text-slate-400"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </form>
              <p className="mt-4 text-center text-[10px] font-bold uppercase tracking-widest text-slate-400">
                由智谱 GLM 与 SQLite 图谱引擎驱动
              </p>
            </div>
          </section>

          <aside className="flex w-[450px] flex-col gap-6 border-l border-slate-200 bg-slate-50/50 p-6 backdrop-blur-sm">
            <div className="flex min-h-0 flex-1 flex-col">
              <GraphView data={activeGraph} />
            </div>

            <div className="flex h-48 min-h-0 flex-col">
              <div className="mb-3 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-indigo-600" />
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">知识证据片段</span>
              </div>
              <div className="custom-scrollbar flex-1 space-y-3 overflow-y-auto pr-2">
                {activeSources.length > 0 ? (
                  activeSources.map((source, index) => (
                    <motion.div
                      key={`${index}-${source.slice(0, 12)}`}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.08 }}
                      className="rounded-xl border border-slate-200 bg-white p-3 text-[11px] leading-relaxed text-slate-600 shadow-sm"
                    >
                      {source}
                    </motion.div>
                  ))
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-2 text-slate-300">
                    <Search className="h-8 w-8 opacity-40" />
                    <p className="text-[10px] font-bold uppercase tracking-widest">暂无证据片段</p>
                  </div>
                )}
              </div>
            </div>
          </aside>
        </div>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(15, 23, 42, 0.12);
          border-radius: 9999px;
        }
      `}</style>
    </div>
  );
}
