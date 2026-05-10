from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

# 停用词
STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
    "一个", "一种", "进行", "通过", "以及", "可以", "需要", "包括", "系统", "教材", "内容",
    "知识", "学习", "学生", "教师", "章节", "相关", "主要", "能够", "不同", "使用",
    "方法", "理论", "概念", "研究", "分析", "过程", "情况", "问题", "发展",
}

# 关系类型映射
RELATION_TYPE_LABELS = {
    "prerequisite": "前置依赖",
    "contains": "包含关系",
    "parallel": "并列关系",
    "related": "相关关系",
    "application": "应用关系",
    "applies_to": "应用关系",
}

# 教材颜色映射
TEXTBOOK_COLORS = [
    "#e74c3c",  # 红色
    "#3498db",  # 蓝色
    "#2ecc71",  # 绿色
    "#f39c12",  # 橙色
    "#9b59b6",  # 紫色
    "#1abc9c",  # 青色
    "#e91e63",  # 粉色
]


def extract_terms(text: str, limit: int = 8) -> list[str]:
    """提取高频知识点"""
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
    normalized = [word.strip().lower() for word in words if word.strip()]
    terms = [word for word in normalized if word not in STOPWORDS and len(word) >= 2]
    counts = Counter(terms)
    return [term for term, _ in counts.most_common(limit)]


def build_knowledge_graph(
    textbooks: list[dict[str, Any]],
    max_terms_per_chapter: int = 6,
) -> dict[str, Any]:
    """
    构建知识图谱

    Returns:
        {"nodes": [...], "edges": [...], "metadata": {...}}
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[tuple[str, str]] = set()

    # 统计知识点在多本教材中的出现频次
    term_frequency: dict[str, int] = defaultdict(int)
    term_sources: dict[str, set[str]] = defaultdict(set)

    for book in textbooks:
        filename = book.get("filename", "-")
        for chapter in book.get("chapters", []):
            terms = extract_terms(chapter.get("content", ""), limit=max_terms_per_chapter)
            for term in terms:
                term_frequency[term] += 1
                term_sources[term].add(filename)

    # 颜色分配
    textbook_colors: dict[str, str] = {}
    color_idx = 0
    for book in textbooks:
        filename = book.get("filename", "-")
        if filename not in textbook_colors:
            textbook_colors[filename] = TEXTBOOK_COLORS[color_idx % len(TEXTBOOK_COLORS)]
            color_idx += 1

    # 构建节点和边
    for book in textbooks:
        textbook_id = book.get("textbook_id", "book")
        filename = book.get("filename", "-")
        book_color = textbook_colors.get(filename, "#95a5a6")
        previous_main_term: str | None = None

        for chapter in book.get("chapters", []):
            chapter_id = chapter.get("chapter_id", "ch")
            terms = extract_terms(chapter.get("content", ""), limit=max_terms_per_chapter)
            if not terms:
                terms = [chapter.get("title", chapter_id)]

            # 章节节点
            chapter_node_id = f"{textbook_id}_{chapter_id}"
            freq = term_frequency.get(terms[0], 1)
            nodes.append({
                "id": chapter_node_id,
                "name": chapter.get("title", chapter_id),
                "category": "chapter",
                "chapter": chapter.get("title", "-"),
                "source_textbook": filename,
                "color": book_color,
                "page": chapter.get("page_start", "-"),
                "definition": f"{filename} 章节结构节点",
                "size": 28,
                "frequency": freq,
            })

            # 知识点节点
            main_term = terms[0]
            if previous_main_term:
                edges.append({
                    "source": f"{textbook_id}_term_{previous_main_term}",
                    "target": f"{textbook_id}_term_{main_term}",
                    "relation_type": "prerequisite",
                    "description": "相邻章节核心概念的前置依赖",
                })
            previous_main_term = main_term

            for idx, term in enumerate(terms):
                node_key = (textbook_id, term)
                node_id = f"{textbook_id}_term_{term}"
                if node_key not in seen_nodes:
                    freq = term_frequency.get(term, 1)
                    source_count = len(term_sources.get(term, {filename}))
                    nodes.append({
                        "id": node_id,
                        "name": term,
                        "category": "core" if idx < 3 else "general",
                        "chapter": chapter.get("title", "-"),
                        "source_textbook": filename,
                        "color": book_color,
                        "page": chapter.get("page_start", "-"),
                        "definition": f"从《{chapter.get('title', '-')}》抽取",
                        "size": 14 + min(freq * 3, 16),
                        "frequency": freq,
                        "source_count": source_count,
                    })
                    seen_nodes.add(node_key)

                edges.append({
                    "source": chapter_node_id,
                    "target": node_id,
                    "relation_type": "contains",
                    "description": "章节包含该知识点",
                })

            # 同章节知识点关系
            for left, right in zip(terms, terms[1:]):
                edges.append({
                    "source": f"{textbook_id}_term_{left}",
                    "target": f"{textbook_id}_term_{right}",
                    "relation_type": "related",
                    "description": "同一章节中共同出现",
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "textbook_colors": textbook_colors,
            "term_frequency": dict(term_frequency),
        },
    }


def graph_to_html(graph: dict[str, Any]) -> str:
    """
    生成交互式知识图谱 HTML

    支持：
    - 节点点击查看详情
    - 频次可视化（大小/颜色深度）
    - 教材来源区分
    - 搜索高亮
    - 缩放拖拽
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    metadata = graph.get("metadata", {})
    textbook_colors = metadata.get("textbook_colors", {})
    term_frequency = metadata.get("term_frequency", {})

    # 统计节点频次
    max_freq = max((n.get("frequency", 1) for n in nodes), default=1)

    # 节点颜色按教材区分
    node_jsons = []
    for node in nodes[:100]:
        freq = node.get("frequency", 1)
        # 频次越高颜色越深
        opacity = 0.4 + (freq / max_freq) * 0.6
        color = node.get("color", "#64748b")
        # 转换为带透明度的颜色
        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            rgba_color = f"rgba({r},{g},{b},{opacity})"
        else:
            rgba_color = color

        category = node.get("category", "concept")
        size = node.get("size", 16)

        node_jsons.append({
            "id": str(node["id"]),
            "label": str(node.get("name", "-")),
            "name": str(node.get("name", "-")),
            "category": str(category),
            "chapter": str(node.get("chapter", "-")),
            "textbook": str(node.get("source_textbook", "-")),
            "page": str(node.get("page", "-")),
            "definition": str(node.get("definition", "-")),
            "frequency": freq,
            "sourceCount": node.get("source_count", 1),
            "size": size,
            "color": rgba_color,
            "fontColor": "#1e293b",
            "fontSize": 14,
            "borderWidth": 2,
            "borderColor": node.get("color", "#64748b"),
            "title": f"{node.get('name', '-')} ({node.get('source_textbook', '-')})",
        })

    # 边数据
    edge_jsons = []
    relation_colors = {
        "prerequisite": "#ef4444",
        "contains": "#3b82f6",
        "parallel": "#10b981",
        "related": "#6b7280",
        "application": "#f59e0b",
    }
    for edge in edges[:200]:
        rel_type = edge.get("relation_type", "related")
        edge_jsons.append({
            "from": str(edge.get("source", "")),
            "to": str(edge.get("target", "")),
            "relation": rel_type,
            "label": RELATION_TYPE_LABELS.get(rel_type, rel_type),
            "color": {"color": relation_colors.get(rel_type, "#94a3b8")},
            "width": 1.5,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "title": str(edge.get("description", "")),
        })

    # 教材图例
    legend_items = "".join(
        f'<div class="legend-item"><span class="legend-color" style="background:{color}"></span>{name}</div>'
        for name, color in textbook_colors.items()
    )
    node_data = json.dumps(node_jsons[:80], ensure_ascii=False)
    edge_data = json.dumps(edge_jsons[:150], ensure_ascii=False)

    return f"""
<style>
  .kg-container {{
    display: flex;
    flex-direction: column;
    height: 650px;
    font-family: "Microsoft YaHei", Arial, sans-serif;
    background: #f8fafc;
    border-radius: 12px;
    overflow: hidden;
  }}
  .kg-toolbar {{
    display: flex;
    gap: 12px;
    padding: 12px 16px;
    background: white;
    border-bottom: 1px solid #e2e8f0;
    align-items: center;
  }}
  .kg-search {{
    flex: 1;
    max-width: 280px;
    padding: 8px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-size: 14px;
  }}
  .kg-search:focus {{
    outline: 2px solid #3b82f6;
    border-color: transparent;
  }}
  .kg-legend {{
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: white;
    border-bottom: 1px solid #e2e8f0;
    font-size: 12px;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-color {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}
  .kg-body {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}
  #mynetwork {{
    flex: 1;
    height: 100%;
  }}
  .kg-panel {{
    width: 320px;
    background: white;
    border-left: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
  }}
  .kg-panel-header {{
    padding: 12px 16px;
    background: #0f172a;
    color: #e2e8f0;
    font-weight: 600;
    font-size: 13px;
  }}
  .kg-panel-body {{
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }}
  .node-detail {{
    display: none;
  }}
  .node-detail.active {{
    display: block;
  }}
  .detail-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }}
  .detail-title {{
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
  }}
  .detail-meta {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px 12px;
    font-size: 13px;
    margin-bottom: 12px;
  }}
  .detail-meta dt {{
    color: #64748b;
  }}
  .detail-meta dd {{
    color: #1e293b;
    margin: 0;
  }}
  .detail-def {{
    font-size: 13px;
    color: #475569;
    line-height: 1.6;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
  }}
  .detail-stats {{
    display: flex;
    gap: 16px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
  }}
  .stat-item {{
    text-align: center;
  }}
  .stat-value {{
    font-size: 20px;
    font-weight: 700;
    color: #3b82f6;
  }}
  .stat-label {{
    font-size: 11px;
    color: #64748b;
  }}
  .kg-stats {{
    padding: 12px 16px;
    background: #f1f5f9;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    color: #64748b;
    display: flex;
    gap: 20px;
  }}
  .kg-stats span {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .highlight-node {{
    background: #fef3c7 !important;
    border-color: #f59e0b !important;
  }}
</style>

<div class="kg-container">
  <div class="kg-toolbar">
    <input type="text" class="kg-search" id="searchInput" placeholder="搜索知识点…" />
    <span style="font-size:12px;color:#64748b;">点击节点查看详情 | 滚轮缩放 | 拖拽移动</span>
  </div>
  <div class="kg-legend">
    {legend_items or '<span style="color:#64748b;">暂无教材</span>'}
    <span style="margin-left:auto;color:#64748b;">节点大小 = 频次</span>
  </div>
  <div class="kg-body">
    <div id="mynetwork"></div>
    <aside class="kg-panel">
      <div class="kg-panel-header">📋 节点详情</div>
      <div class="kg-panel-body">
        <div id="nodeDetail" class="node-detail active">
          <p style="color:#94a3b8;text-align:center;margin-top:40px;">
            点击图谱中的节点查看详情
          </p>
        </div>
        <div id="nodeInfo"></div>
      </div>
    </aside>
  </div>
  <div class="kg-stats">
    <span>📊 节点: {len(nodes)}</span>
    <span>🔗 关系: {len(edges)}</span>
    <span>📚 教材: {len(textbook_colors)}</span>
  </div>
</div>

<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
const nodes = new vis.DataSet([{{id: "placeholder", label: "加载中…"}}]);
const edges = new vis.DataSet([]);

const container = document.getElementById('mynetwork');
const data = {{ nodes: nodes, edges: edges }};
const options = {{
    nodes: {{
        shape: 'dot',
        scaling: {{ min: 10, max: 30 }},
        font: {{ size: 14, face: 'Microsoft YaHei' }},
    }},
    edges: {{
        smooth: {{ type: 'continuous' }},
        font: {{ size: 10, align: 'middle', face: 'Microsoft YaHei' }},
    }},
    physics: {{
        stabilization: {{ iterations: 100 }},
        barnesHut: {{ gravitationalConstant: -2000, springLength: 150 }},
    }},
    interaction: {{
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
    }},
    layout: {{ improvedLayout: true }},
}};

const network = new vis.Network(container, data, options);

// 搜索功能
const searchInput = document.getElementById('searchInput');
searchInput.addEventListener('input', function() {{
    const query = this.value.toLowerCase().trim();
    if (!query) {{
        nodes.forEach(node => {{ node.highlighted = false; }});
        nodes.update(nodes.get());
        return;
    }}
    nodes.forEach(node => {{
        node.highlighted = node.label.toLowerCase().includes(query);
        if (node.highlighted) {{
            node.borderWidth = 4;
            node.borderColor = '#f59e0b';
        }} else {{
            node.borderWidth = 2;
            node.borderColor = node.originalColor || node.borderColor;
        }}
    }});
    nodes.update(nodes.get());
}});

// 节点点击
network.on('click', function(params) {{
    const infoEl = document.getElementById('nodeInfo');
    if (params.nodes.length === 1) {{
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);
        if (node) {{
            infoEl.innerHTML = `
                <div class="detail-card">
                    <div class="detail-title">__PNAME__</div>
                    <dl class="detail-meta">
                        <dt>类型</dt><dd>__PCAT__</dd>
                        <dt>教材</dt><dd>__PTEXT__</dd>
                        <dt>章节</dt><dd>__PCHAP__</dd>
                        <dt>页码</dt><dd>__PPAGE__</dd>
                    </dl>
                    <div class="detail-def">__PDEF__</div>
                    <div class="detail-stats">
                        <div class="stat-item">
                            <div class="stat-value">__PFREQ__</div>
                            <div class="stat-label">出现频次</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">__PSRC__</div>
                            <div class="stat-label">涉及教材</div>
                        </div>
                    </div>
                </div>
            `;
            infoEl.innerHTML = infoEl.innerHTML
                .replace('__PNAME__', node.name || '-')
                .replace('__PCAT__', node.category || '-')
                .replace('__PTEXT__', node.textbook || '-')
                .replace('__PCHAP__', node.chapter || '-')
                .replace('__PPAGE__', node.page || '-')
                .replace('__PDEF__', node.definition || '暂无定义')
                .replace('__PFREQ__', String(node.frequency || 1))
                .replace('__PSRC__', String(node.sourceCount || 1));
        }}
    }} else {{
        infoEl.innerHTML = '';
    }}
}});

// 初始化数据
nodes.clear();
edges.clear();

const nodeData = [{{id: "placeholder", label: "无数据"}}];
const edgeData = [];

try {{
    nodes.add({node_data});
    edges.add({edge_data});
}} catch(e) {{
    console.error('Error loading graph:', e);
}}

// 触发布局
network.once('stabilizationIterationsDone', function() {{
    network.fit({{ animation: true }});
}});
</script>
"""


def get_relation_stats(edges: list[dict[str, Any]]) -> dict[str, int]:
    """统计关系类型分布"""
    stats: dict[str, int] = defaultdict(int)
    for edge in edges:
        rel = edge.get("relation_type", "unknown")
        stats[RELATION_TYPE_LABELS.get(rel, rel)] += 1
    return dict(stats)
