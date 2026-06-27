# EduMerge Agent - 学科知识整合智能体

面向「AI 全栈极速黑客松·学科知识整合智能体开发」赛题的完整项目实现。

## 赛题背景

在高校教学体系中，同一学科往往存在多本教材并行使用的现象。这些教材虽然各有侧重，但彼此之间存在大量内容重复。

**你的任务**：用 AI 帮教师把 7 本教材变成不到 2.1 本的精华，而且变完之后教学效果不打折。

## 核心功能

| 功能 | 说明 | 赛题要求 |
|------|------|----------|
| 多格式教材加载 | PDF / Markdown / TXT / Word / Excel | P0 必须 |
| 知识图谱构建 | 单本教材知识点抽取与可视化 | P0 必须 |
| 跨教材整合 | 识别重叠、互补与缺失 | P0 必须 |
| 30% 精华版 | 压缩到原始体量 30% 以内 | P0 必须 |
| RAG 精准问答 | 带引用的原文问答 | P0 必须 |
| 教师反馈迭代 | 多轮对话优化方案 | P1 建议 |
| 教学大纲生成 | 自动生成课程大纲 | **创新加分** |

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                     Agent 模块层                                  │
│  Parser │ KG Agent │ Alignment │ Compression │ RAG │ Syllabus    │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      AI 服务层                                    │
│  LLM (通义千问/DeepSeek) + Embedding (sentence-transformers)     │
└─────────────────────────────────────────────────────────────────┘
```

**设计决策**：
- 单体应用 + 模块化设计：降低 5 小时黑客松的部署风险
- BM25 + Embedding 混合检索：同时覆盖精确匹配和语义相似
- 本地 embedding：数据隐私安全

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro
LLM_MODEL_OPTIONS=deepseek-ai/DeepSeek-V4-Pro,MiniMax/MiniMax-M2.7:MiniMax
AI_SKIP_AFTER_PROVIDER_ERROR=true
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501

### 4. Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 项目结构

```
.
├── app.py                    # Streamlit 主应用
├── src/
│   ├── ai_analyzer.py        # AI 分析模块（LLM 调用、缓存）
│   ├── aligner.py            # 跨教材对齐（语义对齐 + 决策生成）
│   ├── api_usage.py          # API 调用日志与统计
│   ├── essence.py            # 30% 精华版生成
│   ├── kg_builder.py         # 知识图谱构建与可视化
│   ├── parser.py             # 多格式文件解析
│   ├── rag.py               # 混合检索 RAG
│   ├── report_writer.py     # 整合报告生成
│   ├── syllabus.py           # 教学大纲生成（创新功能）
│   └── utils.py             # 工具函数
├── docs/
│   ├── Agent架构说明.md      # 架构设计文档
│   ├── 需求分析.md           # 需求规格说明
│   ├── 赛题优化建议.md       # 优化建议报告
│   └── 系统设计.md           # 系统设计文档
├── report/                   # 生成的整合报告
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 赛题评分要点

### A. 文档 (15分)
- [x] Agent 架构说明（含设计决策论证）
- [x] 需求分析
- [x] 整合报告
- [x] README

### B. 功能 (25分)
- [x] 多格式教材加载与解析
- [x] 单本教材知识图谱构建
- [x] 跨教材重叠、互补与缺失识别（含语义对齐）
- [x] 30% 精华版整合
- [x] RAG 精准问答
- [x] 教师多轮反馈迭代

### C. 可视化 (13分)
- [x] 节点点击查看详情
- [x] 频次可视化（节点大小）
- [x] 教材来源区分（颜色）
- [x] 搜索高亮

### D. 架构 (20分)
- [x] 模块化设计
- [x] RAG Pipeline（含混合检索说明）
- [x] Prompt 工程
- [x] 已知局限与改进方案

### E. 代码 (17分)
- [x] 前后端分离
- [x] requirements.txt
- [x] .env.example
- [x] Dockerfile / docker-compose

### F. 创新 (10分)
- [x] 自动教学大纲生成
- [x] 前置依赖链可视化

## RAG Pipeline 说明

### 混合检索策略

```
用户问题
    │
    ▼
┌─────────────┐     ┌─────────────┐
│   BM25      │     │  Embedding   │
│  关键词召回  │     │  语义召回    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│     RRF 融合 (k=60)             │
│  score = Σ 1/(60 + rank)       │
└─────────────┬───────────────────┘
              │
              ▼
        Top-K 引用返回
```

### 防幻觉策略

1. **来源约束**：Prompt 明确要求「只基于给定片段回答」
2. **不确定性表达**：如无法确定，明确回答「根据提供内容无法确定」
3. **引用标记**：答案必须包含引用编号

## 部署建议

推荐使用魔搭创空间（免费 CPU，支持 Streamlit）：

1. 推送代码到 GitHub
2. 在魔搭创空间创建应用，选择 Streamlit 模板
3. 配置环境变量
4. 部署

## 注意事项

- 不要将教材 PDF 推送到 GitHub（已配置 .gitignore）
- API 调用会产生费用，注意额度控制
- 大文件建议分批上传处理

---

浙江大学未来学习中心·AI 生态 2026 年 5 月
