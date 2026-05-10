"""
EduMerge Agent - 学科知识整合智能体
面向"AI 全栈极速黑客松"赛题

功能：
- 多格式教材加载与解析
- 单本教材知识图谱构建与可视化
- 跨教材重叠、互补与缺失识别
- 30% 精华版整合策略
- 教师多轮反馈迭代
- RAG 精准问答（BM25 + Embedding 混合检索）
- 自动生成教学大纲
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ai_analyzer import (
    answer_question_with_ai,
    build_ai_integration_plan,
    build_ai_knowledge_graph,
    get_model_name,
    get_model_options,
    is_ai_configured,
    probe_ai_status,
    set_runtime_model,
)
from src.aligner import build_integration_plan, summarize_integration_metrics
from src.api_usage import load_api_events, summarize_api_usage
from src.essence import build_essence_outline, essence_to_markdown
from src.kg_builder import build_knowledge_graph, graph_to_html, get_relation_stats
from src.parser import parse_uploaded_file, save_uploaded_file
from src.rag import answer_question_locally, hybrid_search
from src.report_writer import write_integration_report
from src.syllabus import build_ai_syllabus, syllabus_to_markdown
from src.utils import format_file_size


st.set_page_config(
    page_title="EduMerge Agent",
    page_icon="📚",
    layout="wide",
)


def init_state() -> None:
    """初始化 session state"""
    defaults = {
        "textbooks": [],
        "parse_logs": [],
        "feedback": [],
        "integration_plan": [],
        "parsed_file_keys": [],
        "essence_outline": None,
        "syllabus": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_book_options() -> list[str]:
    """获取教材选项列表"""
    return [f"{book['textbook_id']} | {book['filename']}" for book in st.session_state.textbooks]


def selected_book_from_label(label: str) -> dict:
    """从标签中获取选中的教材"""
    selected_id = label.split(" | ")[0]
    return next(book for book in st.session_state.textbooks if book["textbook_id"] == selected_id)


# 初始化
init_state()

st.title("📚 EduMerge Agent：学科知识整合智能体")
st.caption("多源教材加载 → 单书知识图谱 → 跨教材重叠识别 → 30% 精华整合 → 教师反馈迭代")

# 侧边栏状态
total_chars = sum(book.get("total_chars", 0) for book in st.session_state.textbooks)
graph_snapshot = (
    build_knowledge_graph(st.session_state.textbooks)
    if st.session_state.textbooks
    else {"nodes": [], "edges": []}
)
api_events = load_api_events()
api_summary = summarize_api_usage(api_events)

with st.sidebar:
    st.header("项目状态")
    st.metric("已上传教材", len(st.session_state.textbooks))
    st.metric("累计字符数", f"{total_chars:,}")
    st.metric("图谱节点", len(graph_snapshot["nodes"]))
    st.metric("图谱关系", len(graph_snapshot["edges"]))
    st.divider()
    st.markdown("**当前覆盖**：PDF / Markdown / TXT / Word / Excel")
    st.markdown("**整合目标**：压缩到原始体量 30% 以内")
    st.divider()
    ai_ready = is_ai_configured()
    st.metric("AI 分析", "已启用" if ai_ready else "未配置")
    st.metric("API 状态", api_summary["health"])
    model_options = get_model_options()
    current_model = st.session_state.get("selected_llm_model", get_model_name())
    if current_model not in model_options:
        current_model = model_options[0]
    selected_model = st.selectbox(
        "AI 模型",
        options=model_options,
        index=model_options.index(current_model) if current_model in model_options else 0,
        disabled=not ai_ready,
    )
    st.session_state.selected_llm_model = selected_model
    set_runtime_model(selected_model)
    if not ai_ready:
        st.caption("在 .env 中填入 API Key 后启用 AI 分析。")
    st.caption("已启用短上下文、超时和本地缓存。")

# 主 Tab
tab_upload, tab_kg, tab_merge, tab_feedback, tab_rag, tab_syllabus, tab_docs, tab_api = st.tabs([
    "① 多源加载",
    "② 单书图谱",
    "③ 跨教材整合",
    "④ 教师反馈",
    "⑤ RAG问答",
    "⑥ 教学大纲",
    "⑦ 报告交付",
    "⑧ API仪表板",
])


# ============================================================
# Tab 1: 多源加载
# ============================================================
with tab_upload:
    st.subheader("① 多格式教材加载与章节解析")

    uploaded_files = st.file_uploader(
        "上传教材或资料，支持 PDF / Markdown / TXT / Word(.docx) / Excel(.xlsx, .xls)，可多选",
        type=["pdf", "md", "markdown", "txt", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        parse_clicked = st.button("开始解析上传文件", type="primary", use_container_width=True)
    with col_b:
        clear_clicked = st.button("清空当前会话数据", use_container_width=True)

    if clear_clicked:
        st.session_state.textbooks = []
        st.session_state.parse_logs = []
        st.session_state.feedback = []
        st.session_state.integration_plan = []
        st.session_state.parsed_file_keys = []
        st.session_state.essence_outline = None
        st.session_state.syllabus = None
        st.success("已清空当前会话数据。")

    if parse_clicked:
        if not uploaded_files:
            st.warning("请先上传至少一个文件。")
        else:
            overall_progress = st.progress(0, text="准备解析")
            file_status = st.empty()
            parsed_keys = set(st.session_state.parsed_file_keys)

            for file_idx, uploaded_file in enumerate(uploaded_files):
                file_key = f"{uploaded_file.name}:{uploaded_file.size}"
                if file_key in parsed_keys:
                    st.session_state.parse_logs.append({
                        "filename": uploaded_file.name,
                        "format": uploaded_file.name.split(".")[-1].lower(),
                        "size": format_file_size(uploaded_file.size),
                        "status": "已跳过",
                        "message": "同名同大小文件已解析，避免重复耗时",
                    })
                    overall_progress.progress((file_idx + 1) / len(uploaded_files), text="跳过重复文件")
                    continue

                current_progress = st.progress(0, text=f"准备解析 {uploaded_file.name}")

                def update_parse_progress(value: float, message: str, idx: int = file_idx) -> None:
                    current_progress.progress(value, text=f"{uploaded_file.name}：{message}")
                    overall_value = (idx + value) / len(uploaded_files)
                    overall_progress.progress(overall_value, text=f"整体进度：{idx + 1}/{len(uploaded_files)}")

                try:
                    file_status.info(f"正在保存并解析：{uploaded_file.name}")
                    update_parse_progress(0.02, "保存上传文件")
                    file_path = save_uploaded_file(uploaded_file)
                    textbook = parse_uploaded_file(
                        file_path=file_path,
                        textbook_id=f"book_{len(st.session_state.textbooks) + 1:02d}",
                        progress_callback=update_parse_progress,
                    )
                    st.session_state.textbooks.append(textbook)
                    parsed_keys.add(file_key)
                    st.session_state.parse_logs.append({
                        "filename": uploaded_file.name,
                        "format": uploaded_file.name.split(".")[-1].lower(),
                        "size": format_file_size(uploaded_file.size),
                        "status": "已完成",
                        "message": f"识别章节 {len(textbook.get('chapters', []))} 个",
                    })
                except Exception as exc:  # noqa: BLE001
                    st.session_state.parse_logs.append({
                        "filename": uploaded_file.name,
                        "format": uploaded_file.name.split(".")[-1].lower(),
                        "size": format_file_size(uploaded_file.size),
                        "status": "失败",
                        "message": str(exc),
                    })
                finally:
                    current_progress.empty()

            st.session_state.parsed_file_keys = sorted(parsed_keys)
            overall_progress.progress(1.0, text="解析完成，生成本地整合建议")
            st.session_state.integration_plan = build_integration_plan(
                st.session_state.textbooks,
                st.session_state.feedback,
            )
            st.session_state.essence_outline = build_essence_outline(
                st.session_state.textbooks,
                st.session_state.integration_plan,
                st.session_state.feedback,
            ) if st.session_state.textbooks else None
            file_status.empty()
            st.success("解析流程已完成。")

    st.markdown("### 文件解析状态")
    if st.session_state.parse_logs:
        st.dataframe(pd.DataFrame(st.session_state.parse_logs), use_container_width=True, hide_index=True)
    else:
        st.info("暂无文件。上传教材后，这里会显示文件名、格式、大小和解析状态。")

    st.markdown("### 教材结构预览")
    if not st.session_state.textbooks:
        st.info("还没有解析完成的教材。")
    else:
        selected_title = st.selectbox("选择一本教材查看章节结构", options=get_book_options())
        book = selected_book_from_label(selected_title)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("教材标题", book.get("title", "-"))
        c2.metric("总页数/片段", book.get("total_pages", "-"))
        c3.metric("总字符数", f"{book.get('total_chars', 0):,}")
        c4.metric("章节数", len(book.get("chapters", [])))

        chapter_rows = [
            {
                "chapter_id": chapter.get("chapter_id"),
                "title": chapter.get("title"),
                "page_start": chapter.get("page_start"),
                "page_end": chapter.get("page_end"),
                "char_count": chapter.get("char_count"),
            }
            for chapter in book.get("chapters", [])
        ]
        st.dataframe(pd.DataFrame(chapter_rows), use_container_width=True, hide_index=True)

        with st.expander("查看统一结构 JSON 示例", expanded=False):
            preview = dict(book)
            preview["chapters"] = preview["chapters"][:2]
            st.json(preview)


# ============================================================
# Tab 2: 单书图谱
# ============================================================
with tab_kg:
    st.subheader("② 为每本教材构建知识图谱")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        selected_title = st.selectbox("选择教材", options=get_book_options(), key="kg_book_select")
        book = selected_book_from_label(selected_title)
        use_ai_graph = st.checkbox(
            "使用 AI 抽取知识图谱",
            value=is_ai_configured(),
            disabled=not is_ai_configured(),
        )
        if use_ai_graph:
            ai_progress = st.progress(0, text="AI 正在抽取结构化知识点")
            with st.spinner("AI 分析中，已启用短上下文和缓存。"):
                ai_progress.progress(0.4, text="生成图谱 JSON")
                graph = build_ai_knowledge_graph(book)
                ai_progress.progress(1.0, text="图谱生成完成")
            ai_progress.empty()
        else:
            graph = build_knowledge_graph([book])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("节点数", len(graph.get("nodes", [])))
        k2.metric("关系数", len(graph.get("edges", [])))
        k3.metric("教材来源", len(set(n.get("source_textbook") for n in graph.get("nodes", []))))

        # 关系类型分布
        relation_stats = get_relation_stats(graph.get("edges", []))
        if relation_stats:
            k4.metric("主要关系", list(relation_stats.keys())[0] if relation_stats else "-")

        components.html(graph_to_html(graph), height=650, scrolling=False)

        st.markdown("**图谱交互说明**：点击节点查看详情 | 滚轮缩放 | 拖拽移动 | 搜索高亮")
        with st.expander("查看图谱 JSON", expanded=False):
            st.json(graph)


# ============================================================
# Tab 3: 跨教材整合
# ============================================================
with tab_merge:
    st.subheader("③ 跨教材识别重叠、互补与缺失")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        use_ai_merge = st.checkbox(
            "使用 AI 分析重叠、互补与缺失",
            value=is_ai_configured(),
            disabled=not is_ai_configured(),
        )
        if st.button("重新生成整合建议", type="primary", use_container_width=True):
            analysis_progress = st.progress(0, text="准备生成整合建议")
            with st.spinner("正在分析跨教材重叠、互补与缺失。"):
                analysis_progress.progress(0.35, text="生成整合决策")
                st.session_state.integration_plan = (
                    build_ai_integration_plan(st.session_state.textbooks, st.session_state.feedback)
                    if use_ai_merge
                    else build_integration_plan(st.session_state.textbooks, st.session_state.feedback)
                )
                analysis_progress.progress(0.75, text="生成 30% 精华版草案")
                st.session_state.essence_outline = build_essence_outline(
                    st.session_state.textbooks,
                    st.session_state.integration_plan,
                    st.session_state.feedback,
                )
                analysis_progress.progress(1.0, text="分析完成")
            analysis_progress.empty()

        plan = st.session_state.integration_plan or build_integration_plan(
            st.session_state.textbooks,
            st.session_state.feedback,
        )
        st.session_state.integration_plan = plan

        # 整合指标
        metrics = summarize_integration_metrics(st.session_state.textbooks, plan)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("原始总字数", f"{metrics.get('total_chars', total_chars):,}")
        m2.metric("整合知识点", len(plan))
        m3.metric("语义对齐", metrics.get("aligned_pairs", 0))
        m4.metric("需复核", sum(1 for d in plan if d.get("needs_teacher_review")))

        if plan:
            df = pd.DataFrame(plan)
            display_cols = [
                col
                for col in ["term", "action", "confidence", "source_count", "is_aligned", "needs_teacher_review", "reason"]
                if col in df.columns
            ]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            st.markdown("### 决策分布")
            st.bar_chart(df["action"].value_counts())
        else:
            st.info("当前文本太少，尚未生成有效整合建议。")

        st.markdown("### 30% 精华版草案")
        if st.button("生成/刷新精华版草案", use_container_width=True):
            essence_progress = st.progress(0, text="筛选核心章节")
            st.session_state.essence_outline = build_essence_outline(
                st.session_state.textbooks,
                st.session_state.integration_plan,
                st.session_state.feedback,
            )
            essence_progress.progress(1.0, text="精华版草案已生成")
            essence_progress.empty()

        outline = st.session_state.essence_outline
        if outline:
            e1, e2, e3 = st.columns(3)
            e1.metric("目标字符数", f"{outline.get('target_chars', 0):,}")
            e2.metric("精华版字符数", f"{outline.get('selected_chars', 0):,}")
            e3.metric("实际压缩比", f"{outline.get('compression_ratio', 0):.1%}")
            with st.expander("查看精华版内容草案", expanded=False):
                st.markdown(essence_to_markdown(outline))
        else:
            st.info("解析教材后可生成精华版草案。")


# ============================================================
# Tab 4: 教师反馈
# ============================================================
with tab_feedback:
    st.subheader("④ 教师多轮反馈与整合方案迭代")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        role = st.selectbox("反馈来源", ["学科教师", "教研组", "课程负责人", "评审专家"])
        feedback_text = st.text_area(
            "输入反馈意见",
            placeholder="例如：某个核心概念不能压缩；需要保留实验案例；章节顺序要先基础后应用。",
            height=140,
        )
        if st.button("记录反馈并更新方案", type="primary", use_container_width=True):
            if not feedback_text.strip():
                st.warning("请输入反馈内容。")
            else:
                st.session_state.feedback.append({"role": role, "content": feedback_text.strip()})
                st.session_state.integration_plan = build_integration_plan(
                    st.session_state.textbooks,
                    st.session_state.feedback,
                )
                st.session_state.essence_outline = build_essence_outline(
                    st.session_state.textbooks,
                    st.session_state.integration_plan,
                    st.session_state.feedback,
                )
                st.session_state.syllabus = build_ai_syllabus(
                    st.session_state.textbooks,
                    st.session_state.integration_plan,
                    st.session_state.feedback,
                )
                st.success("反馈已记录。整合方案和大纲会自动更新。")

        if st.session_state.feedback:
            st.markdown("### 已记录反馈")
            st.dataframe(pd.DataFrame(st.session_state.feedback), use_container_width=True, hide_index=True)
        else:
            st.info("暂无反馈。")

        st.markdown("### 反馈驱动的调整原则")
        st.write(
            "系统会把教师反馈写入报告，并将包含「保留、不能压缩、补充、前置、案例、实验」等词的意见作为人工复核重点。"
        )


# ============================================================
# Tab 5: RAG 问答
# ============================================================
with tab_rag:
    st.subheader("⑤ 本地教材问答（BM25 + Embedding 混合检索）")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        question = st.text_input("请输入问题", placeholder="例如：什么是二叉树？有哪些排序算法？")
        col_use_ai, col_use_embed = st.columns([1, 1])
        with col_use_ai:
            use_ai_rag = st.checkbox(
                "使用 AI 生成答案",
                value=is_ai_configured(),
                disabled=not is_ai_configured(),
            )
        with col_use_embed:
            use_embedding = st.checkbox(
                "使用 Embedding 语义检索",
                value=True,
                help="关闭则仅使用 BM25 关键词检索",
            )

        if st.button("提交问题", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("请输入问题。")
            else:
                qa_progress = st.progress(0, text="检索教材片段")
                with st.spinner("正在生成回答。"):
                    qa_progress.progress(0.45, text="混合检索中（BM25 + Embedding）")
                    if use_ai_rag and is_ai_configured():
                        answer = answer_question_with_ai(st.session_state.textbooks, question)
                    else:
                        answer = answer_question_locally(
                            st.session_state.textbooks,
                            question,
                            use_embedding=use_embedding,
                        )
                    qa_progress.progress(1.0, text="问答完成")
                qa_progress.empty()
                st.markdown("### 回答")
                st.markdown(answer["answer"])
                if answer.get("citations"):
                    st.markdown("### 引用片段")
                    for idx, citation in enumerate(answer["citations"], start=1):
                        score_info = f"融合分数: {citation.get('fusion_score', 'N/A')}" if use_embedding else ""
                        with st.expander(f"引用 {idx}：{citation.get('textbook', '-')} / {citation.get('chapter', '-')} {score_info}"):
                            st.caption(f"页码/片段：{citation.get('page', '-')}")
                            st.write(citation.get("text", ""))


# ============================================================
# Tab 6: 教学大纲（创新功能）
# ============================================================
with tab_syllabus:
    st.subheader("⑥ 自动生成教学大纲（创新功能）")
    st.caption("基于整合方案自动生成结构化教学大纲，包含学习目标、章节安排和学时分配")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        use_ai_syllabus = st.checkbox(
            "使用 AI 优化大纲结构",
            value=is_ai_configured(),
            disabled=not is_ai_configured(),
            help="AI 可以生成更合理的章节组织和学时分配",
        )

        if st.button("生成教学大纲", type="primary", use_container_width=True):
            syllabus_progress = st.progress(0, text="正在生成教学大纲...")
            with st.spinner("分析整合方案，生成大纲结构。"):
                syllabus_progress.progress(0.5, text="整合知识点")
                if use_ai_syllabus and is_ai_configured():
                    st.session_state.syllabus = build_ai_syllabus(
                        st.session_state.textbooks,
                        st.session_state.integration_plan,
                        st.session_state.feedback,
                    )
                else:
                    from src.syllabus import build_syllabus_outline
                    st.session_state.syllabus = build_syllabus_outline(
                        st.session_state.textbooks,
                        st.session_state.integration_plan,
                        st.session_state.feedback,
                    )
                syllabus_progress.progress(1.0, text="大纲生成完成")
            syllabus_progress.empty()

        syllabus = st.session_state.syllabus
        if syllabus:
            # 大纲指标
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("课程名称", syllabus.get("title", "-"))
            s2.metric("总学时", f"{syllabus.get('total_hours', 0)} 小时")
            s3.metric("知识点数", syllabus.get("total_points", 0))
            s4.metric("压缩比", f"{syllabus.get('compression_ratio', 0):.1%}")

            # 大纲内容
            st.markdown("### 学习目标")
            for obj in syllabus.get("objectives", []):
                st.write(f"- {obj}")

            st.markdown("### 章节结构")
            for i, chapter in enumerate(syllabus.get("chapters", []), 1):
                with st.expander(f"📖 {chapter.get('title', f'第{i}章')}（{chapter.get('hours', 2)}学时 | {chapter.get('difficulty', '基础')}）"):
                    terms = chapter.get("knowledge_terms", [])
                    if terms:
                        st.markdown("**核心知识点**：")
                        for term in terms:
                            st.write(f"- {term}")

            # 下载 Markdown
            md_content = syllabus_to_markdown(syllabus)
            st.download_button(
                "📥 下载大纲 Markdown",
                md_content,
                file_name="教学大纲.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("点击上方按钮生成教学大纲。")


# ============================================================
# Tab 7: 报告交付
# ============================================================
with tab_docs:
    st.subheader("⑦ 整合报告与交付文档")

    if not st.session_state.textbooks:
        st.warning("请先上传并解析教材。")
    else:
        if st.button("生成整合报告", type="primary", use_container_width=True):
            report_path = write_integration_report(st.session_state.textbooks, st.session_state.feedback)
            st.success(f"报告已生成：{report_path}")

        report_path = "report/整合报告.md"
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                st.markdown(report_content)
        except FileNotFoundError:
            st.info("点击上方按钮生成整合报告。")

    st.markdown("### 已覆盖赛题要求")
    st.checkbox("多格式教材加载：PDF / Markdown / TXT / Word / Excel", value=True, disabled=True)
    st.checkbox("单本教材知识图谱构建与可视化", value=True, disabled=True)
    st.checkbox("跨教材重叠、互补与缺失识别（含语义对齐）", value=True, disabled=True)
    st.checkbox("整合压缩到 30% 以内的精华版策略", value=True, disabled=True)
    st.checkbox("教师多轮反馈与报告沉淀", value=True, disabled=True)
    st.checkbox("RAG 精准问答（BM25 + Embedding 混合检索）", value=True, disabled=True)
    st.checkbox("自动生成教学大纲（创新功能）", value=True, disabled=True)


# ============================================================
# Tab 8: API 仪表板
# ============================================================
with tab_api:
    st.subheader("⑧ API 额度与调用健康仪表板")
    st.caption(
        "通过调用日志、429、insufficient_quota、缓存命中和估算 token 判断当前 API 健康状态。"
    )

    latest_events = load_api_events()
    latest_summary = summarize_api_usage(latest_events)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("API 健康", latest_summary["health"])
    a2.metric("调用总数", latest_summary["total_calls"])
    a3.metric("成功 / 失败", f"{latest_summary['success_calls']} / {latest_summary['failure_calls']}")
    a4.metric("缓存命中", latest_summary["cache_hits"])

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("额度错误", latest_summary["quota_errors"])
    b2.metric("限流错误", latest_summary["rate_limits"])
    b3.metric("估算 Token", f"{latest_summary['estimated_tokens']:,}")
    b4.metric("实际 Token", f"{latest_summary['actual_tokens']:,}" if latest_summary["actual_tokens"] else "未返回")

    st.markdown("### 当前配置")
    config_rows = pd.DataFrame([
        {"项目": "AI 是否配置", "值": "是" if is_ai_configured() else "否"},
        {"项目": "提供方", "值": "ModelScope"},
        {"项目": "模型", "值": get_model_name()},
        {"项目": "可选模型", "值": "；".join(get_model_options())},
        {"项目": "状态判定", "值": latest_summary["health"]},
        {"项目": "最后调用时间", "值": latest_summary["last_ts"]},
        {"项目": "最后状态", "值": latest_summary["last_status"]},
        {"项目": "最后错误类型", "值": latest_summary["last_error_type"] or "-"},
        {"项目": "最后错误信息", "值": latest_summary["last_error_message"] or "-"},
    ])
    st.dataframe(config_rows, use_container_width=True, hide_index=True)

    col_probe, col_refresh = st.columns([1, 1])
    with col_probe:
        if st.button("检测 API 状态", type="primary", use_container_width=True, disabled=not is_ai_configured()):
            with st.spinner("正在发送最小测试请求，会消耗极少额度。"):
                result = probe_ai_status()
            if result.get("ok"):
                st.success(f"API 可用：{result.get('message', '')}")
            else:
                st.error(f"API 不可用：{result.get('error_type', 'unknown')} / {result.get('message', '')}")
    with col_refresh:
        if st.button("刷新仪表板", use_container_width=True):
            st.rerun()

    st.markdown("### 最近调用明细")
    if latest_events:
        event_frame = pd.DataFrame(latest_events[-80:])
        visible_cols = [
            "ts", "status", "operation", "model", "duration_ms",
            "estimated_input_tokens", "estimated_output_tokens",
            "actual_total_tokens", "error_type", "error_message",
        ]
        visible_cols = [col for col in visible_cols if col in event_frame.columns]
        st.dataframe(event_frame[visible_cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("暂无 API 调用记录。使用 AI 功能后会生成记录。")

    st.markdown("### 速度与额度优化建议")
    st.write(
        "优先依赖缓存命中；演示时可降低 AI_MAX_CHAPTERS、AI_CHAPTER_CHAR_LIMIT、AI_INTEGRATION_CHAR_LIMIT，"
        "减少输入 token；如果出现 quota_exhausted，需要更换 token 或等待额度恢复。"
    )
