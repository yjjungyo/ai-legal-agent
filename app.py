# -*- coding: utf-8 -*-
"""
Streamlit 메인 앱 (ChatGPT 유사 UI)
- 사이드바: 새 채팅, 채팅 검색, 이전 채팅, PDF 인덱싱
- 메인: 대화 히스토리, 질의 입력, 멀티에이전트 답변 출력
- 인덱싱: src/rag/indexing.index_pdfs_by_collection 사용 (컬렉션 분리)
- 멀티에이전트: src/agents/workflow.build_legal_graph 사용 (SQLite 메모리)
"""

import os
import time
import glob
import streamlit as st
import logging, traceback

from src.agents.constants import CATEGORIES
from src.utils.logging_utils import setup_logging
logger = setup_logging()

# --- RAG 인덱싱 / 저장경로 ---
from src.config import CHROMA_PERSIST, DATA_PATH
from src.rag.indexing import (
    list_collections,
    list_index_inventory,
    index_pdfs_by_collection,
)


# --- 대화 DB (제공된 모듈 가정) ---
from src.db import (
    init_db, new_chat, add_message, list_chats,
    get_chat_messages, search_messages, rename_chat
)

# --- LangGraph 멀티에이전트 그래프 ---
from src.agents.workflow import build_legal_graph
from src.utils.tracing import langfuse_config



# -----------------------------------------------------------------------------
# Streamlit 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 법률 정보 서비스", page_icon="⚖️", layout="wide")
st.title("⚖️ AI 법률 정보 서비스")
st.subheader(f"{CATEGORIES} 8개 분야 AI 법률 전문가 상시 대기 중 !!!")

# -----------------------------------------------------------------------------
# 초기화(디렉토리/DB/세션)
# -----------------------------------------------------------------------------
init_db()
os.makedirs(CHROMA_PERSIST, exist_ok=True)

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "proposed_title" not in st.session_state:
    st.session_state.proposed_title = ""

# 멀티턴 메모리용 LangGraph 인스턴스 (캐시)
@st.cache_resource
def _get_graph():
    """
    LangGraph 그래프를 1회 빌드하여 캐시합니다.
    """
    logger.debug("[APP] build_legal_graph() - MemorySaver 시작")
    app = build_legal_graph()
    logger.debug("[APP] build_legal_graph() - MemorySaver 완료")
    return app

GRAPH = _get_graph()


# -----------------------------------------------------------------------------
# 사이드바
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("메뉴")

    # 새 채팅
    if st.button("➕ 새 채팅", width='stretch'):
        logger.debug("[UI] 새 채팅 클릭")
        st.session_state.chat_id = new_chat("새 채팅")
        st.session_state.proposed_title = ""
        st.rerun()

    # 채팅 검색(FTS)
    st.text_input("채팅 검색(FTS)", key="search_q", placeholder="키워드 입력")
    if st.session_state.search_q:
        logger.debug("[UI] 채팅 검색: '%s'", st.session_state.search_q)
        st.write("검색 결과:")
        results = search_messages(st.session_state.search_q)
        for cid, snippet in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"열기 #{cid}", key=f"open_{cid}"):
                    st.session_state.chat_id = cid
                    st.rerun()
            with col2:
                st.caption(snippet)

    st.divider()

    # 이전 채팅 목록
    st.write("이전 채팅")
    for cid, title, created in list_chats():
        if st.button(f"#{cid} {title}", key=f"cid_{cid}", width='stretch'):
            logger.debug("[UI] 채팅 선택: id=%s", cid)
            st.session_state.chat_id = cid
            st.rerun()

    st.divider()



# ─────────────────────────────────────────────────────────────
# 안전 콜백/호출 래퍼
# ─────────────────────────────────────────────────────────────
def _safe_notify(cb, msg: str) -> None:
    """on_progress 콜백 안전 호출(예외 무시 + 로그)."""
    try:
        if cb:
            cb(msg)
    except Exception as e:
        logger.warning("on_progress callback error: %s", e)

def _run_with_status(label: str, func, *args, **kwargs):
    """
    Streamlit 최신(st.status)과 구버전(spinner) 모두 지원.
    func 실행을 try/except로 감싸고, 성공/실패 메시지/로그 출력.
    """
    # 진행 메시지 표시용
    progress_box = st.empty()

    def on_progress(msg: str):
        progress_box.write(msg)

    # kwargs에 on_progress 주입(이미 있으면 그대로 사용)
    if "on_progress" not in kwargs:
        kwargs["on_progress"] = on_progress

    # 최신 status 위젯 시도
    try:
        with st.status(label, expanded=True) as status:
            try:
                result = func(*args, **kwargs)
                status.update(label=f"✅ 완료: {label}", state="complete")
                return result
            except Exception as e:
                logger.exception("실행 실패: %s", e)
                status.update(label=f"💥 실패: {label}", state="error")
                # 화면에 자세한 예외 표시
                st.error(f"에러: {e}")
                st.code("".join(traceback.format_exc()))
                return None
    except Exception:
        # st.status 없는 버전 -> spinner로 폴백
        with st.spinner(label):
            try:
                result = func(*args, **kwargs)
                st.success(f"✅ 완료: {label}")
                return result
            except Exception as e:
                logger.exception("실행 실패(spinner): %s", e)
                st.error(f"에러: {e}")
                st.code("".join(traceback.format_exc()))
                return None

# ─────────────────────────────────────────────────────────────
# 패널 상태
# ─────────────────────────────────────────────────────────────
if "idx_panel_open" not in st.session_state:
    st.session_state.idx_panel_open = False

st.sidebar.markdown("### 📚 PDF 인덱싱")
if st.sidebar.button("인덱싱 / 상태 보기", width='stretch'):
    st.session_state.idx_panel_open = True
if st.sidebar.button("닫기", width='stretch'):
    st.session_state.idx_panel_open = False

# ─────────────────────────────────────────────────────────────
# 인덱싱 패널 (예외 처리 포함)
# ─────────────────────────────────────────────────────────────
if st.session_state.idx_panel_open:
    st.write("## 📚 현재 인덱스 상태")

    # 인벤토리 조회도 try/except로 보호
    try:
        inv = list_index_inventory()
        cols = list_collections()
    except Exception as e:
        logger.exception("인벤토리 조회 실패: %s", e)
        st.error(f"인벤토리 조회 실패: {e}")
        st.code("".join(traceback.format_exc()))
        inv, cols = [], []

    if not inv:
        st.info("인덱스가 없습니다. 아래 버튼으로 처음 인덱싱을 수행하세요.")
        if st.button("🧭 최초 인덱싱 실행", type="primary"):
            result = _run_with_status(
                "초기 인덱싱 중...",
                index_pdfs_by_collection,
                data_dir="./data",
                chunk_size=800,
                chunk_overlap=120,
                reindex=False,     # 최초
            )
            if isinstance(result, int):
                st.success(f"초기 인덱싱 완료: 총 {result} chunks")
    else:
        st.caption(f"컬렉션 {len(cols)}개 / 색인 파일 {len({r['file_name'] for r in inv})}개")
        st.dataframe(inv, width='stretch')

        st.divider()
        st.write("### ♻️ 재인덱싱")

        file_options = sorted({r["file_name"] for r in inv})
        with st.form("reindex_form"):
            target_files = st.multiselect(
                "재인덱싱할 파일(미선택 시 전체)",
                options=file_options,
                placeholder="파일을 선택하지 않으면 전체 재인덱싱합니다.",
            )
            reidx = st.form_submit_button("재인덱싱 실행", type="primary")

        if reidx:
            result = _run_with_status(
                "재인덱싱(중복 삭제 후 다시 색인) 진행 중...",
                index_pdfs_by_collection,
                data_dir="./data",
                chunk_size=800,
                chunk_overlap=120,
                reindex=True,                   # 기존 청크 삭제 후 재색인
                target_files=target_files or None,
            )
            if isinstance(result, int):
                st.success(f"재인덱싱 완료: 총 {result} chunks")

# -----------------------------------------------------------------------------
# 메인: 대화 렌더링 & 입력
# -----------------------------------------------------------------------------
if not st.session_state.chat_id:
    st.info("좌측에서 **새 채팅**을 눌러 시작하거나, 기존 채팅을 선택하세요.")
else:
    # 기존 메시지 렌더링 (DB 저장 기준)
    for role, content in get_chat_messages(st.session_state.chat_id):
        with st.chat_message(role):
            st.markdown(content)

    # 입력창
    if q := st.chat_input("질문을 입력하세요 (예: 전세보증금 반환 절차는?)"):
        logger.debug("[UI] 사용자 질문 입력(len=%d)", len(q))

        # 1) DB에 사용자 메시지 저장 & 화면 표시
        add_message(st.session_state.chat_id, "user", q)
        with st.chat_message("user"):
            st.markdown(q)

        # 2) LangGraph 멀티에이전트 실행
        with st.chat_message("assistant"):
            with st.spinner("분석/검색 중…"):
                try:
                    # thread_id를 chat_id와 매핑(멀티턴 메모리 일관성)
                    thread_id = f"chat-{st.session_state.chat_id}"

                    # LangGraph MessagesState 포맷: [("user", q)]
                    inputs = {"messages": [("user", q)]}

                    logger.debug("[APP] graph.invoke 시작 (thread_id=%s)", thread_id)
                    with langfuse_config(thread_id) as config:
                        result = GRAPH.invoke(inputs, config=config)
                    logger.debug("[APP] graph.invoke 완료")

                    # 3) 결과 메시지 중 마지막 AI 응답 추출
                    ai_text = None
                    for m in reversed(result["messages"]):
                        if m.type == "ai":
                            ai_text = m.content
                            break
                    ai_text = ai_text or "(응답 생성 실패)"

                    # 4) 화면 출력 & DB 저장
                    st.markdown(ai_text)
                    add_message(st.session_state.chat_id, "assistant", ai_text)
                    logger.debug("[APP] 응답 저장(len=%d)", len(ai_text))

                except Exception as e:
                    logger.exception("[APP] 멀티에이전트 실행 실패")
                    st.error(f"오류가 발생했습니다: {e}")

        # 첫 메시지로 제목 자동 제안
        if not st.session_state.proposed_title:
            st.session_state.proposed_title = q[:20]
            rename_chat(st.session_state.chat_id, st.session_state.proposed_title)

# 하단 안내
st.caption("※ 본 서비스는 AI Bootcamp 최종 과제 제출용이며 법률자문이 아닙니다.")
