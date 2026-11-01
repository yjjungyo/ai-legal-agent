# -*- coding: utf-8 -*-
"""
모듈명: file_parser.py

[모듈 개요]
Streamlit 채팅 창에서 사용자가 업로드한 첨부 파일(PDF, Word, 텍스트, 이미지 등)을
분석하여 텍스트를 추출하고, 로컬 디렉터리에 안전하게 저장하며, LLM 질의 및 멀티에이전트
호출 시 참조할 수 있도록 프롬프트 형태로 포맷팅하는 유틸리티 모듈이다.

[주요 기능]
1. PDF(.pdf): PyMuPDF(fitz) 기반 고속 페이지별 텍스트 추출
2. Word(.docx): python-docx 기반 문단 및 표(Table) 텍스트 추출
3. 일반 텍스트(.txt, .md, .csv, .json 등): 다중 인코딩(UTF-8, CP949, EUC-KR) 자동 감지 및 디코딩
4. 이미지(.png, .jpg, .jpeg, .webp): Azure OpenAI Vision을 활용한 법률 서류/계약서 OCR 전사
5. 첨부 파일 로컬 저장 (storage/attachments/{chat_id}/)
6. Streamlit UI 렌더링용 마스다운(접이식 details) 및 LLM 전송용 컨텍스트 빌더 제공
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import setup_logging

logger = setup_logging()

# 허용되는 첨부 파일 확장자 목록
SUPPORTED_EXTENSIONS = [
    "pdf", "txt", "docx", "md", "csv", "json", "png", "jpg", "jpeg", "webp"
]

# 단일 파일당 LLM 컨텍스트 전달 최대 글자 수 (토큰 오버플로 방지)
MAX_FILE_CHARS = 15000


@dataclass
class AttachmentData:
    """첨부 파일 메타데이터 및 추출 텍스트를 담는 데이터 클래스."""
    filename: str
    file_type: str
    file_size: int
    saved_path: str
    extracted_text: str
    summary_snippet: str


def format_file_size(size_bytes: int) -> str:
    """바이트 단위 크기를 사람이 읽기 쉬운 문자열(KB, MB)로 변환한다.

    Args:
        size_bytes: 파일 바이트 수

    Returns:
        str: 포맷된 파일 크기 (예: '124.5 KB')
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def _extract_pdf_text(file_bytes: bytes, max_chars: int = MAX_FILE_CHARS) -> str:
    """PyMuPDF(fitz)를 사용하여 PDF 파일에서 페이지별 텍스트를 추출한다.

    Args:
        file_bytes: PDF 파일의 바이너리 데이터
        max_chars: 최대 추출 글자 수 제한

    Returns:
        str: 페이지 번호와 함께 포맷팅된 추출 텍스트
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: List[str] = []
        total_len = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text() or ""
            text = text.strip()
            if text:
                header = f"--- [페이지 {page_idx + 1}] ---"
                pages_text.append(f"{header}\n{text}")
                total_len += len(text)
                if total_len >= max_chars:
                    pages_text.append(f"\n... (문서가 길어 {page_idx + 1}페이지까지 추출됨) ...")
                    break

        doc.close()
        return "\n\n".join(pages_text).strip() if pages_text else "(PDF 텍스트가 비어 있거나 스캔 이미지 형태입니다)"
    except Exception as exc:
        logger.warning("[FileParser] PDF 추출 실패, pypdf로 재시도: %s", exc)
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text() or ""
                if t.strip():
                    pages_text.append(f"--- [페이지 {i + 1}] ---\n{t.strip()}")
            return "\n\n".join(pages_text).strip() if pages_text else "(PDF 텍스트 추출 실패)"
        except Exception as pypdf_exc:
            logger.error("[FileParser] PDF 2차 추출 실패: %s", pypdf_exc)
            return f"(PDF 파싱 오류: {exc})"


def _extract_docx_text(file_bytes: bytes, max_chars: int = MAX_FILE_CHARS) -> str:
    """python-docx를 사용하여 Word(.docx) 파일의 문단과 표 텍스트를 추출한다.

    Args:
        file_bytes: DOCX 파일의 바이너리 데이터
        max_chars: 최대 추출 글자 수 제한

    Returns:
        str: 문단 및 표를 포함한 추출 텍스트
    """
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        parts: List[str] = []
        cur_len = 0

        # 1. 일반 문단 추출
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)
                cur_len += len(text)
                if cur_len >= max_chars:
                    parts.append("\n... (문서 길이 초과로 이후 내용 생략) ...")
                    return "\n".join(parts)

        # 2. 표(Table) 데이터 추출
        for t_idx, table in enumerate(doc.tables):
            table_lines: List[str] = [f"\n[표 {t_idx + 1}]"]
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(row_cells):
                    table_lines.append(" | ".join(row_cells))
            parts.append("\n".join(table_lines))

        return "\n\n".join(parts).strip() if parts else "(Word 문서 본문이 비어 있습니다)"
    except Exception as exc:
        logger.error("[FileParser] DOCX 텍스트 추출 실패: %s", exc)
        return f"(Word 문서 파싱 오류: {exc})"


def _extract_plain_text(file_bytes: bytes, max_chars: int = MAX_FILE_CHARS) -> str:
    """다양한 인코딩(UTF-8, CP949, EUC-KR)을 안전하게 시도하여 일반 텍스트를 디코딩한다.

    Args:
        file_bytes: 텍스트 파일 바이너리 데이터
        max_chars: 최대 추출 글자 수 제한

    Returns:
        str: 디코딩된 텍스트 문자열
    """
    encodings = ["utf-8", "cp949", "euc-kr", "utf-16", "latin-1"]
    decoded = None
    for enc in encodings:
        try:
            decoded = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if decoded is None:
        decoded = file_bytes.decode("utf-8", errors="replace")

    decoded = decoded.strip()
    if len(decoded) > max_chars:
        decoded = decoded[:max_chars] + f"\n... (전체 {len(decoded)}자 중 {max_chars}자까지만 추출) ..."
    return decoded or "(빈 파일입니다)"


def _ocr_image_with_vision(file_bytes: bytes, ext: str) -> str:
    """Azure OpenAI Vision 기능을 호출하여 이미지 문서의 텍스트를 전사(OCR)한다.

    Args:
        file_bytes: 이미지 파일 바이너리
        ext: 파일 확장자 (예: '.png', '.jpg')

    Returns:
        str: 이미지 전사 결과 텍스트
    """
    try:
        from src.config import get_llm
        from langchain_core.messages import HumanMessage

        # MIME 타입 결정
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext.lower(), "image/jpeg")
        b64_data = base64.b64encode(file_bytes).decode("utf-8")

        llm = get_llm()
        ocr_prompt = (
            "당신은 법률 문서 전사(OCR) 및 분석 전문가입니다.\n"
            "첨부된 이미지(계약서, 영수증, 등기부등본, 내용증명, 판결문 등)에 포함된 텍스트를 "
            "원문 그대로 정확히 읽고, 누락 없이 한국어로 전사해 주세요. "
            "서식이나 표 구조가 있다면 읽기 쉽게 구조화하여 작성해 주세요."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
            ]
        )
        logger.debug("[FileParser] Vision OCR 호출 시작 (크기: %d bytes)", len(file_bytes))
        resp = llm.invoke([message])
        text = (getattr(resp, "content", str(resp)) or "").strip()
        logger.debug("[FileParser] Vision OCR 완료 (추출 길이: %d)", len(text))
        return text if text else "(이미지에서 텍스트를 인식하지 못했습니다)"
    except Exception as exc:
        logger.warning("[FileParser] Vision OCR 실패(폴백): %s", exc)
        return f"[이미지 첨부 파일] (이미지 텍스트 자동 추출 실패: {exc})"


def extract_text_from_upload(
    file_bytes: bytes,
    filename: str,
    max_chars: int = MAX_FILE_CHARS,
) -> str:
    """업로드된 파일의 확장자에 맞추어 최적의 파서를 실행하고 텍스트를 추출한다.

    Args:
        file_bytes: 파일 바이너리 데이터
        filename: 파일 이름 (확장자 포함)
        max_chars: 최대 텍스트 추출 글자 수

    Returns:
        str: 정제된 추출 텍스트
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _extract_pdf_text(file_bytes, max_chars=max_chars)
    elif ext in (".docx", ".doc"):
        return _extract_docx_text(file_bytes, max_chars=max_chars)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        return _ocr_image_with_vision(file_bytes, ext)
    elif ext in (".txt", ".md", ".csv", ".json", ".hwp", ".log", ".xml"):
        return _extract_plain_text(file_bytes, max_chars=max_chars)
    else:
        # 기타 형식 시도
        return _extract_plain_text(file_bytes, max_chars=max_chars)


def save_attachment(
    file_bytes: bytes,
    filename: str,
    chat_id: Any,
) -> str:
    """업로드된 첨부 파일을 로컬 디스크에 안전하게 저장한다.

    저장 위치: storage/attachments/chat_{chat_id}/{timestamp}_{filename}

    Args:
        file_bytes: 파일 바이너리
        filename: 원본 파일명
        chat_id: 대화 세션 ID

    Returns:
        str: 저장된 로컬 파일 경로
    """
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", filename)
    timestamp = int(time.time())
    dest_dir = Path("storage") / "attachments" / f"chat_{chat_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / f"{timestamp}_{safe_name}"
    with open(dest_file, "wb") as f:
        f.write(file_bytes)

    logger.debug("[FileParser] 파일 저장 완료: %s", dest_file)
    return str(dest_file)


def process_uploaded_file(
    file_obj: Any,
    chat_id: Any,
) -> AttachmentData:
    """Streamlit UploadedFile 객체를 읽어 저장 및 텍스트 추출을 일괄 처리한다.

    Args:
        file_obj: Streamlit UploadedFile 객체
        chat_id: 현재 채팅 세션 ID

    Returns:
        AttachmentData: 추출된 텍스트 및 메타데이터 객체
    """
    filename = getattr(file_obj, "name", "unknown_file")
    file_bytes = file_obj.getvalue()
    file_size = len(file_bytes)
    ext = os.path.splitext(filename)[1].lower()

    # 1. 로컬 저장
    saved_path = save_attachment(file_bytes, filename, chat_id)

    # 2. 텍스트 추출
    extracted = extract_text_from_upload(file_bytes, filename)

    # 3. 요약 스니펫 생성 (최대 120자)
    clean_snippet = extracted.replace("\n", " ").strip()
    snippet = clean_snippet[:120] + ("..." if len(clean_snippet) > 120 else "")

    return AttachmentData(
        filename=filename,
        file_type=ext,
        file_size=file_size,
        saved_path=saved_path,
        extracted_text=extracted,
        summary_snippet=snippet,
    )


def format_llm_input(
    user_query: str,
    attachments: List[AttachmentData],
) -> str:
    """사용자 질의와 첨부 문서 내용을 멀티에이전트 LLM 입력용 통합 프롬프트로 구성한다.

    Args:
        user_query: 사용자 질문 텍스트
        attachments: 추출된 첨부 문서 리스트

    Returns:
        str: LLM 및 LangGraph 호출용 통합 텍스트
    """
    if not attachments:
        return user_query.strip()

    parts: List[str] = [
        "[사용자 첨부 문서 안내]",
        f"사용자가 법률 상담을 위해 총 {len(attachments)}개의 관련 문서를 첨부하였습니다.\n",
    ]

    for idx, att in enumerate(attachments, 1):
        parts.append(
            f"=== [첨부 문서 {idx}: {att.filename} ({format_file_size(att.file_size)}) 시작] ===\n"
            f"{att.extracted_text}\n"
            f"=== [첨부 문서 {idx}: {att.filename} 끝] ===\n"
        )

    parts.append(
        "[사용자 질문 및 상담 요청]\n"
        f"{user_query.strip() if user_query.strip() else '첨부된 문서를 종합 검토하여 주요 법률적 쟁점, 권리관계 및 유의사항을 분석해 주세요.'}\n\n"
        "[지침]\n"
        "- 위 첨부 문서의 구체적인 조항, 사실관계, 수치, 당사자 관계를 바탕으로 법률적 판단 및 솔루션을 제공하세요."
    )

    return "\n".join(parts)


def format_chat_display_message(
    user_query: str,
    attachments: List[AttachmentData],
) -> str:
    """Streamlit 채팅 화면 및 SQLite 대화 히스토리에 저장할 마크다운 메시지를 생성한다.

    HTML <details> 태그를 사용하여 첨부 파일의 원문 텍스트를 접이식으로 깔끔하게 제공한다.

    Args:
        user_query: 사용자 질문 텍스트
        attachments: 첨부 파일 목록

    Returns:
        str: 화면 출력 및 DB 저장용 마크다운
    """
    if not attachments:
        return user_query.strip()

    file_badges: List[str] = []
    details_sections: List[str] = []

    for att in attachments:
        file_badges.append(f"- 📄 **`{att.filename}`** ({format_file_size(att.file_size)})")
        details_sections.append(
            f"**[파일: {att.filename}]**\n```text\n{att.extracted_text[:2000]}\n"
            f"{'...(생략)...' if len(att.extracted_text) > 2000 else ''}\n```"
        )

    badges_md = "\n".join(file_badges)
    details_md = "\n\n".join(details_sections)

    display_parts = [
        f"📎 **첨부 파일 ({len(attachments)}개)**:\n{badges_md}\n",
        user_query.strip() if user_query.strip() else "*(첨부 문서 종합 분석 요청)*",
        "\n<details>\n<summary>🔍 첨부 파일 추출 내용 미리보기 (클릭)</summary>\n\n"
        f"{details_md}\n\n</details>",
    ]

    return "\n\n".join(display_parts)

