# -*- coding: utf-8 -*-
"""
테스트 모듈: test_file_parser.py

[테스트 개요]
file_parser 모듈의 파일 형식별 텍스트 추출, 로컬 파일 저장,
LLM 입력용 포맷팅 및 채팅 마크다운 생성 기능을 단위 테스트한다.
"""

import io
import os
import shutil
import unittest
from pathlib import Path

import docx
import fitz

from src.utils.file_parser import (
    AttachmentData,
    extract_text_from_upload,
    format_chat_display_message,
    format_file_size,
    format_llm_input,
    save_attachment,
)


class FileParserTests(unittest.TestCase):
    """첨부 파일 파서 및 포맷터 단위 테스트 클래스."""

    def setUp(self):
        self.test_dir = Path("storage") / "attachments" / "chat_test_999"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_format_file_size(self):
        """파일 크기 바이트 단위 문자열 포맷팅 검증."""
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(1536), "1.5 KB")
        self.assertEqual(format_file_size(1024 * 1024 * 3), "3.00 MB")

    def test_extract_plain_text_utf8_and_cp949(self):
        """UTF-8 및 CP949 인코딩 텍스트 파싱 검증."""
        utf8_content = "제1조 (목적) 본 계약은 보증금 반환에 관한 사항을 정한다."
        extracted_utf8 = extract_text_from_upload(utf8_content.encode("utf-8"), "test.txt")
        self.assertIn("보증금 반환", extracted_utf8)

        cp949_content = "제2조 (임대차 기간) 2024년 1월 1일부터 2년간으로 한다."
        extracted_cp949 = extract_text_from_upload(cp949_content.encode("cp949"), "test_euckr.txt")
        self.assertIn("임대차 기간", extracted_cp949)

    def test_extract_pdf_text(self):
        """PyMuPDF(fitz)를 활용한 합성 PDF 텍스트 추출 검증."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Lease Agreement: Deposit Refund Clause and Rights")
        pdf_bytes = doc.tobytes()
        doc.close()

        extracted = extract_text_from_upload(pdf_bytes, "계약서.pdf")
        self.assertIn("--- [페이지 1] ---", extracted)
        self.assertIn("Lease Agreement", extracted)

    def test_extract_docx_text(self):
        """python-docx를 활용한 Word 파일 문단 및 표 추출 검증."""
        doc = docx.Document()
        doc.add_paragraph("제3조 (차임 지급 의무) 매월 25일에 지급한다.")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "구분"
        table.cell(0, 1).text = "금액"
        table.cell(1, 0).text = "보증금"
        table.cell(1, 1).text = "50,000,000원"

        buf = io.BytesIO()
        doc.save(buf)
        docx_bytes = buf.getvalue()

        extracted = extract_text_from_upload(docx_bytes, "계약서.docx")
        self.assertIn("차임 지급 의무", extracted)
        self.assertIn("보증금 | 50,000,000원", extracted)

    def test_save_attachment(self):
        """로컬 파일 저장 위치 및 내용 일치 검증."""
        sample_bytes = b"Sample File Content for Legal Agent"
        saved_path = save_attachment(sample_bytes, "sample.txt", chat_id="test_999")
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "rb") as f:
            self.assertEqual(f.read(), sample_bytes)

    def test_format_llm_input(self):
        """LLM 호출용 프롬프트에 문서 내용과 질문이 통합되는지 검증."""
        att = AttachmentData(
            filename="전세계약서.pdf",
            file_type=".pdf",
            file_size=1024 * 10,
            saved_path="path/to/file",
            extracted_text="특약사항: 임대인은 근저당권을 말소한다.",
            summary_snippet="특약사항 요약",
        )

        llm_input = format_llm_input("특약 불이행 시 계약 해지 가능한가요?", [att])
        self.assertIn("[사용자 첨부 문서 안내]", llm_input)
        self.assertIn("전세계약서.pdf", llm_input)
        self.assertIn("근저당권을 말소한다", llm_input)
        self.assertIn("특약 불이행 시 계약 해지 가능한가요?", llm_input)

    def test_format_chat_display_message(self):
        """채팅 화면용 마크다운에 배지 및 접이식 details 태그 생성 검증."""
        att = AttachmentData(
            filename="내용증명.txt",
            file_type=".txt",
            file_size=2048,
            saved_path="path/to/file",
            extracted_text="보증금 반환을 촉구하오니...",
            summary_snippet="보증금 반환 촉구",
        )

        display_msg = format_chat_display_message("내용증명 보냈는데 다음 절차는?", [att])
        self.assertIn("첨부 파일 (1개)", display_msg)
        self.assertIn("`내용증명.txt`", display_msg)
        self.assertIn("<details>", display_msg)
        self.assertIn("보증금 반환을 촉구하오니", display_msg)


if __name__ == "__main__":
    unittest.main()
