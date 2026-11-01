# AI Legal Helper - System Inventory (기존 시스템 인벤토리)

## 1. 개요 및 실행 환경
기존 AI 법률자문서비스의 전반적인 시스템 구성 요소와 실행 환경을 분석한 결과입니다.

- **Python 버전 및 의존성**: `requirements.txt`에 명시된 주요 패키지를 사용하며, `langchain(0.3.27)`, `langgraph(0.6.7)`, `streamlit(1.49.1)`, `chromadb(1.0.20)`, `openai(1.107.1)` 등이 포함되어 있습니다.
- **실행 명령**: `streamlit run app.py`를 통해 웹 서버를 구동합니다.
- **환경 변수**: `.env` 파일과 `src/config.py`를 통해 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_CHAT`, `CHROMA_PERSIST`, `LOG_PATH` 등을 관리합니다.

## 2. 진입점 및 화면 구성
진입점인 `app.py`는 Streamlit을 사용하여 사용자 인터페이스를 제공합니다.
- **사이드바**: 새 채팅, 채팅 검색, 이전 채팅 목록 조회, RAG 인덱싱 관리 패널 (`src/rag/indexing.py`의 함수 호출).
- **메인 영역**: 대화 히스토리 출력, 채팅 입력창, LangGraph(`build_legal_graph`) 인보크 및 결과 렌더링.

## 3. 테스트 구성
- 별도의 단위/통합 테스트 폴더(`tests/`)나 pytest 설정은 확인되지 않았습니다.
- 루트에 `retrieve_test.py` 스크립트가 있어 부분적인 검색 테스트 용도로 사용된 것으로 보입니다.

## 4. 핵심 모듈
- `app.py`: UI 진입점
- `src/config.py`: 환경변수 및 모델 팩토리
- `src/db.py`: 대화 기록 저장(SQLite)
- `src/agents/`: LangGraph 및 Agent 구현체 
- `src/rag/`: Chroma 색인 및 검색
- `src/utils/logging_utils.py`: 전역 로깅

