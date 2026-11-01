# AI Legal Helper - LLM & Observability Analysis

## 1. LLM 인터페이스 및 추적

### 회사 제공 LLM 호출 인터페이스
- `src/config.py`에 정의된 `get_temp_llm(temperature)`과 `get_llm()`을 통해 `AzureChatOpenAI` 인스턴스를 일관되게 생성합니다.
- 모델: `aicore-gpt4o` 등 배포된 모델을 사용 중입니다.

### 직접 LLM 호출 코드 위치
다음 위치에서 LLM을 직접 호출(invoke)합니다.
- `src/rag/indexing.py`: `classify_law_category_with_llm` (314라인), `classify_filename_with_llm` (377라인)
- `src/rag/query_rewrite.py`: `rewrite_query` (80라인)
- `src/agents/router.py`: `route_categories` (38라인)
- `src/agents/synthesizer.py`: `synthesize` (134라인)
- `src/agents/base_agent.py`: `build_react_agent` (53라인) 내 `create_react_agent`

### 토큰 사용량 제공 여부
- **미확인(부재)**: 현재 코드베이스 어디에도 LLM 응답 객체에서 `usage_metadata`나 토큰 사용량을 추출하여 추적/로깅하는 코드가 없습니다. P0 필수 요건인 토큰 추적 로직 추가가 시급합니다.

## 2. AOP / Observability 현황

### 기존 Python Logging 방식
- `src/utils/logging_utils.py`에 공통 로깅이 구현되어 있습니다.
- `@trace` 데코레이터를 통해 함수 진입(`[ENTER]`), 종료(`[EXIT]`), 예외(`[RAISE]`) 시 실행 시간을 밀리초 단위로 남깁니다.

### 파일명/함수명/라인 번호 기록
- 로깅 포맷 템플릿: `%(asctime)s,%(msecs)03d %(levelname)s [ %(filename)-20.20s %(funcName)-20.20s :%(lineno)4d ] %(message)s` (27라인).
- 파일명, 함수명, 라인 번호가 정상적으로 출력되는 구조입니다.

### Trace ID 및 Agent 실행 추적
- **Trace/Run ID 전파**: `app.py`에서 `thread_id`를 LangGraph에 넘겨주긴 하지만, 이것이 `logging_utils.py`의 전역 로거나 함수 컨텍스트(AOP)로 전파되어 분산 추적(Distributed Tracing) 기능을 수행하지는 않습니다.
- **Agent 실행 추적**: `base_agent.py`에서 `AgentExecutor`의 `return_intermediate_steps=False`로 설정되어 상세 스텝 관찰이 비활성화되어 있습니다(92라인).

### 개인정보 및 프롬프트 로그 노출 위험
- **위험**: 높음.
- `src/agents/base_agent.py` 79라인(`logger.debug("prompt=%s", prompt)`) 및 여러 곳에서 프롬프트 전체와 사용자 입력값을 그대로 디버그 로그에 남기고 있습니다. 개인정보가 포함될 경우 마스킹 없이 로그 파일에 영구 저장될 위험이 존재합니다.

