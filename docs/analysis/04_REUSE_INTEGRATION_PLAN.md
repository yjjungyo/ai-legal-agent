# AI Legal Helper - Reuse & Integration Plan

## 1. 개요
기존 법률자문서비스의 모듈별 재사용 전략입니다. 새로운 메인 아키텍처 통합 및 P0 요구사항(AOP, 토큰 추적, Temporal/Agentic Verification 확장)을 달성하기 위한 방안을 제시합니다.

## 2. 모듈별 분류 및 통합 계획

### [1] 그대로 재사용
- **모델 팩토리**: `src/config.py`의 `get_temp_llm`, `get_embeddings` (단, 토큰 수집 인터페이스 래핑이 필요할 수 있으나 설정 자체는 유지)
- **프롬프트 템플릿**: `src/prompts.py` 내의 기본 지시문

### [2] Adapter를 통해 재사용
- **Chroma 및 Retrieval 모듈 (`src/rag/`)**:
  - 현재 `Chroma` 래퍼 및 `similarity_search` 로직은 잘 구성되어 있으나, 향후 GraphRAG 연동 및 근거 재검색(Self-Reflective) 루프를 추가하려면 기존 `retrieval.py` 위쪽에 Adapter Layer를 두어 검색 방식을 다형성 있게 호출하도록 통합해야 합니다.

### [3] 리팩터링 필요
- **LangGraph 워크플로우 (`src/agents/workflow.py`)**:
  - 현재 순차적 파이프라인 형태이며 `MemorySaver`에 강하게 결합되어 서브그래프로 바로 편입하기 힘듭니다.
  - 신규 시스템의 메인 Router/Supervisor에 편입될 수 있도록, `LegalState` 정의를 표준화하고, Node들을 재조립해야 합니다.
- **Agent 구조 (`src/agents/base_agent.py` 등)**:
  - 카테고리별 에이전트 정적 매핑(`_GET_AGENT`)을 동적 레지스트리로 변경해야 합니다.
  - Agentic Verification 단계를 삽입하기 위한 Node/Edge 리팩터링이 필수적입니다.
- **AOP 및 로깅 (`src/utils/logging_utils.py`)**:
  - P0 요구사항 만족을 위해 `@trace` 데코레이터에서 `ContextVar`를 활용해 Trace ID/Run ID를 전파하도록 수정해야 합니다.
  - 개인정보/프롬프트 덤프를 막기 위한 마스킹 필터 추가가 필요합니다.

### [4] 신규 개발 필요
- **LLM Token Tracking Middleware**:
  - LLM 호출별 비용을 모니터링하기 위해 LangChain의 `get_openai_callback` 혹은 `BaseCallbackHandler` 구현체를 새로 개발하여 전역적으로 주입해야 합니다.
- **Temporal Case Intelligence**:
  - 기존에는 시간적 맥락을 고려한 판례 분석이 없으므로, 해당 도메인을 처리하는 신규 노드 및 도구가 필요합니다.
- **지속성 Checkpointer DB 연동**:
  - `MemorySaver` 대신 실제 분산 환경이나 영구 저장을 지원하는 Postgres/SQLite 기반 Checkpointer 연동 계층 개발.

