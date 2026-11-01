# AI Legal Helper - LangGraph & RAG Analysis

## 1. LangGraph 워크플로우 분석

### 그래프 위치 및 State
- **그래프 생성 위치**: `src/agents/workflow.py` 의 `build_legal_graph` 함수 (211라인).
- **State 정의**: `LegalState(MessagesState)` (34라인). `question`, `categories`, `plan`, `results`를 포함합니다.

### Node 및 Edge 구성
1. **`route_node`**: `src/agents/router.py`를 호출하여 질문을 카테고리로 분류 (115라인).
2. **`plan_node`**: `src/agents/planner.py`를 호출하여 실행할 에이전트 식별 (129라인).
3. **`run_agents_node`**: `ThreadPoolExecutor`를 사용하여 카테고리별 에이전트(정적 매핑된 `get_agent`)를 병렬 실행 (178라인).
4. **`synthesize_node`**: `src/agents/synthesizer.py`를 호출해 LLM으로 에이전트 결과를 종합 (204라인).
- **흐름**: `START -> route -> plan -> run_agents -> synthesize -> END`

### 라우팅 및 Checkpointer
- `route_node`가 사실상 라우터 역할을 하나 Conditional Edge 방식이 아닌 순차적 파이프라인 형태입니다.
- **Checkpointer**: `MemorySaver()`를 사용하고 있어 메모리 상에서만 상태가 유지되며(224라인), 재시작 시 휘발됩니다. DB 연동 Checkpointer 부재.
- **기존 Graph 재사용 여부**: 현재 워크플로우는 특정 카테고리 분류-병렬실행 구조로 강하게 결합되어 있어, 전체 메인 아키텍처에 직접 서브그래프로 편입하기엔 무리가 있으며(리팩터링 요망) 핵심 컴포넌트 단위의 Adapter를 통한 재사용이 권장됩니다.

## 2. RAG 및 임베딩 분석

### Chroma 저장 위치 및 컬렉션
- **저장 위치**: `vectorstore/chroma/` (환경변수 `CHROMA_PERSIST`)
- **클라이언트**: `chromadb.PersistentClient` 싱글톤 패턴 (`src/rag/retrieval.py` 18라인).
- **컬렉션 명명 규칙**: `law_docs_{category_slug}` 형식으로 법률 분야별 분리 저장.
- **임베딩 모델**: `src/config.py`의 `AzureOpenAIEmbeddings` (`text-embedding-3-large`).

### 검색 및 재검색 전략
- **쿼리 재작성**: `src/rag/query_rewrite.py`를 통해 LLM으로 핵심 법률 용어를 추출해 원본 질문과 병합.
- **검색 방식**: `src/rag/retrieval.py`의 `retrieve` 함수에서, 카테고리가 주어지면 단일 컬렉션, 아니면 모든 컬렉션에서 K개 씩 검색 후 점수순 정렬.
- **재검색/근거 반환**: 에이전트의 프롬프트(`src/agents/base_agent.py` 15라인)에서 검색 도구 1회 사용을 강제하고 있으며, 에이전트가 `[metadata]`를 확인해 파일명/페이지 기반으로 근거를 반환하도록 지시되어 있습니다. 자동화된 Agentic Verification이나 재검색 루프(Self-Reflective RAG)는 아직 구현되지 않았습니다.

