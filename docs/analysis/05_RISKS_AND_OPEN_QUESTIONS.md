# AI Legal Helper - Risks & Open Questions

## 1. 주요 위험 요소 (Top 5 Risks)

1. **개인정보 및 프롬프트 로그 평문 노출 (보안)**
   - **위험**: `logging_utils.py` 및 여러 곳에서 사용자 질의와 생성된 프롬프트를 디버그 로그에 그대로 출력 중. 민감한 법률 상담 내용과 개인정보가 로그 서버나 파일 시스템에 평문으로 남을 위험.
   - **대책**: AOP 데코레이터 단에서 로깅 마스킹 필터 도입.

2. **LLM 호출 비용 추적 부재 (비용/운영)**
   - **위험**: 여러 에이전트가 병렬로 LLM을 호출하고 합성하는 등 복잡한 워크플로우를 가지나 토큰 사용량을 전혀 기록하지 않음. 서비스 확장 시 과금 폭탄 위험.
   - **대책**: LangChain 콜백이나 래퍼를 통한 토큰 사용량 통합 집계 로직 개발.

3. **병렬 처리 중 예외 발생 시의 안정성 (안정성)**
   - **위험**: `workflow.py`의 `run_agents_node`에서 `ThreadPoolExecutor`를 사용할 때 하나의 에이전트가 타임아웃이나 예외를 뱉으면 전체 워크플로우 응답 지연/실패로 이어질 가능성이 존재.
   - **대책**: 에이전트 레벨 타임아웃 강제 및 부분 실패를 우아하게 처리하는 Fallback 로직 강화.

4. **단일 MemorySaver 사용으로 인한 상태 유실 (상태 관리)**
   - **위험**: LangGraph `checkpointer=MemorySaver()`를 사용 중이므로 서버 재시작 시 진행 중인 워크플로우나 멀티턴 메모리가 즉시 증발. 상용화 불가.
   - **대책**: SQLite/Postgres 기반의 LangGraph DB Checkpointer 적용.

5. **하드코딩된 에이전트 매핑으로 인한 확장성 저해 (구조적 위험)**
   - **위험**: `workflow.py` 상단에 8개 분야 에이전트를 `import`하고 사전(`_GET_AGENT`)으로 강결합해 두어, 향후 새로운 도메인(예: Temporal Case Agent) 추가 시마다 핵심 코드를 수정해야 함.
   - **대책**: 동적 에이전트 등록(Registry) 패턴을 통해 의존성 분리.

## 2. Open Questions (해결되지 않은 의문점)
- 기존 Streamlit DB(`src/db.py`)와 LangGraph Checkpointer 간의 메모리 정합성(동기화)은 어떻게 보장할 것인가?
- 현재 Chunk Size(800)와 Overlap(120)이 한국 법령 문서의 복잡한 조문 구조를 담기에 충분히 최적화되어 있는가? (조문별 편차가 큼)
- GraphRAG와 Temporal Case Intelligence 기능은 별도의 컬렉션을 운용할 것인가, 기존 메타데이터 필드를 확장하여 사용할 것인가?

