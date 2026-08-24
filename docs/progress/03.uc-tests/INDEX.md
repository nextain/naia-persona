# 03. 시나리오 테스트 Registry (TEST-S) — V모델 03

<!--
스키마: 이 한 파일 registry. UC(02)와 NFR(01)을 검증하는 시스템/인수 테스트.
추적: 모든 UC는 ≥1 TEST-S로 닫힌다. TEST-S는 ≥1 UC 또는 NFR-REQ를 가리킨다(역추적, orphan 0).
컬럼 = | ID | 검증대상(UC/REQ) | 시나리오 요약 | 형태 | test_ref | 상태 |
예시 = | TEST-S-001 | UC-001 | <시나리오> | 통합 스모크 | <test 경로 또는 —> | Planned |
유닛테스트는 여기 문서화하지 않는다 (코드 src/test, @spec SPEC-### 태그로 추적).
빈 상태(이 안내 주석만) = bootstrap. 채우면 enforce.
-->

_빈 registry — 프로젝트가 시나리오 테스트를 채운다._
