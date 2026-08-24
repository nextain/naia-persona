# 04. 기능 설계 Registry (SPEC) — V모델 04

<!--
스키마: 이 한 파일 registry. UC(02)를 구현 가능한 기능 단위(SPEC)로 분해.
추적: 모든 SPEC는 ≥1 UC를 가리키고(역추적), ≥1 TEST-F(05)로 닫힌다 (orphan 0).
컬럼 = | ID | 유도 UC | 기능 요약 | area | 상태 | TEST-F |
예시 = | SPEC-001 | UC-001 | <기능 한 줄> | <도메인 태그> | Draft | — |
마크다운은 SPEC(기능 의도)까지만 — 그 아래 unit/함수는 코드(src/main), 유닛테스트는 src/test.
빈 상태(이 안내 주석만) = bootstrap. 채우면 enforce.
-->

_빈 registry — 프로젝트가 기능 설계를 채운다._
