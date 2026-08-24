# 99. 개발자 의사소통 (ledger)

날짜별 진행·검토·설계·이슈 메모. append-only. SDLC registry(01~05)와 분리.

## 문서 영향 증적

프로덕션 변경은 `issue-{번호}-documentation-impact.json`을 이번 diff에 포함합니다.

```json
{
  "schema_version": 1,
  "issue": 42,
  "production_files": ["src/main/example.ts"],
  "targets": {
    "repository_docs": {
      "status": "UPDATED",
      "evidence": ["README.md"]
    },
    "user_manual": {
      "status": "N/A",
      "rationale": "사용자가 발견하거나 설정·조작할 동작과 알려진 한계가 바뀌지 않았다."
    },
    "reusable_learning": {
      "status": "UPDATED",
      "evidence": ["https://github.com/org/book/commit/0123456789abcdef0123456789abcdef01234567"]
    }
  }
}
```

- `UPDATED`: evidence 중 하나 이상이 이번 diff의 로컬 파일이거나 40~64자리 커밋 해시가 있는 외부 URL이어야 합니다.
- `production_files`: 이번 diff에서 gate를 trigger한 production 파일 전부와 정확히 같아야 합니다.
- 로컬 evidence는 대상별 설정 경로와 일치해야 하고, 같은 evidence를 여러 대상에 재사용할 수 없습니다.
- `N/A`: 구체적인 비적용 이유를 적습니다. `PENDING`, 대상 누락, 짧거나 반복된 placeholder는 실패합니다.
- 파일명의 이슈 번호와 `issue` 값이 같아야 하며, symlink receipt는 허용하지 않습니다.
