# Naia v2 reference dataset

Naia v1이 Qwen3.8-27B의 일반·안전 능력을 유지했지만 정체성을 바꾸지 못한 결과를 바탕으로 만든 공개 합성 데이터셋입니다. 평가 문항을 복사하지 않고 정체성 표현을 한국어·영어로 다양화했으며, 기억·실행·권한 경계와 일반·안전 예제를 함께 유지합니다.

- identity: 150 (deterministic train split retains at least 120)
- behavioral principles: 40
- memory/runtime/permission boundaries: 40
- general capability anchors: 24
- safety anchors: 16

```bash
python3 examples/naia-v2/build_dataset.py
python3 scripts/validate_dataset.py examples/naia-v2/source.jsonl
```

이 데이터는 실제 사용자 대화나 Alpha persona를 포함하지 않습니다.
