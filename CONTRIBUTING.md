# 기여 가이드

quant-pitfalls는 양자화의 원리와 실패 조건을 작은 실험으로 배우는 교육용 프로젝트다. 설명 개선, 수치 오류 재현, 경계 조건 테스트, 기존 실험의 대조군 추가를 환영한다.

## 개발 환경

Python 3.10 이상과 uv를 사용한다. 저장소 루트에서 실행한다.

```bash
uv sync --frozen --all-groups
uv run pytest -q
uv run jupyter lab notebooks/
```

CPU로 수학·합성 실험과 기록된 결과의 그림을 재생성할 수 있다. CUDA가 필요한 학습·체크포인트 분석의 조건은 [재현 방법](docs/reproduction.md)을 참고한다.

## 코드와 실험

- 들여쓰기는 공백 4칸, 함수·모듈은 `snake_case`, 클래스는 `PascalCase`를 사용한다. 공개 함수에 타입과 dtype·반올림 규칙을 설명한다.
- 코드·주석·docstring·그림 레이블은 영어, 학습 문서는 한국어로 쓴다. 별도 formatter나 linter는 설정하지 않았다.
- 실험에서 바꾸는 변수, 고정하는 조건, 시드, 장비, 표본 수를 명시한다. 합성 입력과 학습된 모델, 한 시드와 여러 시드, 부분집합과 전체 평가를 구분한다.
- 수치 변경은 독립적인 계산이나 참조 구현으로 뒷받침한다. 기존 테스트를 통과시키기 위한 기대값·허용 오차 변경은 피하고, 필요한 변경은 근거와 함께 설명한다.
- 공개 API와 저장된 결과 형식의 호환성을 유지한다. 데이터·체크포인트·실행 로그는 `data/`와 `runs/`에 둔다.

## 검증

변경에 맞는 테스트를 먼저 실행한다. 수치 구현·재현 파이프라인을 바꾸면 전체 테스트와 관련 노트북을 확인한다.

```bash
uv run pytest tests/test_rounding.py -q
bash scripts/reproduce.sh --from-results
uv build
```

`--from-results`는 학습 기록에서 그림과 표를 다시 만든다. 새 정확도 측정은 별도 학습·평가가 필요하다. 측정값을 바꿨다면 [수치 근거](docs/evidence.md), 기록의 해시, 관련 문서와 그림도 함께 갱신한다.

## 변경 제안

커밋 제목은 `Explain clipping error`처럼 변경을 짧게 요약한다. Pull Request에는 해결하는 문제, 재현 명령, 검증 결과를 적고 관련 이슈가 있으면 연결한다. 그림이 바뀌면 새 그림과 해석을 포함한다. 기여한 코드는 저장소의 [MIT 라이선스](LICENSE)를 따른다.
