# 재현 방법

`uv.lock`을 사용한다. 의존성을 처음 준비할 때 `uv sync --frozen --all-groups`를 실행한다. 이후 저장소 루트에서 다음 명령 하나로 테스트와 노트북 9개를 실행하고 SVG 14개와 결과 표를 생성한다.

```bash
bash scripts/reproduce.sh --from-results
```

기본 모드도 `--from-results`다. 수학·합성 입력 실험은 다시 계산하고, 학습·정수 평가의 정확도와 학습 곡선은 [공개 수치 기록](results/manifest.json)으로 다시 그린다. 이 과정은 체크포인트를 새로 평가하거나 정확도를 독립적으로 재측정하는 과정이 아니다. 데이터셋과 체크포인트 없이 CPU에서도 사용할 수 있다. CUDA가 있으면 노트북의 합성 샘플 실험과 해당 테스트가 GPU를 사용한다. 몬테카를로 값과 시간은 장비에 따라 달라질 수 있다.

| 모드 | 학습 | 필요한 입력 | 수행하는 작업 |
|---|---|---|---|
| `--from-results` (기본) | 없음 | 저장소의 압축 JSON | 수학·합성 실험 계산, 기록된 학습·정수 평가 수치로 그림·표 재생성 |
| `--skip-training` | 없음 | 완료된 105회 실행의 metrics, 관측 실험 diagnostics, 분석용 체크포인트, CIFAR-10, CUDA | 체크포인트 분석 재계산, 정수 평가, 그림·표 재생성 |
| `--train` | 실행·재개 | CIFAR-10 다운로드 환경, CUDA, 충분한 실행 시간 | 전체 학습 큐, 체크포인트 분석, 정수 평가, 그림·표 재생성 |

```bash
bash scripts/reproduce.sh --skip-training
bash scripts/reproduce.sh --train
```

학습 생략 모드는 float 세 시드와 `nb04-pt-fold`, `nb04-pt-raw`, `nb04-pc-raw`, 단계별 균일 W4A8의 세 시드 `best.pt`를 검사한다. 필요한 파일이 없으면 상대 경로와 이유를 출력하고 종료한다. 기록 재생성 모드로 자동 전환하지 않는다. 체크포인트 분석 함수는 현재 CUDA를 요구한다.

전체 학습 모드는 학습 실행 큐의 완료 작업 건너뛰기·epoch 경계 재개를 사용한다. float 정확도·시간 제한 때문에 큐가 중단되면 파이프라인도 실패한다. 정수 평가는 batch size 32로 실행하며 설정·체크포인트·구현 해시가 일치하면 완료된 평가를 재사용한다. 새 측정을 원하면 기존 `runs/m4/`를 별도 위치에 보관한 후 실행한다.

노트북을 단독 실행하면 로컬 `runs/`를 우선 사용하고 파일이 없을 때 공개 기록을 사용한다. `QUANT_PITFALLS_RESULT_SOURCE=archive` 또는 `live`로 명시할 수 있다. 불완전한 로컬 결과는 완료된 과거 결과로 대체하지 않는다. 노트북은 사용한 상대 경로와 기록된 실험 환경을 출력한다. 노트북 08의 표도 그림과 같은 입력의 세 시드 결과를 집계한다.

산출물은 실행된 `notebooks/`, `docs/figures/`, `runs/reproduce/results.md`, `pytest.xml`, `summary.json`, `log.txt`이다. 요약에는 사용한 수치 출처, 실행 시간, 생성한 SVG 해시가 있다. 로그의 저장소·홈 경로는 익명화한다. `data/`, `runs/`, 체크포인트와 환경 디렉터리는 공개 대상에서 제외한다.

`uv build`의 소스 배포본에도 스크립트·테스트·노트북·수치 기록·문서·lockfile을 포함한다. [uv의 source-include 설정](https://docs.astral.sh/uv/concepts/build-backend/#file-inclusion-and-exclusion)을 사용하며 wheel에는 Python 패키지만 넣는다. 데이터와 체크포인트는 두 배포 형식 모두에 포함하지 않는다.

공개 기록은 원본 JSON을 gzip으로 압축한 것이며 `manifest.json`에 압축 파일과 복원한 JSON의 SHA-256을 기록한다. `m3.json.gz`는 완료된 105회 실행·학습 이력·분석 입력, `m4.json.gz`는 정확도·중간 코드 비교·레이어별 오버플로 통계를 포함한다. `historical.json.gz`에는 최초 수학·합성 실험의 수치·환경·테스트 요약과 학습 시간 추정·스모크 측정·재개 큐 시간을 보관한다. 해시는 파일 손상을 검출하며 게시자의 신원을 인증하지 않는다. 체크포인트와 이미지 원본은 포함하지 않는다.

## 개별 실험 실행

학습에는 CIFAR-10과 GPU를 사용한다. 모델 구조·학습 예산·보정 조건은 [학습 가이드](README.md#6-종단-실험-cifar-10--mobilenetv2)를 따른다.

```bash
uv run python train.py float --epochs 200 --seed 0
uv run python train.py ptq --ckpt runs/float-baseline-s0/best.pt --per-channel
uv run python train.py qat --ckpt runs/float-baseline-s0/best.pt --per-channel --schedule staged --kd
uv run python scripts/run_m3.py --dry-run
uv run python scripts/collect_results.py
uv run python scripts/run_m4.py --estimate-only
```

`run_m3.py`는 105회 학습·보정 비교, `finish_m3.py`는 체크포인트 분석, `run_m4.py`는 정수 누산기 평가를 담당한다. 기존 실행 결과와 연결되는 스크립트명과 `runs/m3/`·`runs/m4/` 경로를 유지한다. 한 번의 스모크 정확도를 완료된 세 시드 결과 표에 넣지 않는다.

## 학습 결과 해석

세 시드의 평균 ± 표본 표준편차, float와 같은 시드로 짝지은 정확도 차이를 보고한다. 정수 평가는 시드 0 하나의 결과이며 1,000장 부분집합과 10,000장 전체 테스트를 구분한다. 장비별 실행 시간이나 Monte Carlo 샘플 값은 달라질 수 있다. [수치 근거](evidence.md)에서 각 결과의 측정 범위를 확인한다.

코드는 [MIT 라이선스](../LICENSE)를 따른다. CIFAR-10의 출처·인용·라이선스 확인 상태는 [데이터셋 문서](datasets.md)에 있다.
