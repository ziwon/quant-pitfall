# 수치와 그림의 근거

문서의 기대값, 합성 실험 측정, 학습 측정을 구분한다. [재현 스크립트](reproduction.md)는 테스트와 수학·합성 실험을 다시 실행하고, 학습 그림은 공개 기록으로 재생성한다. 체크포인트 재평가는 별도 모드다.

| 문서에 사용한 수치·조건 | 생성·검증 위치 | 공개 기록·그림 |
|---|---|---|
| Gaussian W4 최적 클립 2.474445σ, MSE 0.01288928; 고정 클립·min/max·샘플 수 비교 | [노트북 01](../notebooks/01_clipping-mse.ipynb), [클립 테스트](../tests/test_clip.py) | [SVG](figures/01_clipping-mse.svg), `historical.json.gz`의 `m1` |
| nearest/floor 오차 모멘트, 클리핑 없는 비 3.998, 최적 클립 비 3.448, full/narrow 코드 범위 | [노트북 02](../notebooks/02_floor-rounding.ipynb), [반올림](../tests/test_rounding.py)·[클립](../tests/test_clip.py)·[경계](../tests/test_quantization_boundaries.py) 테스트 | [합성 SVG](figures/02_floor-rounding.svg), [학습 레이어 bias SVG](figures/02_floor-layer-bias.svg), `m3.json.gz`의 `floor_bias` |
| APoT 16개 크기 레벨·부호 포함 31레벨; Gaussian 1.626, Laplace 0.755, Student-t(ν=3) 0.511; 적분 수렴 오차 | [노트북 03](../notebooks/03_apot-vs-uniform.ipynb), [APoT](../tests/test_apot.py)·[적분 경계](../tests/test_quantization_boundaries.py) 테스트 | [분포 SVG](figures/03_apot-vs-uniform.svg), [첨도 SVG](figures/03_weight-kurtosis.svg), `m3.json.gz`의 `weight_distributions` |
| 합성 BN 접기 코드 차이 per-tensor 5.90%·per-channel 0%; 학습 raw→배포 top-1 85.03→57.98%·92.09→92.15%, −27.05 pp (n=3) | [노트북 04](../notebooks/04_bn-folding.ipynb), [BN](../tests/test_conv_bn.py)·[상태](../tests/test_quantization_state.py)·[배포](../tests/test_training.py) 테스트 | [합성 SVG](figures/04_bn-folding.svg), [학습 SVG](figures/04_bn-folding-trained.svg), `m3.json.gz`의 `fold_order` |
| EMA c=0.01의 반감기 약 69·95% 약 298스텝, 첫 정수 299; α₆₉=1.500163·α₂₉₈=1.949963; 워밍업 손실·클리핑·BN 변화 (n=3) | [노트북 05](../notebooks/05_observe-before-quant.ipynb), [observer](../tests/test_observer.py)·[스위치](../tests/test_quantization_state.py) 테스트 | [EMA SVG](figures/05_observe-before-quant.svg), [학습 SVG](figures/05_observer-training.svg), `m3.json.gz`의 `reports/nb05-*` |
| FP16·BF16·FP32 가수 비트·ulp·SGD/Adam 업데이트 소실 임계값; lr와 optimizer별 가중치·코드 변경 비율 | [업데이트](../tests/test_fp_update.py)·[0과 subnormal](../tests/test_quantization_boundaries.py) 테스트, [노트북 06](../notebooks/06_fp16-master-weights.ipynb) | [SVG](figures/06_fp16-master-weights.svg), `m3.json.gz`의 `reports/nb06-*` |
| BN 동결 뒤 코드·통계 유지; batch 32/128와 동결 경계 0/.5/1에 따른 손실·정확도 (n=3) | [노트북 07](../notebooks/07_bn-stats-in-qat.ipynb), [BN 동결](../tests/test_conv_bn.py)·[QAT 상태](../tests/test_quantization_state.py) 테스트 | [합성 SVG](figures/07_bn-stats-in-qat.svg), [학습 SVG](figures/07_bn-freeze-training.svg), `m3.json.gz`의 `reports/nb07-*`와 기본 staged 실행 |
| 분류 결과 표의 평균·표본 표준편차·시드 대응 pp; float 92.40±0.12%, staged W4A8 92.22±0.07%, 각 n=3; 학습 곡선·스케줄 경계 | [노트북 08](../notebooks/08_ptq-vs-qat.ipynb), [집계 코드](../scripts/collect_results.py), [집계 검증](../tests/test_experiment_results.py)·[기록 검증](../tests/test_result_archives.py) | [SVG](figures/08_ptq-vs-qat.svg), `m3.json.gz`의 `reports`와 `table` |
| MobileNetV2 파라미터 2,236,682개·마지막 4×4; float 3×200 epoch, PTQ 12회, QAT 1,116 epoch, 총 105회; 레시피·배치·시드 | [구조 테스트](../tests/test_training.py), [실행 큐](../scripts/run_m3.py), [큐 테스트](../tests/test_experiment_results.py), [학습 CLI](../train.py) | `m3.json.gz`의 실행별 `config`·`history`, `historical.json.gz`의 `m3_estimate` |
| 학습 1 epoch 스모크 float 7.90초/1.25 GiB·QAT 38.62초/3.64 GiB; 추정 23.14시간·30% 여유 30.08시간; 재개 큐 49,564초 | [학습·계측 코드](../src/quant_pitfalls/training.py), [큐 제한](../scripts/run_m3.py) | `historical.json.gz`의 `m3_benchmarks`·`m3_estimate`·`m3_queue`. 원래 장비의 기록이며 시간 재현을 보장하지 않는다. 큐 시간은 총 GPU 학습 시간이 아니다. |
| 누산 곱 상한 74,615,040/1,713,600/16,065와 28/22/15비트; 포화 순서 결과 27/100, wrap 100 | [누산기 테스트](../tests/test_accumulator.py), [정수 경계 테스트](../tests/test_integer_model.py), [노트북 09](../notebooks/09_accumulator-saturation.ipynb) | [SVG](figures/09_accumulator-saturation.svg) |
| 정수 평가 n=1 전체 10,000장: 포화32 92.18%·포화16 65.59%·FP32 fake-quant 92.14%; 부분집합 12설정; bias 포함 최대 47비트·첫 레이어 초과 28.125% | [노트북 09](../notebooks/09_accumulator-saturation.ipynb), [실행 코드](../scripts/run_m4.py) | `m4.json.gz`의 `sweep`·`full`·`fake_quant_full`·레이어 통계 |
| 고정소수점 참조: 32장·124텐서·109,502,784코드 일치; 기존 FP32는 57,024,512개 중 8,360,767개 차이·예측 차이 0; 시간 151.70초·추정 0.38시간 | [노트북 09](../notebooks/09_accumulator-saturation.ipynb), [독립 정수 참조 테스트](../tests/test_integer_model.py), [코드 비교](../scripts/run_m4.py) | `m4.json.gz`의 `parity`·`estimate`·`runtime_seconds`. 기존 FP32와 bit-exact라는 의미가 아니다. |
| 현재 테스트 통과·실패·스킵 수, 노트북·그림 생성 수 | [재현 스크립트](../scripts/reproduce.py) | 실행 후 `runs/reproduce/pytest.xml`과 `summary.json`에서 확인한다. |

압축 JSON과 SHA-256은 [수치 기록 manifest](results/manifest.json)에 있다. 최초 수학·합성 실험의 실행 시간·GPU·Python·torch 정보도 `historical.json.gz`에 보관한다. CIFAR-10 구성·다운로드·인용과 라이선스 확인 상태는 [데이터셋 문서](datasets.md)를 따른다. 외부 논문의 정확도를 이 프로젝트의 측정값으로 옮기지 않았다.

[README 결과 지도](figures/readme/experiment-overview.svg)와 [11개 설정 정확도 그림](figures/readme/end-to-end-accuracy.svg)은 [생성 스크립트](../scripts/render_readme_figures.py)가 SHA-256을 검증한 기록에서 만든다. 클리핑·반올림·APoT는 `historical.json.gz`, 학습 비교는 `m3.json.gz`를 사용한다. 노트북 05의 10.00·92.27·92.15%와 노트북 07의 배치 128 동결 시점별 92.14·92.22·92.01%는 해당 실행의 `best_top1`을 시드 3개로 평균한 값이다. 노트북 06의 SGD·lr=10⁻⁶ 변경률 27.692%(FP32)·0.028%(FP16)는 각 한-epoch 실행의 `history[0].weight_change_fractions`를 스텝별로 평균하고 다시 시드 3개로 평균한다. 정확도나 누적 변경률과 구분한다. 노트북 09의 순서 예제는 `accumulate`로 다시 계산한다. `uv run python scripts/render_readme_figures.py --check`로 두 SVG의 최신 여부를 확인한다.

`m1`·`m2`·`m3`·`m4`는 기존 기록과 실행 경로를 연결하는 식별자다. 각각 수학 실험, 상태 검증, 학습 비교, 정수 평가를 뜻한다. 측정 기록의 키와 파일명은 재현 입력의 호환성을 위해 유지한다.

최초 합성 실험은 시드 0 하나로 실행했다. 수치 적분은 CPU float64, 몬테카를로 샘플은 GPU를 사용했다. 클리핑 없는 floor/nearest MSE 비는 최초 기록에서 3.998335, 다른 장비의 재실행에서 3.998671이었다. 이와 같은 샘플 추정·실행 시간의 변동은 결정적 적분 값과 구분한다.
