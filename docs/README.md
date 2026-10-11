# 양자화 학습 가이드

> 신경망 양자화(PTQ, QAT, 정수 추론)에서 자주 밟는 함정 9개를 함정마다 작은 실험 하나로 재현하고, 고치는 방법까지 숫자로 보여주는 교육용 프로젝트.

양자화의 수학, 학습 중 상태 변화, 정수 추론의 경계를 실험으로 연결한다. 각 절은 원리와 확인할 변수, 측정 결과를 함께 제시한다. **검증됨**은 수식과 테스트의 기대값이며, **측정**은 시드·표본 수·환경이 명시된 실행 결과다. [재현 방법](reproduction.md)에서 기록된 그림을 다시 그리는 모드와 새 학습·평가 모드를 선택할 수 있다. 모든 수치의 출처는 [근거 표](evidence.md)에 있다.

## 목차

1. [목적과 독자](#1-목적과-독자)
2. [원칙](#2-원칙)
3. [저장소 구조](#3-저장소-구조)
4. [표기](#4-표기)
5. [함정 9개](#5-함정-9개)
6. [종단 실험: CIFAR-10 + MobileNetV2](#6-종단-실험-cifar-10--mobilenetv2)
7. [설치와 실행](#7-설치와-실행)
8. [학습 순서와 확장 실습](#8-학습-순서와-확장-실습)
9. [기여와 라이선스](#9-기여와-라이선스)
10. [참고문헌](#10-참고문헌)

## 1. 목적과 독자

- **독자**: PyTorch로 CNN을 학습해 본 컴퓨터 비전 엔지니어. 양자화는 "8-bit로 바꾸면 작고 빨라진다" 정도로 알고 있는 사람.
- **목표**: 양자화 설정마다 그렇게 정하는 이유를 숫자로 보여준다. "이렇게 해라"보다 "이렇게 안 하면 이만큼 나빠진다"를 재현 가능한 실험으로 보여준다.
- **산출물**
  - 작은 PyTorch 라이브러리 (`src/quant_pitfalls/`)
  - 함정마다 노트북 1개 (`notebooks/01~09`)
  - 종단(end-to-end) 실험 1개 (`train.py`)
  - 학습 측정 기록과 그림·표 재생성 스크립트

## 2. 원칙

1. **근거는 공개 자료뿐.** 논문, 표준(IEEE 754), 공개 프레임워크(PyTorch)의 문서화된 동작만 쓴다.
2. **모든 숫자는 이 저장소 코드로 재현한다.** 노트북 맨 위에 시드, 환경, 실행 시간을 적는다. 문서에 나오는 숫자는 노트북이나 테스트 중 하나가 만들어 낸다.
3. **하드웨어 중립.** 특정 칩이나 런타임을 가정하지 않는다.
   - 기본값은 가장 흔한 설정이다: 가중치 4-bit, 활성화 8-bit, 반올림 nearest, 누산기 32-bit.
   - 하드웨어에서 흔한 제약(내림 시프트, 좁은 누산기, APoT 같은 비균일 레벨)은 옵션으로 켠다.
4. **함정 하나에 변수 하나.** 노트북마다 바꾸는 변수는 하나로 제한하고 나머지는 고정한다.

## 3. 저장소 구조

```
quant-pitfalls/
├── train.py                      # float, PTQ, QAT 학습 CLI
├── src/quant_pitfalls/
│   ├── quant/                    # 반올림, 양자화기, observer, fake-quant
│   ├── layers/                   # BN 접기와 양자화 레이어
│   ├── sim/                      # 정수 누산, 재양자화, 모델 평가
│   ├── models/                   # CIFAR-10용 MobileNetV2
│   ├── analysis.py               # MSE 적분, 클립 탐색, ulp
│   ├── convert.py                # QAT 변환과 스케줄
│   ├── training.py               # 학습·보정·체크포인트
│   └── results.py                # 로컬·기록된 수치 입력
├── notebooks/                    # 01–09 학습 노트북
├── tests/                        # 수치 기대값·경계·상태 검증
├── scripts/                      # 실험 실행, 결과 집계, 재현
├── docs/
│   ├── README.md                 # 학습 가이드
│   ├── reproduction.md           # 실행 모드와 입력 조건
│   ├── evidence.md               # 수치·그림의 근거
│   ├── results/                  # 측정 기록과 해시
│   └── figures/                  # SVG 그림
├── CONTRIBUTING.md               # 기여와 검증 방법
└── LICENSE                       # MIT
```

## 4. 표기

대칭 균일 양자화 (zero-point 0):

```math
s = \frac{\alpha}{q_{\max}}, \qquad
q = \operatorname{clip}\!\Big(\operatorname{round}\big(x/s\big),\ q_{\min},\ q_{\max}\Big), \qquad
\hat{x} = s\,q
```

- $\alpha$는 클립 값(표현할 최대 크기), $s$는 스텝이다.
- 부호 있는 $b$-bit, narrow range: $q_{\max} = 2^{b-1}-1$, $q_{\min} = -q_{\max}$. 레벨은 $2^b-1$개다.
- 부호 있는 $b$-bit, full range: $q_{\min} = -2^{b-1}$. 부호 없는 $b$-bit: $q_{\min} = 0$, $q_{\max} = 2^b - 1$.
- 오차는 입력 분포의 표준편차 $\sigma$ 단위로 쓴다. 비교하는 분포는 모두 분산을 1로 맞춘다 (Laplace는 척도 파라미터를 $\sigma/\sqrt{2}$로 둔다).

학습할 때 round의 기울기는 straight-through estimator(STE)로 근사한다 (Bengio et al., 2013):

```math
\frac{\partial \hat{x}}{\partial x} \approx \mathbf{1}\big[\, q_{\min} \le x/s \le q_{\max} \,\big]
```

## 5. 함정 9개

각 함정은 같은 틀로 쓴다: **질문 → 핵심 → 실험 → 기대 결과 → 구현 위치**.

| # | 노트북 | 한 줄 요약 |
|---|---|---|
| 1 | `01_clipping-mse.ipynb` | 클립 값은 MSE 곡선의 바닥이다 |
| 2 | `02_floor-rounding.ipynb` | 내림(floor)은 반올림 오차를 4배로 만든다 |
| 3 | `03_apot-vs-uniform.ipynb` | APoT는 분포에 따라 이기기도 지기도 한다 |
| 4 | `04_bn-folding.ipynb` | BN은 접고 나서 양자화해야 한다 |
| 5 | `05_observe-before-quant.ipynb` | 범위를 관측하기 전에 양자화를 켜면 생기는 일 |
| 6 | `06_fp16-master-weights.ipynb` | 작은 lr의 QAT는 FP16 가중치로는 멈춘다 |
| 7 | `07_bn-stats-in-qat.ipynb` | QAT 중 BN 통계를 계속 움직이면 무너진다 |
| 8 | `08_ptq-vs-qat.ipynb` | PTQ와 QAT, 그리고 단계별 QAT와 KD |
| 9 | `09_accumulator-saturation.ipynb` | 누산기 비트 폭과 포화 |

### 함정 1. 클립 값은 MSE 곡선의 바닥이다

**질문.** 4-bit로 줄일 때 텐서의 최솟값·최댓값으로 범위를 잡으면 왜 나쁜가?

**핵심.** 양자화 MSE는 반올림 오차와 클리핑 오차의 합이다. $\alpha$를 키우면 스텝 $s$가 커져 반올림 오차가 늘고, 줄이면 범위 밖 값이 잘려 클리핑 오차가 는다.

```math
\mathrm{MSE}(\alpha) \approx
\underbrace{\frac{s^2}{12}\,\Pr\big(\lvert x \rvert \le \alpha\big)}_{\text{반올림}}
+ \underbrace{2\int_{\alpha}^{\infty} (x-\alpha)^2\, p(x)\,dx}_{\text{클리핑}},
\qquad s = \frac{\alpha}{q_{\max}}
```

**실험.**

- 입력 $\mathcal{N}(0,1)$, 4-bit narrow range(−7..7). $\alpha$를 $0.5\sigma$에서 $5\sigma$까지 바꾸며 수치 적분 MSE와 몬테카를로 MSE를 겹쳐 그린다.
- min/max 클립과 비교한다. min/max 클립은 샘플이 많아질수록 커진다.

**기대 결과 (검증됨).** [수치별 테스트·노트북 근거](evidence.md)를 따른다.

| 클립 | $\alpha$ | MSE ($\sigma^2$) | 최적 대비 |
|---|---|---|---|
| MSE 최적 | $2.47\sigma$ | 0.0129 | 1× |
| 고정 | $4\sigma$ | 0.0272 | 2.1× |
| min/max, 샘플 $10^6$개 | 약 $4.8\sigma$~$5.1\sigma$ | 약 0.04 | 약 3.3× |

비트 수를 지정하면 최적 클립은 분포 모양만으로 정해진다. 분석적 클립 계산은 Banner et al. (2019)을 참고한다. 이 논문의 표는 $2^b$개 레벨 기준이라, $2^b-1$개 레벨인 narrow range 값과 조금 다르다.

**구현 위치.** `analysis.expected_mse`, `analysis.optimal_clip`, `quant/uniform.py`

### 함정 2. 내림(floor)은 반올림 오차를 4배로 만든다

**질문.** 하드웨어에서 흔한 오른쪽 시프트처럼 floor로 양자화하면 무엇이 달라지나?

**핵심.** 스텝 단위 오차를 $e = \hat{x} - x$라 하면 nearest는 $e \sim U(-s/2,\, s/2)$, floor는 $e \sim U(-s,\, 0)$이다.

```math
\mathbb{E}[e^2]_{\text{nearest}} = \frac{s^2}{12}, \qquad
\mathbb{E}[e^2]_{\text{floor}} = \underbrace{\frac{s^2}{12}}_{\text{분산}} + \underbrace{\Big(\frac{s}{2}\Big)^2}_{\text{편향}^2} = \frac{s^2}{3}
```

- floor의 편향 $-s/2$는 평균이 0이 아니라서 합산할 때 상쇄되지 않는다.
- $K$개를 더하면 편향은 $-Ks/2$로 $K$배 커지지만, 평균 0인 오차의 표준편차는 $\sqrt{K}$배로만 커진다.

**실험.**

1. 같은 스텝에서 nearest와 floor의 오차 히스토그램을 비교한다.
2. 8-bit에서 각자 MSE 최적 클립을 찾아 비교한다.
3. floor 활성화를 쓴 레이어 출력의 평균 이동을 재고, 다음 레이어의 bias나 BN이 얼마나 흡수하는지 본다.
4. 고치는 방법 두 가지를 비교한다: $\lfloor x/s + 1/2 \rfloor$ (반 LSB를 더한 뒤 시프트), 다음 레이어의 bias 보정.

**기대 결과 (검증됨, $\mathcal{N}(0,1)$, 8-bit full range −128..127).** [노트북 02](../notebooks/02_floor-rounding.ipynb)와 [클립 테스트](../tests/test_clip.py)에서 확인한다.

| 조건 | nearest | floor | floor / nearest |
|---|---|---|---|
| $\alpha = 4\sigma$ 고정 | $8.83\times10^{-5}$ | $3.36\times10^{-4}$ | 3.8× |
| 각자 최적 $\alpha$ | $3.90\sigma$, $8.77\times10^{-5}$ | $3.60\sigma$, $3.02\times10^{-4}$ | 3.45× |

- 반올림 오차만 있는 영역에서는 정확히 4배($s^2/12 \to s^2/3$)다. 클리핑 오차가 섞이면 4배보다 조금 작아진다.
- floor는 최적 클립이 더 작다. 반올림 오차가 4배라서 스텝을 줄이는 쪽으로 균형이 옮겨 간다.

**구현 위치.** `quant/rounding.py`, `quant/uniform.py` (`rounding="floor"`)

### 함정 3. APoT는 분포에 따라 이기기도 지기도 한다

**질문.** 2의 거듭제곱 합으로 만든 레벨(APoT)은 균일 양자화보다 정확한가?

**핵심.** APoT 레벨 (Li et al., 2020). 기본 비트 폭이 $k$이고 항이 $n = b/k$개일 때:

```math
\mathcal{Q}_{\text{APoT}}(\alpha;\, b, k) = \gamma \cdot \Big\{ \sum_{i=0}^{n-1} p_i \Big\}, \qquad
p_i \in \Big\{0,\ 2^{-i},\ 2^{-(i+n)},\ \dots,\ 2^{-(i + (2^k-2)n)}\Big\}
```

- $\gamma$는 최대 레벨이 $\alpha$가 되도록 맞춘다.
- $b=4$, $k=2$이면 레벨은 $\{0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 33, 36, 48\} \cdot \alpha/48$이다. 0 근처는 촘촘하고 끝으로 갈수록 성기다.
- **관찰 포인트:** 32와 33은 간격이 1/48로 거의 겹친다. 바로 옆 간격(24→32는 8/48, 33→36은 3/48)과 비교하면 레벨 하나가 사실상 낭비된다.

**실험.**

- 논문 그림의 예시인 부호 없는 4-bit(16 레벨)를 $\lvert x \rvert$에 적용한다. 대칭 분포에서는 "부호 + 4-bit 크기"(31 레벨) 양자화와 MSE가 같다.
- 같은 레벨 수의 균일 양자화와, 각자 MSE 최적 클립에서 비교한다. 분포는 Gaussian, Laplace, Student-t.
- 학습된 MobileNetV2의 레이어별 가중치 첨도(kurtosis)로 어느 쪽이 이길지 예측하고, 실제 MSE로 확인한다.

**기대 결과 (검증됨, 레벨 수 동일).** [APoT 테스트](../tests/test_apot.py)와 [노트북 03](../notebooks/03_apot-vs-uniform.ipynb)에서 확인한다.

| 분포 | 균일: $\alpha^\ast$, MSE | APoT: $\alpha^\ast$, MSE | APoT / 균일 |
|---|---|---|---|
| Gaussian | $2.90\sigma$, 0.00369 | $3.36\sigma$, 0.00600 | 1.63 (APoT 짐) |
| Laplace | $4.31\sigma$, 0.00908 | $5.73\sigma$, 0.00686 | 0.755 (APoT 이김) |
| Student-t (ν=3, 수치 적분 측정) | $8.274\sigma$, 0.07648 | $18.449\sigma$, 0.03909 | 0.51116 (APoT 이김) |

Student-t 행은 [노트북 03](../notebooks/03_apot-vs-uniform.ipynb)의 결정적 수치 적분 결과다 (실행 시드 0, 1개). 꼬리를 포함하고, 구간 수를 24,000에서 96,000으로 늘렸을 때 MSE의 상대 차이는 균일 1.8e-06, APoT 2.2e-05였다. 최적 클립은 적분 격자와 단계별 탐색으로 얻은 근사값이다.

- MSE 관점의 이득은 분포 모양에 달려 있다. 꼬리가 무거우면 이기고, 가벼우면 진다.
- APoT의 다른 장점은 곱셈을 시프트와 덧셈으로 바꿀 수 있다는 비용 쪽이다.
- 논문은 학습 가능한 클립과 가중치 정규화를 함께 쓴다. 논문의 정확도 이득을 레벨 모양 하나만의 효과로 읽으면 안 된다.

**구현 위치.** `quant/apot.py`

### 함정 4. BN은 접고 나서 양자화해야 한다

**질문.** Conv 가중치를 양자화해 학습하고, 배포할 때 BN을 접으면 무엇이 어긋나나?

**핵심.** 추론 때 BN은 Conv에 접힌다:

```math
W_{\text{fold}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}\, W, \qquad
b_{\text{fold}} = \beta + \frac{\gamma\,(b - \mu)}{\sqrt{\sigma^2 + \epsilon}}
```

- 출력 채널마다 다른 배율 $\gamma/\sqrt{\sigma^2+\epsilon}$이 곱해진다. 공유 스케일로 먼저 양자화한 $W$는 접은 뒤의 격자와 일반적으로 맞지 않는다.
- 배포할 때 다시 양자화하면 추가 반올림으로 학습 때 본 가중치와 배포된 가중치가 달라질 수 있다. 양의 채널별 배율에 맞춰 스케일도 함께 바꾸면 이 차이가 없는 경우도 있다.
- 그래서 학습 중에 $W_{\text{fold}}$를 만들어 그것을 양자화한다 (Jacob et al., 2018; Krishnamoorthi, 2018).

**실험.**

- (a) $W$를 양자화해 학습 → 배포 때 접고 재양자화
- (b) 학습 중 $W_{\text{fold}}$를 양자화
- 학습-배포 정확도 차이와, 학습 때 정수 코드와 배포 정수 코드의 불일치율을 비교한다. per-tensor와 per-channel 모두.

**기대 결과.**

- (b)는 학습 때 본 정수 가중치가 배포 가중치와 같다 (불일치 0). 이 동치는 테스트로 확인한다.
- (a)는 공유 스케일에서 불일치가 생길 수 있다. 실제 학습·배포 정확도 차이는 아래의 학습된 모델 비교에서 확인한다.

**합성 실험 측정 (시드 0, 1개).** 고정된 무작위 BN 통계, 양의 배율, 4-bit min/max 클립으로 두 순서의 결과를 같은 배포 격자에서 비교했다. 코드 불일치율은 per-tensor 5.9028%, per-channel 0%였다. 먼저 접은 경로와 배포 코드의 불일치는 두 방식 모두 0%다. per-channel 결과는 클립이 BN 배율에 비례하는 조건의 교환 가능성을 보여 준다. 학습 중 통계 변화까지 일반화하지 않는다.

**학습 경로.** [Krishnamoorthi (2018), §3.2.2](https://arxiv.org/abs/1806.08342)의 running 통계 보정을 사용한다. $\sigma_R,\sigma_B$는 각각 running·배치 분산에 ε을 더한 표준편차이며, $\mu_B$는 bias 없는 float convolution의 배치 평균이다.

```math
W_R = \gamma W / \sigma_R, \qquad
y_{\text{train}} = \operatorname{Conv}(x,Q(W_R))\,\sigma_R/\sigma_B + \beta - \gamma\mu_B/\sigma_B
```

BN running 통계는 bias를 포함한 float convolution으로 갱신한다. 통계 동결 후에는 running 통계로 접은 배포 경로를 사용한다. `folded_weight_and_bias()`는 모드와 관계없이 running 통계의 배포용 float 값을 반환한다.

**구현 위치.** `layers/conv_bn.py` (`QuantConvBn2d`)

### 함정 5. 범위를 관측하기 전에 양자화를 켜면 생기는 일

**질문.** QAT 첫 스텝부터 fake-quant를 켜면 무슨 일이 생기나?

**핵심.** 범위를 아직 관측하지 않았으면 기본 스케일(예: $s = 1$)이나 첫 배치 하나로 양자화한다. EMA observer

```math
m_t = (1-c)\,m_{t-1} + c\,x_t
```

에서 처음 값의 영향은 $(1-c)^t$로 줄어든다. $c = 0.01$(PyTorch moving-average observer의 기본값)이면:

- 반감기: $\ln 2 / -\ln(0.99) \approx 69$ 스텝
- 변화의 95%를 따라잡는 데: $\ln 20 / -\ln(0.99) \approx 298$ 스텝

그동안 잘못된 스케일로 양자화된 활성화가 손실을 튀게 하고 BN running 통계를 오염시킨다.

**실험.**

- (a) observer만 1 epoch 돌린 뒤 양자화 켜기
- (b) 처음부터 observer와 양자화를 함께 켜기
- (c) 처음부터 양자화만 켜고 스케일은 기본값
- 처음 1,000 스텝의 손실, 활성화 클리핑 비율, BN running mean 변화를 기록한다.

**기대 결과.** 반감기 약 69 스텝, 95% 지점 약 298 스텝은 검증됨. (b)와 (c)의 손실·관측 변화는 세 시드의 [노트북 05](../notebooks/05_observe-before-quant.ipynb)와 [학습 SVG](figures/05_observer-training.svg)에서 측정했다.

**합성 실험 측정 (시드 0, 1개).** [−1,1]에서 [−2,2]로 범위를 바꾸면 α₆₉=1.500163, α₂₉₈=1.949963이었다. 연속 계산의 95% 지점은 298.072852스텝이므로 최초의 정수 스텝은 299다. 기존 테스트의 근삿값과 허용 오차는 유지했다. 관측·양자화 스위치의 네 조합도 검증했다.

**구현 위치.** `quant/observer.py`, `quant/fake_quant.py`

### 함정 6. 작은 lr의 QAT는 FP16 가중치로는 멈춘다

**질문.** 메모리를 아끼려고 `model.half()`로 가중치까지 FP16으로 두고 lr $10^{-6}$으로 미세조정하면?

**핵심.** 업데이트가 반 ulp보다 작으면 덧셈 결과가 원래 값으로 반올림된다.

```math
\mathrm{fl}(w - \eta g) = w \quad \text{if} \quad \lvert \eta g \rvert < \tfrac{1}{2}\,\mathrm{ulp}(w),
\qquad \mathrm{ulp}(w) = 2^{\lfloor \log_2 \lvert w \rvert \rfloor - m}
```

$m$은 가수 비트 수다: FP32 23, FP16 10, BF16 7.

- **SGD**: 업데이트가 $\eta g$이므로 $\lvert g \rvert < \mathrm{ulp}(w) / (2\eta)$이면 사라진다.
- **Adam류**: 업데이트 크기가 대략 $\eta$라서 기울기와 상관없이 $\eta < \mathrm{ulp}(w)/2$이면 사라진다.

**기대 결과 (검증됨, $\eta = 10^{-6}$).** [업데이트 임계값 테스트](../tests/test_fp_update.py)와 [노트북 06](../notebooks/06_fp16-master-weights.ipynb)에서 확인한다.

| 형식 | 가중치 크기 [0.5, 1)의 ulp | SGD: 사라지는 기울기 크기 | Adam류: 사라지는 $\eta$ | 가중치 크기 [2⁻⁷, 2⁻⁶)에서 SGD 임계 | 같은 구간 Adam류 임계 |
|---|---|---|---|---|---|
| FP32 | $2^{-24}$ | $< 0.0298$ | $< 2.98\times10^{-8}$ | $< 4.66\times10^{-4}$ | $< 4.66\times10^{-10}$ |
| FP16 | $2^{-11}$ | $< 244$ | $< 2.44\times10^{-4}$ | $< 3.81$ | $< 3.81\times10^{-6}$ |
| BF16 | $2^{-8}$ | $< 1953$ | $< 1.95\times10^{-3}$ | $< 30.5$ | $< 3.05\times10^{-5}$ |

- FP16 가중치에 lr $10^{-6}$이면, 크기 0.01 근처 가중치도 Adam 업데이트($\approx 10^{-6}$)가 반 ulp($3.81\times10^{-6}$)보다 작아 움직이지 않는다.
- FP32의 임계값은 FP16보다 $2^{13} = 8192$배 낮아서 보통 쓰는 QAT lr 범위에서는 문제가 되지 않는다.
- 고치는 방법: FP32 마스터 가중치. `torch.autocast`는 파라미터를 FP32로 두고 연산만 낮은 정밀도로 한다. 다른 방법으로 확률적 반올림이나 Kahan 합이 있다.

**실험.**

- 학습된 float 모델에서 QAT 1 epoch를 돌린다.
- 조건: (FP32 마스터 / FP16 가중치) × (SGD / Adam) × lr $\{10^{-4}, 10^{-5}, 10^{-6}\}$
- 스텝마다 실제로 값이 바뀐 가중치 비율과, 정수 코드가 바뀐 가중치 비율을 기록한다.

**구현 위치.** `analysis.vanishing_update_threshold`, `train.py --master-dtype`

### 함정 7. QAT 중 BN 통계를 계속 움직이면 무너진다

**질문.** QAT 내내 BN을 train 모드로 두면 어떻게 되나?

**핵심.**

- train 모드 BN은 배치 통계로 정규화한다. 그래서 학습 중 접은 가중치 $W_{\text{fold}}$가 배치마다 흔들린다.
- 격자 경계 근처의 가중치는 배치마다 정수 코드가 바뀐다 (진동).
- running 통계는 양자화된 네트워크의 분포를 늦게 따라간다. 학습 손실과 평가 정확도가 따로 논다.
- Krishnamoorthi (2018)는 일정 시점 이후 BN 통계를 동결하라고 권한다.

**실험.** BN 통계 동결 시점 {처음부터, 중간부터, 동결 안 함} × 배치 크기 {32, 128}. 평가 정확도 곡선, train/eval 손실 차이, 스텝당 가중치 코드 변경률을 비교한다.

**기대 결과 (가설).** 동결하지 않으면 train/eval 차이와 코드 진동이 커지고, 작은 배치에서 더 심하다. 세 시드의 정확도·업데이트 진단은 [노트북 07](../notebooks/07_bn-stats-in-qat.ipynb)과 [학습 SVG](figures/07_bn-freeze-training.svg)에서 확인한다.

**주의.** PyTorch에서 `model.train()`을 부르면 모든 BN이 다시 train 모드가 된다. 동결은 모듈 모드가 아니라 별도 플래그(`freeze_bn_stats`)로 구현해야 epoch마다 `model.train()`을 불러도 풀리지 않는다.

**합성 입력 검증 (시드 0, 1개).** 가중치·observer를 고정하고 같은 합성 배치 24개(배치 크기 32)에 동결 시점만 바꿨다. 처음부터 동결하거나 12스텝부터 동결하면 이후 통계와 코드 변경률은 0이었다. 동결하지 않은 경로의 마지막 스텝 코드 변경률은 11.5741%였다. 매 스텝 `train()`을 호출해도 동결은 유지됐다. 학습된 모델의 정확도와 손실은 아래의 세 시드 비교에서 별도로 해석한다.

**구현 위치.** `layers/conv_bn.py`, `convert.py`

### 함정 8. PTQ와 QAT, 그리고 단계별 QAT와 KD

**질문.** QAT가 필요한 시점은 언제이고, 켜는 순서는 왜 중요한가?

**실험.** 종단 실험의 표([6절](#6-종단-실험-cifar-10--mobilenetv2))를 그대로 쓴다.

- PTQ: W8A8 per-tensor / per-channel, W4A8 per-channel (min/max 클립 vs MSE 클립)
- QAT W4A8: 한 번에 모두 켜기 vs 단계별
- 단계별 QAT + float teacher로 지식 증류(KD)

**단계별 순서 (원칙).** 단계 경계는 QAT 전체 epoch 대비 비율로 지정하고, 노트북에서 비율을 sweep한다.

1. float 모델을 수렴시킨다.
2. observer만 켜서 범위를 관측한다 (양자화는 끔).
3. 가중치 fake-quant를 켠다.
4. 활성화 fake-quant를 켠다.
5. BN 통계를 동결한다.
6. observer를 동결해 스케일을 고정한 뒤 마무리한다.

**KD 손실** (Hinton et al., 2015). $z_s$는 학생(양자화 모델), $z_t$는 float teacher의 로짓이다.

```math
\mathcal{L} = (1-\lambda)\,\mathrm{CE}(y,\, z_s)
+ \lambda\,\tau^2\,\mathrm{KL}\Big(\mathrm{softmax}(z_t/\tau)\ \Big\Vert\ \mathrm{softmax}(z_s/\tau)\Big)
```

**기대 결과 (문헌 기반 가설).** 실제 세 시드의 정확도는 6절 표, 학습 곡선은 [노트북 08](../notebooks/08_ptq-vs-qat.ipynb)에 기록했다.

- W8A8 per-channel PTQ는 float에 가깝다.
- depthwise conv는 채널마다 범위 차이가 커서 per-tensor PTQ가 크게 무너질 수 있다 (Nagel et al., 2019).
- W4는 PTQ만으로는 손실이 크고, QAT가 대부분 회복한다.
- 단계별 QAT는 한 번에 켜는 것보다 초반 손실 급등이 작다 (함정 5와 연결).

**구현 위치.** `convert.py` (`QATSchedule`), `train.py`

### 함정 9. 누산기 비트 폭과 포화

**질문.** 곱셈은 저비트인데 합에는 몇 비트가 필요한가? 넘치면 포화와 랩어라운드 중 어느 쪽이 나은가?

**핵심.** 최악의 경우:

```math
\Big\lvert \sum_{k=1}^{K} w_k a_k \Big\rvert \le K \cdot \max\lvert w \rvert \cdot \max\lvert a \rvert,
\qquad
B_{\text{acc}} = \Big\lceil \log_2\big(K \max\lvert w \rvert \max\lvert a \rvert + 1\big) \Big\rceil + 1
```

**기대 결과 (검증됨).** [수치별 테스트·노트북 근거](evidence.md)를 따른다.

| 설정 | $K$ | 최악의 합 크기 | 필요한 부호 있는 비트 |
|---|---|---|---|
| 8-bit 가중치(±127) × 8-bit 부호 없는 활성화, 3×3×256 | 2304 | 74,615,040 | 28 |
| 4-bit 가중치(±7) × 8-bit 부호 없는 활성화, 1×1×960 | 960 | 1,713,600 | 22 |
| 4-bit 가중치(±7) × 8-bit 부호 없는 활성화, depthwise 3×3 | 9 | 16,065 | 15 |

- **포화 덧셈은 결합법칙이 성립하지 않는다.** 8-bit 누산기에서 [100, 100, −100]을 차례로 더하면 포화 때문에 27이 되고, 순서를 [100, −100, 100]으로 바꾸면 100이 된다.
- **2의 보수 랩어라운드**는 중간에 넘쳐도 최종 합이 범위 안이면 정답(100)이 나온다.
- 실제 합은 최악의 경우보다 훨씬 작다 (대략 $\sqrt{K}$배로 커진다). 그래서 최악 기준보다 좁은 누산기도 대부분의 입력에서는 맞는다. 문제는 드물게 오는 입력이다.

**실험.**

- 학습된 W4A8 MobileNetV2를 정수 시뮬레이터로 돌린다. 누산기 비트 {12, 14, 16, 18, 24, 32} × {포화, 랩}에서 정확도와 레이어별 포화 비율을 잰다.
- 32-bit 누산기에서 정수 경로의 출력이 fake-quant 경로와 bit-exact인지 확인한다.

**정수 경로와 fake-quant가 어긋나는 흔한 원인.**

- 반올림 tie 규칙: `torch.round`는 half-to-even이고, 정수 커널은 보통 반 LSB를 더한 뒤 시프트한다.
- requantize 배율의 고정소수점 근사: 실수 배율 $M$을 $M_0 \cdot 2^{-n}$으로 바꾼다 (Jacob et al., 2018).
- 누산 순서와 포화.

**정수 시뮬레이터 구현 규칙.** 하드웨어의 네이티브 정수 커널이 아닌 수치 시뮬레이터다. 코드 convolution의 절대 합 상한이 2⁵³ 미만이면 float64로 정확한 정수 합을 계산하고 int64로 돌린다. float bias를 convolution에 직접 넣지 않고 정수로 인코딩한 뒤 마지막 항으로 더한다. 폭을 넘는 경우에는 unfold의 입력 채널 → 커널 행 → 커널 열 순서를 유지한다. 매 덧셈 clamp를 보존하는 순서 있는 clipping 함수 합성으로 포화를 가속한다. wrap은 최종 합에 한 번만 적용해도 같다.

- bias는 `floor(b/(s_in*s_w)+0.5)`로 인코딩하며 마지막 덧셈에서도 누산기 오버플로를 적용한다. 중간 표현은 int64이고, 실수 bias를 쓰는 학습 실험 경로와의 차이를 따로 측정한다.
- 재양자화는 정규화한 `M0 * 2^-n`, half-up 또는 명시한 floor를 쓴다. `acc*M0`의 int64 범위를 확인하고, 몫·나머지로 반 LSB 덧셈의 오버플로를 피한다. 채널별 가중치 스케일을 지원한다.
- ReLU6는 누산 코드에서 `[0, floor(6/s_acc+0.5)]`로 제한한다. 잔차의 두 입력은 출력 스케일의 1/2¹⁶ 단위로 맞춘 뒤 넓은 int64로 합하고 한 번 재양자화한다. global pooling은 int64 합과 `s/(H*W)` 배율로 처리한다.
- 첫 convolution과 classifier는 학습 모델과 같은 W8, 중간 convolution은 W4다. classifier도 정수 dot product·bias를 사용하며 마지막에 채널별 스케일로 logits를 복원한다. 잔차·pooling은 int64로 유지해 convolution/linear의 누산 폭만 바꾼다.

**코드 일치 검증 (시드 0, 32장).** 정수 bias·M0/n·반올림·경계 규칙을 공유하고 convolution을 독립적인 float64로 계산하는 `IntegerModel(reference=True)` 옵션과 비교했다. 124개 중간 텐서의 109,502,784개 코드가 모두 같았다. 기존 FP32 fake-quant와는 57,024,512개 활성화 코드 중 8,360,767개가 달랐고, 이 32장의 예측 클래스 차이는 0이었다. 정수 bias 저장과 잔차 guard 비트, 배율 근사, 부동소수점 계산의 영향을 포함한 결과이며 기존 경로와 bit-exact라고 주장하지 않는다. 근거는 [노트북 09](../notebooks/09_accumulator-saturation.ipynb)의 비교 표와 `runs/m4/parity.json`이다.

**bias와 최악 비트 폭.** 위의 검증 식은 곱의 합에 대한 식이다. 실제 체크포인트의 매우 작은 가중치 스케일 채널에서는 정수 bias가 더 많은 비트를 요구했다. 이 시드에서는 bias를 포함한 최악 상한이 최대 47비트였고, 32비트에서도 첫 레이어 출력 누산의 28.125%가 마지막 bias 항에서 범위를 넘었다. 포화 비율은 bias를 포함하고, 곱의 최악 비트·실제 곱 합·bias 크기를 노트북에서 구분한다. 폭넓은 bias 인코딩도 상한을 확인하며 int64 밖이면 실행을 거부한다.

**구현 위치.** `sim/accumulator.py`, `sim/integer.py`

## 6. 종단 실험: CIFAR-10 + MobileNetV2

**왜 이 조합인가.**

- CIFAR-10은 노트북 GPU 하나로 수 시간 안에 float 학습과 여러 QAT 변형을 돌릴 수 있다.
- MobileNetV2는 depthwise conv, ReLU6, 선형 bottleneck이 있어 함정 1~9가 모두 드러난다.

**설정.**

- 모델: torchvision `mobilenet_v2`와 같은 inverted residual 구조. 32×32 입력에 맞춰 stem과 첫 번째 stride-2 단계를 stride 1로 바꿔 마지막 feature map을 4×4로 둔다.
- 데이터: CIFAR-10, 표준 증강(4px 패딩 후 random crop, 좌우 반전).
- 시드 3개로 평균 ± 표준편차를 보고한다.
- 모든 PTQ/QAT는 같은 float 체크포인트에서 출발한다. QAT 예산은 float 학습 epoch의 10~15%.
- 8 GB GPU에서 돌아가는 배치 크기를 기본값으로 둔다.

**시드 대응.** float은 시드 0·1·2로 각각 200 epoch 학습한다. 각 설정의 시드 i는 float 시드 i의 `best.pt`에서 시작한다. 표는 완료된 세 시드의 최고 평가 top-1 평균 ± 표본 표준편차이며, 차이는 시드별 float와 짝지어 평균한다. 미완료·스모크 실행은 결과 표에 넣지 않는다.

**학습 레시피.** float은 batch 128, SGD(lr=0.1, momentum=0.9, weight decay=5e-4), 200 epoch cosine 감쇠, dropout=0.2다. 증강은 4px random crop·좌우 반전이며 CIFAR-10 채널 평균·표준편차로 정규화한다. float 학습만 BF16 autocast를 사용하고 파라미터는 FP32로 유지한다. 모든 평가는 FP32다. QAT는 기본 Adam(lr=1e-4, eps=1e-8, weight decay=0), 20 epoch cosine 감쇠, FP32 연산을 사용한다. KD teacher는 동일 시드의 float 모델을 eval·기울기 비활성 상태로 두고 λ=0.5, τ=4를 사용한다.

**양자화 경계.** 첫 Conv와 마지막 classifier 가중치는 uniform 8-bit, 중간 Conv는 지정한 가중치 양자화기를 쓴다. 가중치의 채널 축은 출력 채널이다. 정규화된 첫 입력은 signed 8-bit, ReLU6 뒤·평균 pooling 뒤는 unsigned 8-bit, 선형 bottleneck과 잔차 합 뒤는 signed 8-bit다. 잔차 합과 평균 pooling은 dequantize된 값으로 계산하고 결과를 다시 양자화한다. bias와 classifier logits는 float로 둔다. `--fold-order raw`는 노트북 04의 비교용 옵션이며 기본값은 먼저 접기다.

**보정과 스케줄.** PTQ는 증강 없는 train 배치 32개에서 전체 min/max를 집계한다. MSE 클립은 100개 후보를 사용하며 activation은 전체 배치에서 최대 16,384개, 가중치는 채널당 최대 2,048개의 균등 샘플로 근사한다. QAT도 같은 초기 보정 후 EMA를 갱신한다. 기본 경계는 가중치 0.05, 활성화 0.10, BN 동결 0.50, observer 동결 0.75다. one-shot은 가중치·활성화 시작만 0으로 바꾼다. 노트북 05는 초기 보정을 생략해 관측 워밍업을 비교한다.

**저장 정밀도.** `--master-dtype fp16`은 Conv/Linear 가중치의 마지막 저장만 FP16으로 반올림한다. forward, 기울기, optimizer 상태·Δ와 BN·bias는 FP32다. FP16 저장본을 FP32 연산용 파라미터에 다시 복사하므로 작은 업데이트를 누적하는 FP32 마스터는 남기지 않는다.

**APoT 비트 표기.** 이 구현의 APoT b=4는 크기 4-bit에 부호가 추가되어 31레벨이다. 균일 W4의 15레벨과 별도로 균일 W5의 31레벨 대조군을 실행한다. APoT와 균일 양자화기의 레벨 수가 같을 때의 비교는 이 대조군을 사용한다.

**결과 표 (학습 실험 측정, 설정마다 시드 3개).** 평균 ± 표본 표준편차다. 같은 시드의 float 체크포인트를 사용한 차이를 pp로 표시했다. [노트북 08](../notebooks/08_ptq-vs-qat.ipynb)과 [집계 코드](../scripts/collect_results.py)의 완료 결과를 그대로 반영했다. 원본 학습 수치는 [공개 기록](results/manifest.json)의 `m3.json.gz`에도 보관한다.

| 설정 | 가중치 | 활성화 | top-1 (%) | float 대비 |
|---|---|---|---|---|
| float | FP32 | FP32 | 92.40 ± 0.12 (n=3) | — |
| PTQ, per-tensor | 8 | 8 | 91.57 ± 0.50 (n=3) | -0.83 pp |
| PTQ, per-channel | 8 | 8 | 92.27 ± 0.13 (n=3) | -0.13 pp |
| PTQ, per-channel, min/max 클립 | 4 | 8 | 88.80 ± 0.58 (n=3) | -3.61 pp |
| PTQ, per-channel, MSE 클립 | 4 | 8 | 88.02 ± 0.80 (n=3) | -4.38 pp |
| QAT, 한 번에 | 4 | 8 | 92.07 ± 0.03 (n=3) | -0.33 pp |
| QAT, 단계별 | 4 | 8 | 92.22 ± 0.07 (n=3) | -0.19 pp |
| QAT, 단계별 + KD | 4 | 8 | 92.28 ± 0.21 (n=3) | -0.12 pp |
| QAT, 단계별, APoT 가중치 | 4 + sign | 8 | 92.21 ± 0.28 (n=3) | -0.19 pp |
| QAT, 단계별, 균일 31레벨 (APoT 대조군) | 5 | 8 | 92.13 ± 0.15 (n=3) | -0.27 pp |
| QAT, 단계별, floor 활성화 | 4 | 8 | 92.03 ± 0.07 (n=3) | -0.38 pp |

MSE 클립 PTQ는 이번 측정에서 min/max PTQ보다 0.78 pp 낮았다. 클립의 국소 가중치 MSE 최소화가 종단 정확도 향상을 보장하지는 않는다. float 평균은 92.40%였다. 문헌에서 보고한 정확도와 비교할 때는 모델 구조·전처리·학습 예산 차이를 함께 고려해야 한다.

**확장 (선택).** Pascal VOC + 작은 FCOS로 같은 표를 mAP로 다시 만든다. 검출 헤드의 박스 회귀 출력이 분류 출력보다 양자화에 민감한지 확인한다.

## 7. 설치와 실행

Python 3.10 이상과 uv를 사용한다. 저장소의 `uv.lock`으로 의존성을 고정한다. 데이터셋과 체크포인트 없이 시작하려면 [기록 기반 재현](reproduction.md)의 `--from-results` 모드를 사용한다.

```bash
uv sync --frozen --all-groups # numpy, torch, torchvision + pytest, jupyterlab, matplotlib
uv run pytest                 # 수치·상태·정수 경로 검증
uv run jupyter lab notebooks/

uv run python train.py float --epochs 200
uv run python train.py ptq --ckpt runs/float-baseline-s0/best.pt --w-bits 8 --a-bits 8 --per-channel
uv run python train.py qat --ckpt runs/float-baseline-s0/best.pt --w-bits 4 --a-bits 8 --schedule staged --per-channel --kd
uv run python scripts/run_m3.py --dry-run
uv run python scripts/run_m3.py             # 백그라운드 실행 권장; 오류·낮은 float 정확도에서 중단
uv run python scripts/collect_results.py    # 완료된 세 시드만 집계
```

- `python train.py --help`는 torch 없이도 동작한다 (torch는 실행 함수 안에서 불러온다).
- GPU 실행 환경은 설치한 PyTorch 빌드와 드라이버가 장비를 지원해야 한다. 재현한 장비와 라이브러리 버전은 각 노트북의 환경 표와 [측정 기록](evidence.md)에 있다.

### 클리핑·반올림·레벨 배치

합성 샘플 실험은 시드 0 하나를 사용했다. 수치 적분은 CPU float64이며, 몬테카를로 실험과 양자화기의 CPU/CUDA 일치 검증은 GPU 환경에서 수행했다. 결과 JSON과 PNG 미리보기는 `runs/m1/`에 저장한다.

| 노트북 | 측정 결과 | SVG |
|---|---|---|
| [01: 클립 MSE](../notebooks/01_clipping-mse.ipynb) | 최적 α=2.474σ, MSE=0.01289; min/max(n=10⁶)은 최적 대비 3.360배 | [그림](figures/01_clipping-mse.svg) |
| [02: floor](../notebooks/02_floor-rounding.ipynb) | 클리핑 없는 MSE 비 3.998, 각자 최적 클립에서 3.448 | [그림](figures/02_floor-rounding.svg) |
| [03: APoT](../notebooks/03_apot-vs-uniform.ipynb) | APoT/균일 MSE 비: Gaussian 1.626, Laplace 0.755, Student-t 0.511 | [그림](figures/03_apot-vs-uniform.svg) |

노트북은 아래 명령으로 처음부터 끝까지 다시 실행한다.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_clipping-mse.ipynb notebooks/02_floor-rounding.ipynb notebooks/03_apot-vs-uniform.ipynb
```

노트북 02의 실제 레이어 bias 보정과 노트북 03의 학습된 가중치 첨도 분석은 세 시드로 실행했다. [bias 보정](figures/02_floor-layer-bias.svg)과 [첨도 비교](figures/03_weight-kurtosis.svg)에 측정 결과가 있다.

### BN과 관측 상태

GPU float64, 시드 0 하나로 실행했다. 결과 JSON과 PNG 미리보기는 `runs/m2/`에 저장한다. CPU/CUDA 테스트는 체크포인트 복원, 채널별 관측과 STE, 0인 채널, 학습 경로의 출력·기울기·BN 갱신, 동결 및 배포 동치를 확인한다.

| 노트북 | 측정 결과 | SVG |
|---|---|---|
| [04: BN 접기](../notebooks/04_bn-folding.ipynb) | 양자화 순서 코드 불일치: per-tensor 5.90%, per-channel 0% (양의 배율에 비례하는 클립) | [그림](figures/04_bn-folding.svg) |
| [05: EMA](../notebooks/05_observe-before-quant.ipynb) | 반감기 68.968, 95% 지점 298.073스텝; 최초 정수 스텝은 각각 69, 299 | [그림](figures/05_observe-before-quant.svg) |
| [07: BN 동결](../notebooks/07_bn-stats-in-qat.ipynb) | 고정 가중치·observer에서 동결 이후 통계·코드 변화 0 | [그림](figures/07_bn-stats-in-qat.svg) |

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_bn-folding.ipynb notebooks/05_observe-before-quant.ipynb notebooks/07_bn-stats-in-qat.ipynb
```

학습된 모델의 비교 결과도 노트북에 포함한다. [접기 순서](figures/04_bn-folding-trained.svg), [관측 워밍업](figures/05_observer-training.svg), [BN 동결·배치 크기](figures/07_bn-freeze-training.svg)에 시드 3개의 측정 결과가 있다.

### 학습 시간과 실행 큐

GPU에서 시드 0 하나로 전체 train/test 1 epoch를 측정했다. float은 7.90초·최대 1.25 GiB, observer·BN 갱신을 켠 W4A8 QAT는 38.62초·최대 3.64 GiB였다. 배치 32, APoT, FP16 저장 및 업데이트·클리핑 진단도 짧은 실행으로 확인했다. 이 실행들의 낮은 정확도는 속도 측정용이며 종단 결과가 아니다.

`scripts/run_m3.py`의 큐는 float 3회, PTQ 12회, QAT·노트북 변형 90회로 총 105회다. float 600 epoch, QAT 1,116 epoch이며 후자 중 180 epoch는 배치 32다. 업데이트 진단 비용을 포함한 추정은 23.14시간, 30% 여유를 포함하면 30.08시간이다. 비용·추정 조건은 `runs/m3/estimate.json`에 보관한다. float 최고 top-1이 92% 미만이면 레시피 검토를 위해 중단하고, 전체 실행이 48시간을 넘으면 다음 작업을 시작하지 않는다.

진행 상태는 `runs/m3/status.json`, 실행별 산출물은 `runs/<설정>-s<시드>/`에 있다. `--resume <last.pt>`는 epoch 경계의 optimizer·scheduler·RNG·DataLoader 상태를 복원하며 학습 하이퍼파라미터를 검증한다. 큐를 다시 실행하면 완료된 작업을 건너뛰고 마지막 checkpoint에서 이어간다. 학습이 모두 끝나면 가중치 첨도·bias 보정·접기 순서 분석과 노트북 02–08을 자동 실행한다.

### 학습·배포 정확도 비교

완료된 105회 실행과 체크포인트 분석은 `runs/m3/notebook-data.json`과 [공개 수치 기록](results/manifest.json)의 `m3.json.gz`에 보관했다. [노트북 04](../notebooks/04_bn-folding.ipynb)에서 접기 순서 수치를 집계한다. 세 시드 평균으로 raw 양자화 후 BN을 접고 배포 격자에 재양자화한 per-tensor 모델은 85.03% → 57.98%로 27.05 pp 떨어졌다. per-channel에서는 92.09% → 92.15%였고, fold-first는 배포 시 변화가 없었다. 이 차이는 동일한 배포 절차로 측정한 결과다.

[FP16 저장 실험](figures/06_fp16-master-weights.svg), [PTQ·QAT 학습 곡선](figures/08_ptq-vs-qat.svg)도 생성했다. 실행 시간 49,564초는 KD 오류 수정 후 큐 재개부터의 시간이며, 이미 끝난 작업의 학습 시간과 합친 총 GPU 시간이 아니다.

### 정수 누산기 비교

시드 0의 단계별 W4A8 체크포인트로 측정했다. 부분집합은 동일한 CIFAR-10 테스트 첫 1,000장이고, 전체 검증은 10,000장이다. 단일 시드 결과이며 학습 실험의 세 시드 평균과 구분한다. 설정마다 레이어별 포화 비율과 최악 비트·정확한 합의 최대 크기를 `runs/m4/results.json`에 기록했다.

| 누산기 비트 | 포화 top-1 (%) | 랩 top-1 (%) | 표본 수 |
|---|---|---|---|
| 12 | 9.00 | 11.30 | 1,000 (시드 1개) |
| 14 | 14.60 | 8.90 | 1,000 (시드 1개) |
| 16 | 64.80 | 11.60 | 1,000 (시드 1개) |
| 18 | 71.90 | 9.80 | 1,000 (시드 1개) |
| 24 | 92.80 | 92.80 | 1,000 (시드 1개) |
| 32 | 92.80 | 92.80 | 1,000 (시드 1개) |

같은 체크포인트의 FP32 fake-quant는 부분집합에서 92.80%, 전체에서 92.14%였다. 전체 테스트에서 **32비트 포화 92.18%, 16비트 포화 65.59%**를 확인했다. 24비트 포화·랩과 32비트 포화·랩은 부분집합에서 모두 92.80%였지만, 24비트는 전체 테스트로 확인하지 않았다. 작은 폭에서는 최종 합만 검사해서 드러나지 않는 중간 오버플로와 bias 덧셈이 영향을 준다.

실행과 전체 검증은 약 151.70초였다. 첫 스모크로 산출한 30% 여유 포함 추정은 0.38시간이며 모든 설정을 가장 느린 설정의 속도로 계산해 보수적이었다. [노트북 09](../notebooks/09_accumulator-saturation.ipynb)를 실행해 [SVG](figures/09_accumulator-saturation.svg)를 만들었다.

```bash
uv run python scripts/run_m4.py --estimate-only
uv run python scripts/run_m4.py --batch-size 32
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_accumulator-saturation.ipynb
```

### 전체 그림과 표 재생성

```bash
uv sync --frozen --all-groups
bash scripts/reproduce.sh --from-results
```

이 명령은 수학·합성 실험을 다시 계산하고 기록된 학습 실험·정수 실험 입력으로 노트북 9개·SVG 14개·결과 표를 재생성한다. 새 학습 정확도 측정과는 구분한다. 기존 체크포인트 분석은 `--skip-training`, 전체 큐 실행·재개는 `--train`을 사용한다. [재현 조건](reproduction.md), [수치 근거](evidence.md), [참고문헌 확인](references.md), [데이터셋 표기](datasets.md)를 제공한다.

## 8. 학습 순서와 확장 실습

1. **오차의 기초 — 노트북 01–03.** 입력 분포와 레벨 수를 고정하고 클립 값·반올림·레벨 배치를 각각 비교한다. 적분 값과 샘플 추정이 일치하는지도 확인한다.
2. **상태와 배포 — 노트북 04·05·07.** 가중치와 관측 범위를 고정한 합성 실험부터 시작한다. 이후 학습된 모델에서 BN 접기 순서, 관측 워밍업, 동결 시점의 영향을 해석한다.
3. **학습과 정수 추론 — 노트북 06·08·09.** 작은 업데이트의 저장, 단계별 QAT, 정수 누산기 오버플로를 연결한다. 같은 체크포인트와 평가 이미지를 사용한다.

다음은 기존 결과에서 출발하는 확장 과제다. 각각 대조군과 바꿀 변수 하나를 먼저 정한다.

- Gaussian의 비트 수를 바꾸어 최적 클립과 MSE 곡선을 비교하고 이론적 이유를 설명한다.
- EMA 계수와 입력 범위를 바꿔 반감기·관측 지연을 예측하고 측정한다.
- 원소의 순서만 바꿔 포화 누산 결과가 달라지는 예를 만들고 wrap과 비교한다.
- 학습된 레이어에서 가중치 MSE가 낮아져도 정확도가 좋아지지 않는 사례를 조사한다.

실험을 추가할 때는 표본 수, 시드, 고정 조건과 새로 측정한 값을 함께 기록한다.

## 9. 기여와 라이선스

설명 개선, 수치 오류 재현, 경계 조건과 대조군 추가는 [기여 가이드](../CONTRIBUTING.md)를 따른다. 코드에는 [MIT 라이선스](../LICENSE)가 적용된다. 데이터셋 출처와 이용 조건은 [데이터셋 문서](datasets.md), 논문 판본과 인용 정보는 [참고문헌 확인](references.md)에 정리했다.

## 10. 참고문헌

**양자화 전반**

- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
- R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," 2018. [arXiv:1806.08342](https://arxiv.org/abs/1806.08342)
- M. Nagel et al., "A White Paper on Neural Network Quantization," 2021. [arXiv:2106.08295](https://arxiv.org/abs/2106.08295)
- H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation," 2020. [arXiv:2004.09602](https://arxiv.org/abs/2004.09602)

**클립과 양자화기**

- R. Banner, Y. Nahshan, D. Soudry, "Post training 4-bit quantization of convolutional networks for rapid-deployment," [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/c0a62e133894cdce435bcb4a5df1db2d-Abstract.html). [arXiv:1810.05723](https://arxiv.org/abs/1810.05723) (판본별 제목·저자는 [확인 기록](references.md) 참고)
- Y. Li, X. Dong, W. Wang, "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks," ICLR 2020. [arXiv:1909.13144](https://arxiv.org/abs/1909.13144)
- J. Choi et al., "PACT: Parameterized Clipping Activation for Quantized Neural Networks," 2018. [arXiv:1805.06085](https://arxiv.org/abs/1805.06085)
- S. K. Esser et al., "Learned Step Size Quantization," ICLR 2020. [arXiv:1902.08153](https://arxiv.org/abs/1902.08153)
- S. P. Lloyd, "Least squares quantization in PCM," IEEE Trans. Information Theory, 1982.
- J. Max, "Quantizing for minimum distortion," IRE Trans. Information Theory, 1960.

**학습**

- Y. Bengio, N. Léonard, A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," 2013. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- M. Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction," ICCV 2019. [arXiv:1906.04721](https://arxiv.org/abs/1906.04721)
- P. Micikevicius et al., "Mixed Precision Training," ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network," 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

**정수 연산과 부동소수점**

- R. Ni et al., "WrapNet: Neural Net Inference with Ultra-Low-Precision Arithmetic," [ICLR 2021](https://openreview.net/pdf?id=3SqrRe8FWQ-). arXiv판: [2007.13242](https://arxiv.org/abs/2007.13242), 제목은 Ultra-Low-Resolution.
- I. Colbert et al., "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance," ICCV 2023. [arXiv:2308.13504](https://arxiv.org/abs/2308.13504)
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic," ACM Computing Surveys, 1991.
- [IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019](https://ieeexplore.ieee.org/document/8766229).

**모델과 데이터**

- M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
- Z. Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection," ICCV 2019. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
- A. Krizhevsky, ["Learning Multiple Layers of Features from Tiny Images"](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Tech. report, 2009. [CIFAR 배포·라이선스 확인](datasets.md)
- M. Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge," IJCV 2010.

**프레임워크 문서**

- PyTorch 공식 문서: [FakeQuantize](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html), [MovingAverageMinMaxObserver](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html), [torch.round](https://docs.pytorch.org/docs/stable/generated/torch.round.html).
