# 참고문헌과 판본

2026-10-10에 arXiv 원문 페이지의 식별자와 제목 17개를 읽기 전용으로 대조했다. 아래 제목은 arXiv 표기다. 학회판과 제목·저자가 다른 경우 구분한다. 논문 수치를 프로젝트 측정값으로 사용하지 않는다.

| arXiv | 원문 제목 |
|---|---|
| [1712.05877](https://arxiv.org/abs/1712.05877) | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference |
| [1806.08342](https://arxiv.org/abs/1806.08342) | Quantizing deep convolutional networks for efficient inference: A whitepaper |
| [2106.08295](https://arxiv.org/abs/2106.08295) | A White Paper on Neural Network Quantization |
| [2004.09602](https://arxiv.org/abs/2004.09602) | Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation |
| [1810.05723](https://arxiv.org/abs/1810.05723) | Post-training 4-bit quantization of convolution networks for rapid-deployment |
| [1909.13144](https://arxiv.org/abs/1909.13144) | Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks |
| [1805.06085](https://arxiv.org/abs/1805.06085) | PACT: Parameterized Clipping Activation for Quantized Neural Networks |
| [1902.08153](https://arxiv.org/abs/1902.08153) | Learned Step Size Quantization |
| [1308.3432](https://arxiv.org/abs/1308.3432) | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation |
| [1502.03167](https://arxiv.org/abs/1502.03167) | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift |
| [1906.04721](https://arxiv.org/abs/1906.04721) | Data-Free Quantization Through Weight Equalization and Bias Correction |
| [1710.03740](https://arxiv.org/abs/1710.03740) | Mixed Precision Training |
| [1503.02531](https://arxiv.org/abs/1503.02531) | Distilling the Knowledge in a Neural Network |
| [2007.13242](https://arxiv.org/abs/2007.13242) | WrapNet: Neural Net Inference with Ultra-Low-Resolution Arithmetic |
| [2308.13504](https://arxiv.org/abs/2308.13504) | A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance |
| [1801.04381](https://arxiv.org/abs/1801.04381) | MobileNetV2: Inverted Residuals and Linear Bottlenecks |
| [1904.01355](https://arxiv.org/abs/1904.01355) | FCOS: Fully Convolutional One-Stage Object Detection |

Banner 논문의 [NeurIPS 2019판](https://proceedings.neurips.cc/paper/2019/hash/c0a62e133894cdce435bcb4a5df1db2d-Abstract.html) 제목은 “Post training 4-bit quantization of convolutional networks for rapid-deployment”이며 저자는 Ron Banner, Yury Nahshan, Daniel Soudry다. arXiv판에는 Elad Hoffer도 포함된다. 기존 참고문헌은 학회판 표기였으므로 arXiv 링크와 판본 설명을 함께 추가했다.

WrapNet의 [ICLR 2021판](https://openreview.net/pdf?id=3SqrRe8FWQ-) 제목은 “WrapNet: Neural Net Inference with Ultra-Low-Precision Arithmetic”다. arXiv판의 “Ultra-Low-Resolution”과 구분해 참고문헌을 수정했다. 새로 연결한 WrapNet·A2Q의 arXiv 번호도 확인했다.

프레임워크 동작은 [torch.round](https://docs.pytorch.org/docs/stable/generated/torch.round.html), [MovingAverageMinMaxObserver](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html), [FakeQuantize](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html)의 공식 문서를 연결한다. 데이터셋 인용은 [별도 확인 기록](datasets.md)에 있다. FCOS와 Pascal VOC는 선택적 확장 참고자료이며 현재 실험에는 사용하지 않았다.
