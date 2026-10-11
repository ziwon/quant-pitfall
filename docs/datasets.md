# 데이터셋 출처와 라이선스 확인

이 프로젝트는 CIFAR-10만 사용한다. 데이터 원본은 `data/`에 따로 내려받으며 저장소와 공개 수치 기록에 포함하지 않는다. VOC·FCOS 확장은 구현하지 않았다.

| 항목 | 출처·확인 상태 |
|---|---|
| 배포자 | Alex Krizhevsky, Vinod Nair, Geoffrey Hinton; [공식 CIFAR 배포 페이지](https://cave.cs.toronto.edu/kriz/cifar.html) |
| 구성 | 32×32 컬러 이미지, 클래스 10개, 학습 50,000장·테스트 10,000장. [공식 설명](https://cave.cs.toronto.edu/kriz/cifar.html)과 실행 전 데이터 검사를 기준으로 한다. |
| 다운로드 | 공식 Python 버전 `cifar-10-python.tar.gz`, MD5 `c58f30108f718f92721af3b95e74349a`. [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)의 다운로드·무결성 검사를 사용한다. |
| 인용 | A. Krizhevsky, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 기술 보고서, 2009. 공식 배포 페이지가 이 보고서 인용을 요청한다. |
| 데이터 라이선스 | **공식 배포 페이지에서 별도의 명시적 데이터 라이선스를 확인하지 못했다** (2026-10-10 확인). MIT·Apache-2.0 또는 다른 라이선스로 추정해 표기하지 않는다. 이 저장소의 코드 라이선스와 구분하며 원본 이미지 재배포 권한을 부여한다고 주장하지 않는다. |

학습·평가의 구분, 시드와 부분집합 설정은 [학습 가이드 6절](README.md#6-종단-실험-cifar-10--mobilenetv2), [학습 코드](../src/quant_pitfalls/training.py), [수치 기록](results/manifest.json)에 있다. 학습 비교 표는 테스트 10,000장, 정수 누산기 sweep은 같은 첫 1,000장, 선택한 16·32비트 포화 평가만 전체 10,000장을 사용한다.

Pascal VOC는 참고문헌과 확장 실습 제안에만 등장한다. 확장할 때 사용 연도·원본 배포처·해당 데이터의 이용 조건을 별도로 확인한다.
