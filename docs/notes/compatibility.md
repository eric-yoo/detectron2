# 다른 라이브러리와의 호환성

## Detectron(및 maskrcnn-benchmark)과의 호환성

Detectron2는 기존의 Detectron에 레거시처럼 남아있던 몇가지 이슈들을 개선했습니다. 이에 따라 모델들이
서로 호환되지 않습니다:
두 코드베이스에서 동일한 모델 가중치(weight)로 추론(inference)해도 결과가 다를 것입니다.

이러한 추론 결과는 아래 차이에서 기인합니다.

- (x1, y1)과 (x2, y2)를 좌표로 갖는 박스(box)의 높이와 너비를
  너비 = x2 - x1 및 높이 = y2 - y1로 계산합니다.
  기존 Detectron에서는 높이와 너비 모두에 "+ 1"이 더해졌습니다.

  참고로, Caffe2의 연산자는 추가 옵션을 통해
  [변경된 컨벤션을 지원](https://github.com/pytorch/pytorch/pull/20550)하고 있습니다.
  따라서 Detectron2 학습된 모델로 이전과 동일하게 Caffe2에서 추론할 수 있습니다.

  가장 눈에 띄는 변화는 높이/너비의 계산 방식 변경입니다.
  - 바운딩 박스 회귀(bounding box regression)의 인코딩/디코딩.
  - 비최대 억제(non-maximum suppression). 다만, 이로 인한 영향은 미미합니다.

- RPN은 이제 양자화 영향이 적은 더 단순한 앵커(anchor)를 사용합니다.

  Detectron에서는 앵커(anchor)가 양자화되어
  [정확한 영역이 없었습니다](https://github.com/facebookresearch/Detectron/issues/227).
  Detectron2의 앵커는 피처 좌표(feature grid points)에 맞춰 중앙 정렬되며 양자화되지 않습니다.

- Classification 계층은 클래스 레이블 순서가 다릅니다.

  여기에는 shape(..., num_categories + 1, ...)이 있는 모든 학습 가능한 매개변수가 포함됩니다.
  Detectron2에서 정수형 레이블 [0, K-1]은 K개(num_categories)의 object 분류를 의미하며, 
  레이블 "K"는 "배경"이라는 특수 분류를 의미합니다.
  Detectron에서는 레이블 "0"이 배경을 의미하고 레이블 [1, K]가 K개의 분류를 의미했습니다.

- ROIAlign 구현이 변경됐습니다. 새로운 구현체는 [Caffe2에서 확인할 수 있습니다](https://github.com/pytorch/pytorch/pull/23706).

  1. 이미지 피처 맵의 alignment 개선을 위해, 모든 ROI에서 Detectron의 절반만큼 픽셀을 이동합니다.
     자세한 내용은 `layers/roi_align.py` 를 참조하십시오.
     이전 Detectron의 결과를 얻으려면 `ROIAlignV2` (기본값) 대신 `ROIAlign(aligned=False)`
     또는 `POOLER_TYPE=ROIAlign` 을 사용하십시오.

  1. ROI의 최소값이 1일 필요가 없습니다.
     이로 인해 출력(output)에 작은 차이가 발생하지만 무시할 수 있습니다.

- 마스크 추론 함수(Mask inference function)가 다릅니다.

  Detectron2에서는 "paste_mask" 함수가 다르며 Detectron에서보다 높은 정확성이 요구됩니다. 이러한 변경은
  COCO의 mask AP를 ~0.5% absolute 만큼 향상시킬 수 있습니다.

학습에도 몇 가지 차이점이 있지만, 모델 간 호환성에는
영향을 미치지 않습니다. 주요 내용은 다음과 같습니다.

- 배치 단위가 아닌 이미지 단위로 `RPN.POST_NMS_TOPK_TRAIN` 을 만들어
  Detectron에 있던 [버그](https://github.com/facebookresearch/Detectron/issues/459)를 수정했습니다.
  이로 인해 키포인트 디텍션과 같은 일부 모델의 정확도가 조금 감소할 수 있으며
  기존 Detectron과 같은 결과를 얻으려면 일부 파라미터 튜닝이 필요합니다.
- 바운딩 박스 회귀의 기본 손실 함수를 smooth L1 loss에서 L1 loss로 간소화합니다.
  이로 인해 box AP50은 약간 감소하는 반면, 더 높은 중첩 임계값(overlap threshold)에 대한
  box AP이 개선되는 경향을 관찰했습니다. 이는 결국 전반적인 box AP를 개선합니다.
- COCO 바운딩 박스 및 세그멘테이션 어노테이션의 좌표는
  `[0, width]` 또는 `[0, height]` 범위의 좌표로 해석합니다. COCO 키포인트
  어노테이션의 좌표는 `[0, width - 1]` 또는 `[0, height - 1]` 범위의 픽셀 인덱스로 해석됩니다.
  이는 flip 증강(augmentation)의 구현 방식에 영향을 미칩니다.


[이 글](https://ppwwyyxx.com/blog/2021/Where-are-Pixels/)은
앞서 언급한 픽셀, 좌표, 그리고 "+1" 등의
문제에 대해 더 자세히 설명합니다.


## Caffe2와의 호환성

앞서 언급했듯이 연산자들이 Detectron과는 호환되지 않더라도
Caffe2에는 구현되어 있습니다.
따라서 detectron2로 학습시킨 모델을 Caffe2에서 변환할 수 있습니다.
[배포](../tutorials/deployment.md) 튜토리얼을 참조하십시오.

## TensorFlow와의 호환성

대부분의 연산자를 TensorFlow에서 사용할 수 있지만
크기 조정(resize) / ROIAlign / 패딩(padding)의 구현에 약간씩 차이가 있을 수 있습니다.
[tensorpack Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/convert_d2)에서 제공하는 변환 스크립트를 사용하면
TensorFlow에서 detectron2의 표준 모델을 실행할 수 있습니다.
