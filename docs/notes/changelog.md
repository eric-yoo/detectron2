# 변경로그 (Change Log) 및 하위 호환성

### 릴리즈
새로운 업데이트 내역은
[https://github.com/facebookresearch/detectron2/releases](https://github.com/facebookresearch/detectron2/releases)의
릴리즈 로그에서 확인하십시오.

### 하위 호환성

이 라이브러리는 연구 목적으로 만들어졌기 때문에 하위 호환되지 않는 변경 사항이 있을 수 있습니다.
그러나 다음과 같은 방법으로 사용자의 방해를 줄이려고 노력합니다.
* 별도 각주가 없을 경우 [API 사용 가이드](https://detectron2.readthedocs.io/modules/index.html)의 API들은
  함수/클래스의 이름과 인수(argument), 문서화된 클래스 속성(attribute) 등이
  *stable*하다고 간주합니다.
  일반적으로 하위 호환성이 보장되지만, 만약 그렇지 않은 변화가 생길 경우 그 전에 적당한 기간동안
  deprecation 경고가 표시되고 릴리스 로그에 문서화될 것입니다.
* 그 외의 함수/클래스/속성은 내부용(internal)으로 간주되며 변경될 가능성이 더 높습니다.
  그러나 물론 이미 다른 프로젝트에서 이 중 일부를 사용하고 있을 수 있으며, 특히
  `detectron2/projects` 에서도 편의를 위해 사용하고 있을 수 있습니다.
  이러한 API의 경우 앞서 설명한 stable한 것으로 취급할 수도 있으며,
  준비가 되는 대로 stable API로 변경될 수 있습니다.
* "detectron2/projects" 아래에 있거나 "detectron2.projects"로 불러온(import) 프로젝트는 연구 프로젝트이기 때문에
  전부 experimental하다고 간주됩니다.
* 클래스/함수가 "default"라는 단어를 포함하거나 "default behavior" 방식으로 동작한다고
  문서에 명시된 경우에는 새 기능이 추가될 때 동작이 변경될 수 있습니다.

API 변경은 실제 코드 변경에 비해 빈도나 범위가 훨씬 작습니다.
따라서 하위 호환을 포기하고 detectron2의 최신 업데이트를 사용하려는 서드파티 프로젝트에서도,
detectron2를 포크(fork)하는 것보다 라이브러리 형태로 사용하는 것이 더 나을 것입니다.

[릴리스 로그](https://github.com/facebookresearch/detectron2/releases)에서 "incompatible changes"를 검색하면 이러한 변경 사항들을 확인할 수 있습니다.

### 환경설정 (Config) 버전 변경로그

Detectron2의 환경설정 버전은 오픈소스 공개 이후 변경된 적이 없습니다.
따라서 오픈소스 사용자는 이에 대해 걱정할 필요가 없습니다.

* v1: `RPN_HEAD.NAME` 에서 `RPN.HEAD_NAME` 로 이름 변경.
* v2: 릴리스 전 많은 환경변수 이름의 일괄 변경.

### 이전 버전의 회귀 버그(Silent Regression)

일부 회귀 버그는 발견하기 힘든 오류를 발생시켜 디버깅하기 어려울 수 있습니다. 아래는 몇 가지 사례입니다.

* 04/01/2020 - 05/11/2020: `TRAIN_ON_PRED_BOXES` 가 True로 세팅되었을 때 정확도(accuracy)가 낮음.
* 03/30/2020 - 04/01/2020: ResNet 계열 모델이 제대로 빌드되지 않음.
* 12/19/2019 - 12/26/2019: aspect ratio grouping을 사용하면 정확도가 떨어짐.
* - 11/9/2019: Test time augmentation을 사용하면 마지막 범주(category)를 예측하지 않음.
