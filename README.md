# WICWIU(What I can Create, What I Understand)

* WICWIU는 국내 대학 최초로 공개하는 딥러닝 오픈소스 프레임워크입니다.

* WICWIU("What I Create, What I Understand.")라는 이름은 Richard Feynman의 "What I cannot create, I do not understand."에서 영감을 받아 지었습니다. "우리가 직접 만들며 이해한 딥러닝 프레임워크"라는 뜻입니다.

* WICWIU는 모든 API가 C++로 제공되어 메모리 및 성능 최적화에 유리합니다. 또한, 응용시스템의 개발 외에도 프레임워크 자체를 특수한 환경에 맞도록 수정 및 확장이 가능합니다.

* Custom Operator, 또는 Module 개발을 위한 API documentation은 추후 올리겠습니다.

* WICWIU는 Apache2.0 라이선스를 적용해 연구 목적 및 상용 목적으로 제약 없이 활용 가능합니다.

## [WICWIU Github Pages](https://wicwiu.github.io)

**[WICWIU Github Pages](https://wicwiu.github.io) 가 새로 개설되었습니다!** [이곳](https://wicwiu.github.io)을 클릭하여 방문해주세요.

- 기존 [**WICWIU**](https://github.com/WICWIU/WICWIU) 의 문서 `doc/` 와 `Documentation/` 가 모두 포팅되어 웹브라우저로 편하게 볼 수 있습니다. [**WICWIU**](https://github.com/WICWIU/WICWIU) 의 **API** 도 모두 포팅이 되었습니다.

- **[WICWIU Github Pages](https://wicwiu.github.io)** 에는 [**WICWIU**](https://github.com/WICWIU/WICWIU) 개발 가이드([**WICWIU**](https://github.com/WICWIU/WICWIU) 로 신경망 구성, [**WICWIU**](https://github.com/WICWIU/WICWIU) 로 딥러닝하기, [**WICWIU**](https://github.com/WICWIU/WICWIU) 코드 같이 분석해보기) 가 있습니다.

- **[WICWIU Github Pages](https://wicwiu.github.io)** 에는 [**WICWIU**](https://github.com/WICWIU/WICWIU) 개발에 필요한 **Git, Github** 으로 협업하기, **GPU Acceleration(CUDA 코딩하기)**, **Makefile** 이해하기, **딥러닝에 필요한 CLI(gotop, tmux, zsh)** 알아보기 등등 좋은 정보들이 지속적으로 업로드되고 있습니다.

## WICWIU는 다음과 같은 내용을 지향하고 있습니다.

* 다양한 신경망 계층 예시

* 일반적인 그래프 형태의 네트워크 구조

* 가독성 높은 저수준 연산의 CPU 코드

* 높은 성능의 GPU 병렬 연산 (cuDNN)

* 학습을 위한 한국어 문서 (준비중; 현재 한국어 주석 포함)

* MNIST, CIFAR10, ImageNet tutorial 지원

## WICWIU는 다음과 같은 환경을 지원하고 있습니다.

| System |
| --- |
| Linux CPU |
| Linux GPU |
| Windows CPU |


## WICWIU는 다음과 같은 요소들로 구성되어 있습니다.

<table>
<tr>
    <td><b> Tensor & Shape </b></td>
    <td> Tensor 클래스는 최대 5차원의 Tensor Data를 저장, 관리하며 모든 신경망 연산은 Tensor에 대해 수행합니다. Tensor의 내부에는 모든 데이터를 1차원 배열의 형태로 저장하지만, Shape 클래스를 이용하여 외부로는 최대 5차원까지의 인터페이스를 제공하고 있습니다. </td>
</tr>
<tr>
    <td><b> Operator </b></td>
    <td> Operator 클래스는 순전파와 역전파를 수행하는 저수준 연산을 포함하며, 각 연산의 결과 값을 각 객체의 멤버 변수로 저장하고 있는 클래스입니다. 저장된 결과는 연결된 다른 Operator나 Loss Function의 피연산자로 사용 가능합니다. 또한, 사용자는 Operator 클래스를 상속받아 새로운 연산자를 정의할 수 있습니다. </td>
</tr>
<tr>
    <td><b> Module </b></td>
    <td> Module 클래스는 복잡한 신경망 모델을 Operator 클래스만을 이용하여 구현하는 것이 불편하여 만들어진 고수준 연산 클래스입니다. 복수의 Operator들을 그래프 구조로 조합하여 정의하며, Operator와 다른 Module와 재귀적 구조로 구성 가능합니다. </td>
</tr>
<tr>
    <td><b> Loss Function & Optimizer  </b></td>
    <td> Loss Function 클래스는 손실 함수를 표현하고, Optimizer 클래스는 경사도 벡터를 이용하여 파라미터를 최적화 하는 알고리즘을 표현하는 클래스입니다. WICWIU에서는 다양한 Loss Function과 Optimizer를 제공하고 있으며, 사용자가 직접 새로운 알고리즘을 정의 가능합니다. </td>
</tr>
<tr> <td><b> Neural Network </b></td>
    <td> Neural Network 클래스는 신경망 모델을 표현하기 위한 클래스입니다. Operator와 Module를 조합하여 신경망 모델을 구성하고 모델 학습의 전반적인 기능을 제공하고 있습니다. </td>
</tr>
</table>


## WICWIU를 사용하기 위해서는 다음 패키지들이 설치되어 있어야 합니다.
* NVIDIA driver
* cuda toolkit (tested on v.9.0)
* cuDNN (tested on v7.0.5)

## WICWIU는 다음과 같은 방법으로 설치하실 수 있습니다.

```bash
$ git clone https://github.com/WICWIU/WICWIU
$ cd WICWIU/tutorials/MNIST
$ make
$ ./main
```
## CPU 동작을 원하시면 다음과 같은 방법으로 실행할 수 있습니다. (처음 받으시면 GPU로 환경설정이 되어 있습니다.)
```bash
$ cd WICWIU/
* makefile에서 ENABLE_CUDNN = -D__CUDNN__ 주석 처리
$ make
$ cd tutorials/MNIST
$ make
$ ./main
```
자세한 사용방법은 예제파일을 참고하여 주시고, 추후 자료를 보강하도록 하겠습니다.


## WICWIU를 만들어가는 사람들을 소개합니다.
> WICWIU는 한동대학교 학부생 주도로 개발되었습니다.

* 한동대학교 전산전자공학부 김인중 교수님.
* 1기: 박천명(팀장), 김지웅, 기윤호, 김지현
* 2기: 최은서(팀장), 윤성결, 김승주, 오준석
* 3기: 권예성(팀장), 윤동휘, 김경협, 전혜원
* 4기: 신치호(팀장), 박참진, 오상진, 서진명, 오준석
* 5기: 선한결(팀장), 한찬솔, 강예빈, 최하영
* 6기: 오준석(팀장), 이찬효, 우옥균, 이세현
* 7기: 선한결(팀장), 이현규, 안현재, 유수민, , 이요한

## 저희팀과 연락하고 싶으신 분들은?
프레임워크에 대해 궁금한 점이 있다면 hgudeeplearning@gmail.com 로 연락하시길 바랍니다.
