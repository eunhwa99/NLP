# NLP
NLP study

### nn.Module
- nn은 Neural Network의 약자로, nn.Module은 모든 Neural Network Model의 Base Class이다. 모든 Neural Network Model(Net)은 nn.Module의 Subclass이다.
- nn.Module을 상속한 어떤 subclass가 Neural Network Model로 사용되려면 두개의 메서드를 override 해야 한다.

- __init__(self): Initialize. 여러분이 사용하고 싶은, Model에 사용될 구성 요소들을 정의 및 초기화한다. 대개 다음과 같이 사용된다.
self.conv1 = nn.Conv2d(1, 20, 5)  
self.conv2 = nn.Conv2d(20, 20, 5)  
self.linear1 = nn.Linear(1, 20, bias=True)  

- forward(self, x): Specify the connections. __init__에서 정의된 요소들을 잘 연결하여 모델을 구성한다. Nested Tree Structure가 될 수도 있다. 주로 다음처럼 사용된다.
x = F.relu(self.conv1(x))  
return F.relu(self.conv2(x))  

### Binary Classification
- 현재 딥러닝에서 분류에 대해 가장 흔히 사용되는 손실함수는 Cross Entropy Error (CEE) 이다.  

  ![image](https://user-images.githubusercontent.com/68810660/104878350-1c09ab80-599f-11eb-9bc5-fa6e96dd5f88.png)  
  t는 정답 값, y는 추론 값

- sigmoid: 이진분류에서 가장 많이 사용되는 활성화 함수
- softmax: 다중분류에서 가장 많이 사용되는 활성화 함수

- 이진분류를 CEE로 나타내면 L=-(tlog(y)+(1-t)log(1-y)), 참(1)/거짓(0)


## pytorch 함수
1. 랜덤넘버 생성함수
랜덤한 값을 가지는 텐서 생성
- torch.rand() : 0과 1 사이의 숫자를 균등하게 생성
- torch.rand_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
- torch.randn() : 평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
- torch.randn_like() :  사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
- torch.randint() : 주어진 범위 내의 정수를 균등하게 생성, 자료형은 torch.float32
- torch.randint_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
- torch.randperm() : 주어진 범위 내의 정수를 랜덤하게 생성

2. 텐서 생성함수-특정한 값을 가지는 텐서 생성
- torch.arange() : 주어진 범위 내의 정수를 순서대로 생성
- torch.ones() : 주어진 사이즈의 1로 이루어진 텐서 생성
- torch.zeros() : 주어진 사이즈의 0으로 이루어진 텐서 생성
- torch.ones_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
- torch.zeros_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
- torch.linspace() : 시작점과 끝점을 주어진 갯수만큼 균등하게 나눈 간격점을 행벡터로 출력
- torch.logspace() : 시작점과 끝점을 주어진 갯수만큼 로그간격으로 나눈 간격점을 행벡터로 출력
