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
- torch.from_numpy(): numpy배열을 torch tesor로 바꾸기

3. Dataset and DataLoader 
- 코드: DataLoader.py, Load dataset.ipynb
- epoch = 1 forward and backward pass of ALL training samples
- batch_size = number of training samples in one forward & backward pass
- iterations: number of passess, each pass using [batch_size] number of samples
  ex) 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
  ### Custom DataLoader
 우리가 직접 만드는 custom dataloader은 다음과 같은 세 파트로 이루어져 있다.
 ####  __ init __ (self) : download, read data 등을 하는 파트
 #### __ getitem __ (self, index) : 인덱스에 해당하는 아이템을 넘겨주는 파트, 어떠한 인덱스 idx를 받았을 때, 그에 상응하는 입출력 데이터 반환
 #### __ len __ (self) : data size를 넘겨주는 파트, 이 dataset의 총 데이터 개수

### Basic Approach to DNN
1. NN 모델 생성
2. 훈련 및 overfitting 확인
- overfitting 되지 않았으면 더 깊고 크게 모델을 만든다.
- overfitting 되었다면, drop-out이나 batch-normalization과 같은 regularizaiton을 행한다.
- overfitting이 되었는지 확인하는 방법?: Validation Loss가 증가하면 Overfitting
3. 2번부터 반복한다.
