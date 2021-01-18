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
- sigmoid: 이진분류에서 가장 많이 사용되는 활성화 함수
- softmax: 다중분류에서 가장 많이 사용되는 활성화 함수
