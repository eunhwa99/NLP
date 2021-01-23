import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, output_size): #하이퍼 파라미터
        super(RNN, self).__init__()

        self.hidden_size=hidden_size
        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(hidden_size, output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        # 입력과 hidden state를 cat 함수로 붙여준다.
        # input으로 들어가는 것은 기존의 input 뿐만이 아니라. 이전 Hidden Layer의 결과 또한 들어간다.
        combined=torch.cat((input_tensor, hidden_tensor),dim=1)

        # 붙인 값을 i2h 및 i2o 에 통과시켜 hidden state를 업데이트하고 결과값을 계산
        hidden=self.i2h(combined)
        output=self.i2o(hidden)
        output=self.softmax(output)
        return output, hidden # input으로서 Hidden layer 값 또한 들어가므로, return 값은 output과 hidden layer의 output

    # 아직 입력이 없을 때(t=0)의 hidden state를 초기화 해준다.
    # 처음 Hidden Layer의 output 값이 없으므로 초기화 필요
    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

category_lines, all_categories=load_data()
n_categories=len(all_categories)

n_hidden=128
rnn=RNN(N_LETTERS, n_hidden,n_categories)

# one step
input_tensor=letter_to_tensor('A')
hidden_tensor=rnn.init_hidden()

output,next_hidden=rnn(input_tensor, hidden_tensor)

#whole sequence/name
input_tensor=line_to_tensor('Albert')
hidden_tensor=rnn.init_hidden()

output,next_hidden=rnn(input_tensor[0], hidden_tensor)


def category_from_output(output):
    category_idx=torch.argmax(output).item() #softmax 함수 사용하므로 가장 큰 값을 가지는 것이 정답
    return all_categories[category_idx]

criterion=nn.NLLLoss() #Negative Likelihood
learning_rate=0.005
optimizer=torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    # line_tensor: 이름
    # category_tensor: 클래스 라벨
    hidden=rnn.init_hidden()

    for i in range(line_tensor.size()[0]): # 이름의 길이
        output, hidden=rnn(line_tensor[i], hidden) # 현재 이름, 위에서 초기화한 hidden

    loss=criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss=0
all_losses=[]
plot_steps, print_steps=1000,5000
n_iters=100000

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    
    predict(sentence)