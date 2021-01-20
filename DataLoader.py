import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset): #Dataset 상속(괄호안에 적기)
    def __init__(self):
        # data loading
        # loadtxt 안에 skiprows=n 하면 n행이 사라짐
        xy=np.loadtxt('data-01-test-score.csv',delimiter=',', dtype=np.float32)
        # numpy에서 torch로 바꾸기
        self.x=torch.from_numpy(xy[:,:-1])
        self.y=torch.from_numpy(xy[:,[-1]])
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset=WineDataset()
first_data=dataset[0]
features, labels=first_data
print(features, labels)

dataloader=DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# num_workers: dataloading 빠르게 해줌

dataiter=iter(dataloader)
data=dataiter.next()
features, lables=data
print(features, lables)

# training loop
num_epochs=2
total_samples=len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if(i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
