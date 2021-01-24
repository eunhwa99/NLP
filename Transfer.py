# ImageFolder
# Scheduler
# Transfer Learning

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # scheduler

import torchvision
from torchvision import datasets, models, transforms # pre-trained model
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
# rescale: 이미지의 크기 조절, randomCrop: 이미지를 무작위로 자른다 ==> data augmentation
# transforms.Compose: 샘플에 전이(trainsform) 적용
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

# 일부 이미지 시각화하기
def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# 학습 데이터의 배치를 얻기
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지 만들기
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# 모델 학습하기
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 epoch은 학습 단계와 검증 단계를 갖는다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # 학습 시에만 연산 기록 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 시에만 backward + optimize
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# 모든 layer를 fine tuning ==> 시간은 조금 더 걸리지만 조금 더 정확하다.
model=models.resnet18(pretrained=True)
num_features=model.fc.in_features

model.fc=nn.Linear(num_features, 2) # 클래스 2개
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.001)

# sheduler: update learning rate
# every 7 epochs, lr is multiplied by gamma
step_lr_scheduler=lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# for epoch in range(100):
 #   train() # optimizer.step()
 #   evaluate()
 #   scheduler.step()

model=train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=3)

# Freeze all the layers in the beginning ==> 시간은 조금 덜 걸리지만 조금 덜 정확하다.
model=models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad=False 

num_features=model.fc.in_features

model.fc=nn.Linear(num_features, 2) # 클래스 2개
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.001)

# sheduler: update learning rate

# every 7 epochs, lr is multiplied by gamma
step_lr_scheduler=lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# for epoch in range(100):
 #   train() # optimizer.step()
 #   evaluate()
 #   scheduler.step()

model=train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=3)



