import torch
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn as nn
import ResNet101
import Dataset_train
import Dataset_test

BATCH_SIZE = 3
LEARNING_RATE = 1e-3
EPOCH = 1

torch.cuda.empty_cache()
data = Dataset_train.CTData('C:/Users/Amax/Desktop/maxillary sinus/train(001-700)')  #导入数据
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)  #创建加载器
model = ResNet101.ResNet101().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss()


def train(network, train_loader, optimizer):
    network.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        prediction = network(data)
        label = label.cuda()
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCH):
    train(model, dataloader, optimizer)

test_data = Dataset_test.CTData('C:/Users/Amax/Desktop/maxillary sinus/test(701-800)')
testloader = DataLoader(test_data, batch_size=1, shuffle=False)  # 创建加载器
sum = torch.tensor(0).cuda()
sum2 = torch.tensor(0).cuda()

MSE = 0
MAE = 0
with torch.no_grad():
    for (data, label) in testloader:
        y = model(data)
        label = label.cuda()
        MSE = MSE + (label - y) ** 2
        MAE = MAE + abs(label - y)

MSE = MSE/200
MAE = MAE/200
print('MSE:', MSE)
print('MAE:', MAE)


