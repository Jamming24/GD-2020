# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Conv_Interoperation.py 
@time: 2020年12月13日22时03分 
"""

# 1.用卷积神经网络进行训练
# 2.使用TF-IDF直接计算得分


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import pickle as pk
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
import math
import torchsnooper
from torch.utils.data import TensorDataset

warnings.filterwarnings('ignore')


def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap='gray')


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionBasicBlock(nn.Module):

    def __init__(self, pool_features=2):
        super().__init__()
        self.branch1x1 = BasicConv2d(1, 4, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(1, 2, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(2, 4, kernel_size=4, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(1, 4, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(4, 4, kernel_size=2, padding=1)

        self.branch_pool = BasicConv2d(1, pool_features, kernel_size=1)
        self.fc1 = nn.Linear(36000, 2560)
        self.fc2 = nn.Linear(2560, 2)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(outputs, 1)
        print("output")
        print(outputs)
        x = outputs.view(-1, 36000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# @torchsnooper.snoop()
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1080, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=4, padding=0)
        x = F.relu(x)
        # view操作时将二维张量转换为一维向量，以便于线性层处理
        x = x.view(-1, 1080)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        # print('output.dtype: ', output.dtype)
        # print('target.dtype: ', target.dtype)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = running_correct.__float__() / len(data_loader.dataset)
    print(f'第{epoch}次迭代, {phase} loss is {loss} and {phase} accuracy is {accuracy}')
    return loss, accuracy


# torch.cuda.is_available()
# BERT lcqmc train
# max_row = 39
# max_col = 48
# test
# max_row = 33
# max_col = 36

train_path = "E:/2020-GD/Semantic Interoperation/train/lcqmc_train_Elmo_interaction.pk"
test_path = "E:/2020-GD/Semantic Interoperation/test/lcqmc_test_Elmo_interaction.pk"
train_data = pk.load(open(train_path, mode='rb'))
test_data = pk.load(open(test_path, mode='rb'))
train_x_data = []
train_y_data = []
max_row = 40
max_col = 50
import numpy as np
for data in train_data:
    x = torch.tensor(data[0], dtype=torch.float)
    left = (max_col - x.shape[1]) // 2
    right = math.ceil((max_col - x.shape[1]) / 2)
    up = (max_row - x.shape[0]) // 2
    down = math.ceil((max_row - x.shape[0]) / 2)
    m = torch.nn.ZeroPad2d(padding=(left, right, up, down))
    matrix = m(x)
    matrix = torch.reshape(matrix, (1, 40, 50))
    train_x_data.append(matrix)
    train_y_data.append(int(data[1]))
train_x_data = [t.numpy() for t in train_x_data]
train_x_data = torch.from_numpy(np.array(train_x_data))
train_x_data = torch.tensor(train_x_data, dtype=torch.float32)
train_y_data = torch.tensor(train_y_data, dtype=torch.int64)


test_x_data = []
test_y_data = []
for data in test_data:
    x = torch.tensor(data[0], dtype=torch.float)
    left = (max_col - x.shape[1]) // 2
    right = math.ceil((max_col - x.shape[1]) / 2)
    up = (max_row - x.shape[0]) // 2
    down = math.ceil((max_row - x.shape[0]) / 2)
    m = torch.nn.ZeroPad2d(padding=(left, right, up, down))
    matrix = m(x)
    matrix = torch.reshape(matrix, (1, 40, 50))
    test_x_data.append(matrix)
    test_y_data.append(int(data[1]))
test_x_data = [t.numpy() for t in test_x_data]
test_x_data = torch.from_numpy(np.array(test_x_data))
test_x_data = torch.tensor(test_x_data, dtype=torch.float32)
test_y_data = torch.tensor(test_y_data, dtype=torch.int64)
test_dataset = TensorDataset(test_x_data, test_y_data)

train_dataset = TensorDataset(train_x_data, train_y_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

model = Net()
# model = InceptionBasicBlock()
model.cuda()
optimizer = optimizer.SGD(model.parameters(), lr=0.01)
data, target = next(iter(train_loader))
output = model(Variable(data.cuda()))
print(output.size())
print(target.size())
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 30):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

# D:\Anaconda3\envs\Pytorch_Learning\python.exe D:/PycharmWorkSpace/GD_2020/Identification/Conv_Interoperation.py
# torch.Size([32, 2])
# torch.Size([32])
# 第1次迭代, training loss is 0.43178733764996025 and training accuracy is 0.7947823391940226
# 第1次迭代, validation loss is 0.5423424869577108 and validation accuracy is 0.739930522955591
# 第2次迭代, training loss is 0.36114093766829525 and training accuracy is 0.840236884648568
# 第2次迭代, validation loss is 0.48849063257176756 and validation accuracy is 0.7660313585578818
# 第3次迭代, training loss is 0.34814572222053103 and training accuracy is 0.8460542958377658
# 第3次迭代, validation loss is 0.5128196311460371 and validation accuracy is 0.751525678340062
# 第4次迭代, training loss is 0.33963572452240937 and training accuracy is 0.849027918547867
# 第4次迭代, validation loss is 0.55284921145419 and validation accuracy is 0.7411041216787156
# 第5次迭代, training loss is 0.33347953629833416 and training accuracy is 0.8522989035289782
# 第5次迭代, validation loss is 0.544478643467679 and validation accuracy is 0.7403999624448409
# 第6次迭代, training loss is 0.3286701916258341 and training accuracy is 0.8541919703810426
# 第6次迭代, validation loss is 0.6086057080157922 and validation accuracy is 0.7112008262135011
# 第7次迭代, training loss is 0.32477820184525114 and training accuracy is 0.8566588207701265
# 第7次迭代, validation loss is 0.5540066108984764 and validation accuracy is 0.7486620974556379
# 第8次迭代, training loss is 0.3204047149064652 and training accuracy is 0.8589958369282059
# 第8次迭代, validation loss is 0.5348163411019834 and validation accuracy is 0.7477232184771383
# 第9次迭代, training loss is 0.31510282089038355 and training accuracy is 0.8616679091662968
# 第9次迭代, validation loss is 0.4897172371553889 and validation accuracy is 0.7644822082433574
# 第10次迭代, training loss is 0.3102884447926203 and training accuracy is 0.8640342427313772
# 第10次迭代, validation loss is 0.4747890695676301 and validation accuracy is 0.7781428973805277
# 第11次迭代, training loss is 0.30573701355753136 and training accuracy is 0.8669241014214755
# 第11次迭代, validation loss is 0.47385702244206424 and validation accuracy is 0.7777673457891278
# 第12次迭代, training loss is 0.30171999932244126 and training accuracy is 0.8694370220215608
# 第12次迭代, validation loss is 0.46757440888907426 and validation accuracy is 0.7859355929020749
# 第13次迭代, training loss is 0.29859483584336116 and training accuracy is 0.8710536676076158
# 第13次迭代, validation loss is 0.4851401351156531 and validation accuracy is 0.7776265139423528
# 第14次迭代, training loss is 0.29585844361714075 and training accuracy is 0.8721384116666527
# 第14次迭代, validation loss is 0.6308123327602121 and validation accuracy is 0.7246267956060464
# 第15次迭代, training loss is 0.29318441271469375 and training accuracy is 0.8738974560867125
# 第15次迭代, validation loss is 0.4808146327932657 and validation accuracy is 0.7815228617031265
# 第16次迭代, training loss is 0.2901970119817741 and training accuracy is 0.8743790992017288
# 第16次迭代, validation loss is 0.5761144612050596 and validation accuracy is 0.7434513191249648
# 第17次迭代, training loss is 0.2888004479553213 and training accuracy is 0.8766114103348048
# 第17次迭代, validation loss is 0.45515074612176337 and validation accuracy is 0.7933057928832974
# 第18次迭代, training loss is 0.2863669665722934 and training accuracy is 0.8768878316008142
# 第18次迭代, validation loss is 0.5096800614943382 and validation accuracy is 0.7605389165336588
# 第19次迭代, training loss is 0.2851439858531782 and training accuracy is 0.8779851402628515
# 第19次迭代, validation loss is 0.4443867704183571 and validation accuracy is 0.8051826119613182

# -2
# 第1次迭代, training loss is 0.6033034464042869 and training accuracy is 0.6706398733488017
# 第1次迭代, validation loss is 0.6938647298496795 and validation accuracy is 0.5595249272368792
# 第2次迭代, training loss is 0.539320194642237 and training accuracy is 0.7291490413207911
# 第2次迭代, validation loss is 0.7253017947994121 and validation accuracy is 0.5892404469063938
# 第3次迭代, training loss is 0.48560594876317015 and training accuracy is 0.7732382332492901
# 第3次迭代, validation loss is 0.6167075628854223 and validation accuracy is 0.682846681062811
# 第4次迭代, training loss is 0.4585178630739835 and training accuracy is 0.7891869026578323
# 第4次迭代, validation loss is 0.5928565254458552 and validation accuracy is 0.7038306262322787
# 第5次迭代, training loss is 0.4410547256214099 and training accuracy is 0.7991590092391714
# 第5次迭代, validation loss is 0.6320212892652868 and validation accuracy is 0.682940568960661
# 第6次迭代, training loss is 0.42778692576560956 and training accuracy is 0.8070118861144384
# 第6次迭代, validation loss is 0.5485989989030978 and validation accuracy is 0.7320908834851187
# 第7次迭代, training loss is 0.41774672517410383 and training accuracy is 0.8118325054656023
# 第7次迭代, validation loss is 0.5651794153920318 and validation accuracy is 0.7195099051732232
# 第8次迭代, training loss is 0.410501918220119 and training accuracy is 0.8160416474707454
# 第8次迭代, validation loss is 0.5657731606561627 and validation accuracy is 0.7202610083560229
# 第9次迭代, training loss is 0.4055221694146239 and training accuracy is 0.8196351239288676
# 第9次迭代, validation loss is 0.5501507695611458 and validation accuracy is 0.7273964885926204
# 第10次迭代, training loss is 0.4007166955156757 and training accuracy is 0.8224914770109647
# 第10次迭代, validation loss is 0.5644328921918589 and validation accuracy is 0.7255187306356211
# 第11次迭代, training loss is 0.3968836941465864 and training accuracy is 0.8236934906980056
# 第11次迭代, validation loss is 0.539320176917615 and validation accuracy is 0.7381466528964417
# 第12次迭代, training loss is 0.392290223111426 and training accuracy is 0.8265247145741018
# 第12次迭代, validation loss is 0.5816410301257922 and validation accuracy is 0.7187588019904234
# 第13次迭代, training loss is 0.38929597186499826 and training accuracy is 0.8284010286221656
# 第13次迭代, validation loss is 0.6193584243676638 and validation accuracy is 0.6884330109848841
# 第14次迭代, training loss is 0.3857345005852865 and training accuracy is 0.8312406289002622
# 第14次迭代, validation loss is 0.5408821854498698 and validation accuracy is 0.7361280630926673
# 第15次迭代, training loss is 0.3835059866489778 and training accuracy is 0.8312406289002622
# 第15次迭代, validation loss is 0.5618698276182753 and validation accuracy is 0.7220918223640973
# 第16次迭代, training loss is 0.3809539131959936 and training accuracy is 0.83205732809529
# 第16次迭代, validation loss is 0.538651886663239 and validation accuracy is 0.7328419866679186
# 第17次迭代, training loss is 0.37854416301012944 and training accuracy is 0.8341933106053626
# 第17次迭代, validation loss is 0.560899814813084 and validation accuracy is 0.737113886020092
# 第18次迭代, training loss is 0.37669554270704886 and training accuracy is 0.8350225744033908
# 第18次迭代, validation loss is 0.558779400958107 and validation accuracy is 0.7292742465496198
# 第19次迭代, training loss is 0.37412080351133253 and training accuracy is 0.8364214335374384
# 第19次迭代, validation loss is 0.5378181726403577 and validation accuracy is 0.7370669420711671