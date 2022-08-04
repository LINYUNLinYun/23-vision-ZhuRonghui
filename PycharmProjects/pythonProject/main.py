#构建一个类线性模型类，继承自nn.Module,nn.m中封装了许多方法
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset    #这是一个抽象类，无法实例化
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

#构建一个compose类的实例，包含转tensor（张量），后面那个是标准化，两个参数分别是均值和标准差
train_transform = transforms.Compose([
                                transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),
                                transforms.RandomRotation((-10,10)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

#这2个值也是调参重灾区……
train_batch_size = 256
learning_rate = 0.06
test_batch_size = 100
random_seed = 2         # 随机种子，设置后可以得到稳定的随机数
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) #为gpu提供随机数


train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=False, transform=train_transform)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=False, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,batch_size=train_batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=test_batch_size,shuffle=False,pin_memory=True)


class ResidualBlock(nn.Module):
    # Residual Block需要保证输出和输入通道数x一样
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # 3*3卷积核，保证图像大小不变将padding设为1
        # 第一个卷积
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        # 第二个卷积
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # 激活
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        # 先求和 后激活
        z = self.conv3(x)
        return F.relu(z + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 192, kernel_size=5, padding=2)

        # 残差神经网络层，其中已经包含了relu
        self.rblock1 = ResidualBlock(32)
        self.rblock2 = ResidualBlock(64)
        self.rblock3 = ResidualBlock(128)
        self.rblock4 = ResidualBlock(192)

        # BN层，归一化，使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(192)

        # 最大池化，一般最大池化效果都比平均池化好些
        self.mp = nn.MaxPool2d(2)

        # fully connectected全连接层
        self.fc1 = nn.Linear(192 * 7 * 7, 256)  # 线性
        self.fc6 = nn.Linear(256, 10)  # 线性

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)  # channels:1-32 w*h:28*28

        x = F.relu(x)
        x = self.bn1(x)
        x = self.rblock1(x)

        x = self.conv2(x)  # channels:32-64    w*h:28*28
        x = F.relu(x)
        x = self.bn2(x)
        x = self.rblock2(x)

        x = self.mp(x)  # 最大池化,channels:64-64  w*h:28*28->14*14
        x = self.drop1(x)

        x = self.conv3(x)  # channels:64-128   w*h:14*14

        x = F.relu(x)
        x = self.bn3(x)
        x = self.rblock3(x)
        # x = self.mp(x)      #channels:128-128  w*h:7*7
        #x = self.drop1(x)

        x = self.conv4(x)  # channels:128-192  w*h:14*14

        x = F.relu(x)
        x = self.bn4(x)
        x = self.rblock4(x)
        x = self.mp(x)  # 最大池化,channels:192-192    w*h:14*14->7*7

        x = x.view(in_size, -1)  # 展开成向量
        x = F.relu(self.fc1(x))  # 使用relu函数来激活
        #x = self.drop2(x)

        return self.fc6(x)


model = Net()

#调用GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True       #启用cudnn底层算法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
torch.cuda.empty_cache()        #释放显存

model.to(device)

# #构建损失函数
criterion = torch.nn.CrossEntropyLoss()      #交叉熵
#
# #构建优化器,参数1：模型权重，参数二，learning rate
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.5)    #带动量0.5

#optimizer = optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.99,momentum = 0.5)
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=0.05,
#                              betas=(0.9, 0.999),
#                              eps=1e-08,
#                              weight_decay=0,
#                              amsgrad=False)
#设置学习率梯度下降，如果连续三个epoch测试准确率没有上升，则降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)



#把训练封装成一个函数
def train(epoch):
    running_loss =0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        #forward,backward,update
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            train_loss_val.append((running_loss/300))
            #print('[%d,%5d] loss:%3f'%(epoch+1,batch_idx+1,running_loss/300))
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

#把测试封装成函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #torch.max()返回的是两个值，第一个是用_接受(蚌埠住了),第二个是pre
            _, predicted = torch.max(outputs.data,dim=1)       #从第一维度开始搜索
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print('Accuracy on test set: %f %% [%d/%d]' % (100 * correct / total, correct, total))

    return correct/total
    #print('Accuracy on test set:%d %%'%(100*correct/total))


train_epoch = []
model_accuracy = []
temp_acc = 0.0
train_loss_val = []
for epoch in range(30):
    train(epoch)
    acc = test()

    print(epoch + 1,acc)
    train_epoch.append(epoch)
    model_accuracy.append(acc)
    scheduler.step(acc)

plt.figure(1)
plt.plot(train_epoch, model_accuracy)  # 传入列表，plt类用来画图
plt.grid(linestyle=':')
plt.ylabel('accuracy')  # 定义y坐标轴的名字
plt.xlabel('epoch')  # 定义x坐标
plt.show()  # 显示

plt.figure(2)
plt.plot(train_epoch*3, train_loss_val)  # 传入列表，plt类用来画图
plt.grid()
plt.ylabel('train_loss')  # 定义y坐标轴的名字
plt.xlabel('epoch')  # 定义x坐标
plt.show()  # 显示


# class Model(torch.nn.Module):
#     #构造函数
#     def __init__(self):
#         super(Model,self).__init__()      #调用父类的构造函数
#         self.linear1 = torch.nn.Linear(8,6)      #torch.linear类，8->6维
#         self.linear2 = torch.nn.Linear(6, 4)  # torch.linear类，6->4维
#         self.linear3 = torch.nn.Linear(4,2)
#         self.linear4 = torch.nn.Linear(2, 1)  # torch.linear类，2->1维
#         self.sigmoid = torch.nn.Sigmoid()       #将其看作网络的一层
#
#
#     def forward(self,x):                        #重写前馈函数
#         x = self.sigmoid(self.linear1(x))
#         x = self.sigmoid(self.linear2(x))
#         x = self.sigmoid(self.linear3(x))
#         x = self.sigmoid(self.linear4(x))       #实际上就是y hat
#         return x
#
# model = Model()                           #实例化类
#
# class DiabetesDataset(Dataset):
#     def __init__(self,filepath):
#         xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
#         self.len = xy.shape[0]      #取得数据集的行数
#         self.x_data = torch.from_numpy(xy[:,:-1])    # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
#         self.y_data = torch.from_numpy(xy[:,[-1]])   # [-1] 最后得到的是个矩阵
#     def __getitem__(self, index):
#         return self.x_data[index],self.y_data[index]        #返回索引值的代表的元素（元组来的）
#     def __len__(self):
#         return self.len
#
# dataset = DiabetesDataset('C:/Users/86132/Desktop/minist/archive/diabetes.csv')
# #实例化一个dataloader类，参数：#1，数据集  #2，batch大小  #3是否打乱  #4线程数
# train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
#
# # #构建损失函数
# criterion = torch.nn.BCELoss(reduction='mean')      #求均值,求和不求的区别是学习率上的影响
# #
# # #构建优化器,参数1：模型权重，参数二，learning rate
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
# #
# e_val = []
# l_val = []
#
# if __name__ == '__main__':
#     #print(dataset.len)
#     for epoch in range(100):
#         for i,data in enumerate(train_loader,0):
#             inputs,labels = data
#             y_pred = model(inputs)
#             loss = criterion(y_pred,labels)
#             #l_val.append(loss.item())
#             #e_val.append(epoch)
#             print(loss,i,loss.item())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#x_test = torch.Tensor([[4.0]])
#y_test = model(x_test)
#print('y_pred = ',y_test.data)

# plt.plot(e_val, l_val)  # 传入列表，plt类用来画图
# plt.ylabel('loss')  # 定义y坐标轴的名字
# plt.xlabel('epoch')  # 定义x坐标
# plt.show()  # 显示

# import torch
#
# x_data = [1.0,2.0,3.0]
# y_data = [2.0,4.0,6.0]  #列表
#
# w1 = torch.Tensor([1.0])     #此时的w已经被承载，不再是一个常数，而是一个tensor
# w1.requires_grad = True      #表明需要计算梯度
# w2 = torch.Tensor([1.0])
# w2.requires_grad = True
# b = torch.Tensor([1.0])
# b.requires_grad = True
#
#
# #m模型设计
# def forward(x):
#     return w1*x**2 + w2*x +b            #此时的*已经被重载，实际上进行的是矩阵等的数乘
#
# def loss(x,y):
#     y_pred = forward(x)
#     return (y_pred-y)**2
#
# print('predict(before training)',4,forward(4).item())
#
# for epoch in range(10000):            #训练100次
#     for x,y in zip(x_data,y_data):
#         l = loss(x,y)               #l不是常数
#         l.backward()                #释放计算图，反向传播,计算loss的偏导数
#         print("\tgrad: ",x,y,w1.grad.item(),w2.grad.item(),b.grad.item())     #使用item和data是为了防止生成图，所以都采用标量的形式进行操作
#         #权重更新
#         w1.data -= 0.01*w1.grad.data
#         w1.grad.data.zero_()          #给权重的梯度清零
#         w2.data -= 0.01 * w2.grad.data
#         w2.grad.data.zero_()  # 给权重的梯度清零
#         b.data -= 0.01 * b.grad.data
#         b.grad.data.zero_()  # 给权重的梯度清零
#
#         print("progress:",epoch,l.item())
# print(w1.data,w2.data,b.data)
# print('predict(after training)',4,forward(4).item())
# import numpy as np
# import matplotlib.pyplot as plt
#
# x_data = [1.0,2.0,3.0]
# y_data = [2.0,4.0,6.0]  #列表还是字典
#
# w = 1.0
#
# def forward(x):     #前馈函数
#     return x * w    #w待定
#
# def cost(xs,ys):         #计算方差
#     cost = 0
#     for x,y in zip(xs,ys):
#         y_pred = forward(x)
#         cost+=(y_pred-y)**2
#     return cost
#
# def gradient(xs,ys):         #梯度算法,计算方差对x的偏导数,其实就是贪心算法，每一次都朝着梯度下降的地方走
#     grad = 0
#     for x,y in zip(xs,ys):
#         #y_pred = forward(x)
#         grad+=2*x*(w*x-y)
#     return grad/len(xs)
#
# print('predict(before training)',4,forward(4))
#
# empty_epoch = []
# empty_cost = []
#
# for epoch in range(100):            #训练100次
#     empty_epoch.append(epoch)
#
#     cost_val = cost(x_data,y_data)      #方差的值
#     empty_cost.append(cost_val)
#     grad_val = gradient(x_data,y_data)  #梯度的值
#     w -= 0.01*grad_val              #学习率是0.01
#     print('epoch = ',epoch, 'w = ',w,'loss = ',cost_val,'grad = ',grad_val)
#
# print('predict(after training)',4,forward(4))
# # w_list = []     #创建一个空列表
# # mse_list = []
# #
# # for w in np.arange(0.0,4.0,0.1):     #头，尾，步长
# #     print('w = ',w)
# #     l_sum = 0
# #     for x_val,y_val in zip(x_data,y_data):       #循环
# #         y_pred_val = forward(x_val)         #predict y
# #         loss_val = loss(x_val,y_val)
# #         l_sum+=loss_val         #calculate loss sum
# #         print('\t',x_val,y_val,y_pred_val,loss_val)
# #     print('mse=',l_sum/3)
# #     w_list.append(w)            #append w to the empty list
# #     mse_list.append(l_sum/3)    #append the average loss to the empty list
# #
# plt.plot(empty_epoch,empty_cost)       #传入列表，plt类用来画图
# plt.ylabel('cost')             #定义y坐标轴的名字
# plt.xlabel('epoch')                 #定义x坐标
# plt.show()                      #显示