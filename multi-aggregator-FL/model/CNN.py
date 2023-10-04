from torch import nn


class CNN_MNIST(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(3,28,28)
                in_channels=3,  # 输入图片的高度，因为mnist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=(5, 5),  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=(1, 1),  # 步长
                padding='same',  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding='same'
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batch_size
        x = x.view(x.size(0), -1)  # view就是把x弄成batch_size行个tensor
        output = self.out(x)
        return output


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 84),
            # nn.ReLU(),
            # nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
