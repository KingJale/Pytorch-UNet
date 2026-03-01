""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):#继承父类
    """(convolution => [BN] => ReLU) * 2"""#卷积->批归一化->激活函数，进行了两次

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()#同步父类的属性
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(#子类直接使用父类中的Sequential方法来构建网络
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #in_channels(int):输入特征图的通道数（上一层输出的通道数）
            #mid_channels(int)输出特征图的通道数(即卷积核的数量)，决定了这一层提取多少不同的特征
            #kernel_size(int or tuple(元组)):卷积核大小，可以写成（3，3）或单个数字3
            #padding(int or tuple):在输入特征图周围填充的像素层数，padding=1表示上下左右各填充1行/列
            #bias(bool):是否使用偏置项(就是y=ax+b中的b)，默认为True。这里不使用偏置项的原因1后面会接BatchNorm层(本身有可学习的放缩和平移参数)
            nn.BatchNorm2d(mid_channels),
            #批归一化是一种深度神经网络训练技巧，通过对每一批（batch）数据进行规范化处理，加速网络训练、提高模型稳定性。
                                                    """
                                                    为什么要用批归一化？
                                                            问题：内部协变量偏移（Internal Covariate Shift）
                                                            网络每一层的输入分布会随着训练而不断变化
                                                            导致网络需要不断调整来适应新的数据分布
                                                            训练过程变得不稳定，收敛慢
                                                            解决方案：批归一化的作用
                                                            稳定训练：保持各层输入分布稳定
                                                            加速收敛：允许使用更大的学习率
                                                            防止梯度消失/爆炸：控制激活值的范围
                                                            轻微的正则化效果：引入小批量统计噪声
                                                    """
            nn.ReLU(inplace=True),#inplace=True表示原地操作，即 ReLU 函数直接修改输入张量本身，而不是创建一个新的张量。
            #以前的激活函数（如sigmoid、tanh）在输入值较大时梯度接近0
            #ReLU因其简单高效成为深度学习中最主流的激活函数。虽然存在"神经元死亡"等问题，但配合批归一化和合理的初始化，在大多数情况下表现优异。它是现代深度神经网络能训练得又快又好的关键技术之一。
            #Leaky ReLU​ - 解决"神经元死亡"问题
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    """nn.Sequential可以允许将多个模块封装成一个模块，
    forward函数接收输入之后，nn.Sequential按照内部模块的顺序自动依次计算并输出结果。
    nn.Sequential内部实现了forward函数，因此可以不用写forward函数。"""









class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
