""" Parts of the U-Net model """

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
            #mid_channels(int),中间的输出可以理解为瓶颈层输出特征图的通道数(即卷积核的数量)，决定了这一层提取多少不同的特征
            #kernel_size(int or tuple(元组)):卷积核大小，可以写成（3，3）或单个数字3
            #padding(int or tuple):在输入特征图周围填充的像素层数，padding=1表示上下左右各填充1行/列
            #bias(bool):是否使用偏置项(就是y=ax+b中的b)，默认为True。这里不使用偏置项的原因1后面会接BatchNorm层(本身有可学习的放缩和平移参数)

            nn.BatchNorm2d(mid_channels),
            #批归一化是一种深度神经网络训练技巧，通过对每一批（batch）数据进行规范化处理，加速网络训练、提高模型稳定性。
            
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






class Down(nn.Module):#这里把下采样和特征提取连接成了一个模块
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#使用最大池化层，尺寸减半
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


"""
线性插值是一种在两个已知数据点之间估算新数值的简单数学方法。其核心思想是“按比例分配”：假设你知道一条直线上两个端点的值，那么这条线上任何中间点的值，都与该点到两个端点的距离成反比关系。

双线性插值是一种常用的图像上采样方法，用于在放大图像时估算新像素点的颜色值。其核心思想是，目标图像中某个新像素点的值，由其原图像中最近的四个已知像素点（2x2邻域）通过两次线性插值（先水平，后垂直，或顺序相反）加权平均得到。

计算步骤简述：
1.  找到目标像素点映射回原图时的对应坐标（通常为非整数坐标）。
2.  确定该坐标周围最近的四个像素点。
3.  先在水平方向对上下两对点分别进行线性插值，得到两个中间值。
4.  再在垂直方向对这两个中间值进行线性插值，得到最终的目标像素值。

特点：
•   优点：计算相对简单，放大后的图像比最近邻插值更平滑，没有明显的块状锯齿。

•   缺点：会使得图像细节变得模糊，是一种低通滤波操作。
"""
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):#bilinear控制是否使用双线性插值法
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #mode='bilinear'使用双线性插值法
            #scale_factor表示放大两倍
            #align_coners控制对齐角点
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)#因为这里实际的输入变成了原来的2倍，所以上采样里面中间过程计算的特征图通道数要减半
        else:
            #转置卷积可以理解为普通卷积的“逆向”操作。通过 stride=2，它在输入的特征图元素间插入间隔（补零），然后通过一个普通的卷积操作（由 kernel_size定义）来生成更大的输出图。这个过程能学习如何从上采样中恢复细节，相比简单的插值法（如双线性插值）具有更强的表达能力。
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)#这里是转置卷积
            #kernel_size=2：卷积核的大小为 2×2。在转置卷积中，这决定了每个输入像素“贡献”到输出区域的局部大小。
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):#这里就是跳跃连接了
        x1 = self.up(x1)#为了把x1和x2的尺寸对齐，这里使用了上采样操作
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]#x2.size() 返回形状 (batch, channel, height, width)，索引 [2] 是高度 H，[3] 是宽度 W。
        #计算两个特征图的高度差 diffY 和宽度差 diffX。由于 x2 通常比上采样后的 x1 略大（例如奇数尺寸导致上采样后不能完全匹配），这两个差值一般为非负整数。

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])#PyTorch 的填充函数，pad 参数是一个列表 [左填充, 右填充, 上填充, 下填充]。
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)#在指定维度上拼接张量，这里 dim=1 表示在通道维度（channel）上拼接。
        #拼接后，新特征图的通道数为 x2 和 x1 的通道数之和。

        return self.conv(x)#通常是一个卷积模块，在 U-Net 中一般是两个卷积层（每个卷积后接 BN 和 ReLU），用于融合拼接后的特征，并可能调整通道数。

"""
在深度学习模型（如U-Net）中，DoubleConv 通常由两个连续的卷积层组成。这里的 in_channels // 2 指定了中间层的通道数，这是一种常见的瓶颈（bottleneck）设计，主要有两个目的：

1.  降低计算复杂度和参数量：第一个卷积将输入通道数 in_channels 压缩到一半（in_channels // 2），减少了后续卷积的输入通道数，从而显著降低了计算量和模型参数。
2.  保持非线性表达能力：尽管通道数被压缩，但通过两个卷积层和激活函数，模型仍能学习复杂的特征。第二个卷积再将通道数调整到目标输出 out_channels，完成特征变换。

这种设计在计算效率和特征提取能力之间取得了平衡，常见于需要轻量化或实时处理的网络结构中。需要注意的是，使用时要确保 in_channels 是偶数，以保证中间通道数为整数。
"""










class OutConv(nn.Module):#本质上就是一个卷积核1*1的简单的卷积层
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)