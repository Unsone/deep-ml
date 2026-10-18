import torch
import torch.nn.functional as F
from typing import Tuple

def train_simple_cnn_with_backprop(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
    kernel_size: int = 3,
    num_filters: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # ==========================================
    # 【第一部分：数据解析与形状推导】
    # ==========================================
    # n_samples: 样本总数（比如100张图片）。
    n_samples, height, width = X.shape
    
    # num_classes: 分类的类别总数（比如识别0-9就是10类）。
    # 来源于标签 y 的第二维度。
    num_classes = y.shape[1]

    # out_h, out_w: 卷积后的输出高度和宽度。
    # 含义：卷积核在图片上能滑动的次数。比如 28-3+1=26。
    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1
    
    # flattened_size: 展平后的尺寸。全连接层只能处理一维向量，必须把特征图拍扁。
    # 公式 = 输出高 * 输出宽 * 卷积核数量(num_filters)。
    flattened_size = out_h * out_w * num_filters

    # ==========================================
    # 【第二部分：参数初始化】
    # ==========================================
    # W_conv: 卷积核（权重）。
    # 形状 (num_filters, 1, kernel_size, kernel_size) 的四个维度含义：
    # num_filters: 输出通道数，即有几个不同的“特征探测器”。
    # 1: 输入通道数(in_channels)。因为题目是灰度图（1层），所以卷积核厚度必须是1。这个维度的信息是描述了权重W_conv看待输入信息的不同层次的不同方式，输入信息有多少层次，权重张量就要以对应数量层次的看待方法。
    # kernel_size: 卷积核的高和宽，决定了它每次读取的局部范围大小。
    W_conv = (torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.01).requires_grad_(True)
    
    # b_conv: 卷积偏置。每个卷积核配一个，加在乘积结果上，改变激活门槛。
    b_conv = torch.zeros(num_filters, requires_grad=True)

    # W_dense: 全连接层（Dense Layer）权重。
    # 作用：把提取到的高维特征组合起来，映射到最终的类别空间。
    W_dense = (torch.randn(flattened_size, num_classes) * 0.01).requires_grad_(True)
    
    # b_dense: 全连接层偏置。给每个类别的打分加一个基础分。
    b_dense = torch.zeros(num_classes, requires_grad=True)

    # optimizer: 优化器对象。负责根据梯度更新参数。
    # 注意：把需要更新的四个参数打包成列表 [] 传给优化器。
    optimizer = torch.optim.SGD([W_conv, b_conv, W_dense, b_dense], lr=learning_rate)


    # ==========================================
    # 【第三部分：训练大循环】
    # ==========================================
    for epoch in range(epochs):
        for i in range(n_samples):
            
            optimizer.zero_grad()

            # X[i] 原形状是 (H, W)。
            # unsqueeze(0) 两次 -> 变成 (1, 1, H, W)：[batch大小, 输入通道数, 高, 宽]。
            # .float() 确保数据类型为32位浮点型。
            x_i = X[i].unsqueeze(0).unsqueeze(0).float() 
            
            # y[i] 原形状是 (num_classes,)，取出后变 (1, num_classes)。
            y_i = y[i].unsqueeze(0).float()

            # ----------------------------------------
            # 【第四部分：前向传播（Forward Pass）】
            # 目标：让数据流过网络，算出预测结果并计算误差。
            # ----------------------------------------
            
            # 1. 卷积层（用 unfold 替代 conv2d，避免测试环境硬件不支持的警告）。
            # F.unfold(x_i, kernel_size): 按照卷积核大小，把图像切分成一个个小方块，并在通道维度上展平。
            # patches 形状变为 (1, k*k, out_h*out_w)。
            patches = F.unfold(x_i, kernel_size)            
            
            # 2. 卷积矩阵乘法（替代真正的卷积相乘累加）。
            # W_conv.view(num_filters, -1): 把卷积核拉平成 (num_filters, 9)。
            # @ patches[0]: 矩阵乘法，一次性算完所有滑窗的乘加。
            # Z_conv 得到 (num_filters, out_h*out_w)。
            Z_conv = W_conv.view(num_filters, -1) @ patches[0]   
            
            # 3. 加上偏置 b_conv，并调整形状为 (num_filters, out_h*out_w)。
            Z_conv = Z_conv + b_conv.view(-1, 1)
            
            # 4. ReLU 激活函数：将所有小于0的值变成0，引入非线性。
            # 5. flatten展平：把 (num_filters, out_h*out_w) 拍扁成一条长向量 (1, flattened_size)。
            ReLU_flat = F.relu(Z_conv).reshape(1, -1)       

            # 6. 全连接层（线性运算）：这里就是标准的 W*X + b 矩阵乘法。
            # 形状：(1, flattened_size) @ (flattened_size, num_classes) -> (1, num_classes)。
            Z_dense = ReLU_flat @ W_dense + b_dense
            
            # ----------------------------------------
            # 【第五部分：计算损失（Calculate Loss）】
            # ----------------------------------------
            # y_i.argmax(dim=1): 把独热编码(如[0,1,0])变成真实索引(如1)。
            # F.cross_entropy: 内部包含 Softmax + 交叉熵计算。
            loss = F.cross_entropy(Z_dense, y_i.argmax(dim=1))
            
            # ----------------------------------------
            # 【第六部分：反向传播与更新（Backward & Update）】
            # ----------------------------------------
            loss.backward()
            optimizer.step()

    # ==========================================
    # 【第七部分：返回结果】
    # ==========================================
    # .detach(): 剥离计算图（停止跟踪梯度，节省内存）。
    # 直接返回 Tensor，因为测试脚本期望看到 torch.Size 的形状。
    return (
        W_conv.detach(),
        b_conv.detach(),
        W_dense.detach(),
        b_dense.detach()
    )