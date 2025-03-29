import torch
import torch.nn as nn
from torch.onnx import export

# 1. 定义一个超简单的模型（纯CPU）
class SimpleAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 仅用一层卷积 + 全局池化 + Sigmoid输出
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)  # 输入3通道，输出2个值（复杂度+运动强度）
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.sigmoid = nn.Sigmoid()  # 输出限制在0~1

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.sigmoid(x)

# 2. 创建模型并导出为ONNX
model = SimpleAIModel()
model.eval()  # 设置为评估模式

# 生成一个随机输入张量（模拟64x64的RGB图像）
dummy_input = torch.randn(1, 3, 64, 64)  # [batch, channels, height, width]

# 导出ONNX模型
export(
    model,
    dummy_input,
    "ai_optimizer.onnx",  # 输出文件名
    input_names=["input"],  # 输入名称
    output_names=["output"],  # 输出名称
    dynamic_axes={
        "input": {0: "batch_size"},  # 支持动态batch
        "output": {0: "batch_size"}
    }
)

print("✅ 成功生成 ai_optimizer.onnx！")