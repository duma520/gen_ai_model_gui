import torch
import torch.nn as nn
from torch.onnx import export
import tkinter as tk
from tkinter import ttk, messagebox
import os

class SimpleAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(x)

def generate_onnx():
    try:
        # 获取用户选择的设备
        device_type = device_var.get()
        
        # 设置计算设备
        if device_type == "CPU":
            device = torch.device("cpu")
        elif device_type == "CUDA" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_type == "DirectML":
            try:
                import torch_directml
                device = torch_directml.device()
            except ImportError:
                messagebox.showerror("错误", "未找到DirectML支持，请安装torch-directml包")
                return
        elif device_type == "OpenCL":
            device = torch.device("opencl")
        else:
            if device_type == "CUDA":
                messagebox.showwarning("警告", "CUDA不可用，将使用CPU代替")
            elif device_type == "OpenCL":
                messagebox.showwarning("警告", "OpenCL不可用，将使用CPU代替")
            device = torch.device("cpu")
        
        # 创建模型并转移到所选设备
        model = SimpleAIModel().to(device)
        model.eval()
        
        # 生成输入张量（同样转移到所选设备）
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        
        # 导出ONNX模型
        onnx_filename = "ai_optimizer.onnx"
        export(
            model,
            dummy_input,
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        
        # 检查文件是否生成成功
        if os.path.exists(onnx_filename):
            messagebox.showinfo("成功", f"✅ 成功生成 {onnx_filename}！\n使用的设备: {device}")
        else:
            messagebox.showerror("错误", "生成ONNX文件失败")
            
    except Exception as e:
        messagebox.showerror("错误", f"生成过程中出现错误:\n{str(e)}")

# 创建主窗口
root = tk.Tk()
root.title("AI模型导出工具")
root.geometry("400x350")

# 设置样式
style = ttk.Style()
style.configure("TLabel", font=("Microsoft YaHei", 10))
style.configure("TButton", font=("Microsoft YaHei", 10))
style.configure("TRadiobutton", font=("Microsoft YaHei", 10))

# 添加标题
title_label = ttk.Label(root, text="AI模型导出工具", font=("Microsoft YaHei", 12, "bold"))
title_label.pack(pady=10)

# 添加说明
info_label = ttk.Label(root, text="请选择计算设备并导出ONNX模型", wraplength=350)
info_label.pack(pady=5)

# 设备选择框架
device_frame = ttk.LabelFrame(root, text="选择计算设备", padding=10)
device_frame.pack(pady=10, padx=20, fill="x")

# 设备选择变量
device_var = tk.StringVar(value="CPU")

# 设备选项
devices = [
    ("CPU", "CPU"),
    ("CUDA (NVIDIA GPU)", "CUDA"),
    ("DirectML (Windows)", "DirectML"),
    ("OpenCL (跨平台GPU)", "OpenCL")
]

for text, mode in devices:
    rb = ttk.Radiobutton(
        device_frame,
        text=text,
        variable=device_var,
        value=mode
    )
    rb.pack(anchor="w", padx=5, pady=2)

# 导出按钮
export_btn = ttk.Button(
    root,
    text="导出ONNX模型",
    command=generate_onnx
)
export_btn.pack(pady=15)

# 运行主循环
root.mainloop()