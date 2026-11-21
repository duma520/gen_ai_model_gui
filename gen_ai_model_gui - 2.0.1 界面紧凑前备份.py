import torch
import torch.nn as nn
from torch.onnx import export
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
from datetime import datetime
import os

class SimpleAIModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(x)

class ModelGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI模型生成工具")
        self.root.geometry("800x800")
        
        # 设备选择
        self.device_var = tk.StringVar(value="cpu")
        self.create_device_selection()
        
        # 模型配置
        self.create_model_config()
        
        # ONNX导出配置
        self.create_export_config()
        
        # 日志系统
        self.create_log_system()
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(pady=10)
        
        # 操作按钮
        self.create_action_buttons()
        
        # 初始化日志
        self.init_logging()
        
    def create_device_selection(self):
        frame = ttk.LabelFrame(self.root, text="设备选择", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        devices = [
            ("CPU", "cpu"),
            ("CUDA (NVIDIA GPU)", "cuda"),
            ("DirectML (Windows GPU)", "dml"),
            ("OpenCL (跨平台 GPU)", "opencl")
        ]
        
        for text, value in devices:
            rb = ttk.Radiobutton(frame, text=text, variable=self.device_var, value=value)
            rb.pack(anchor=tk.W)
    
    def create_model_config(self):
        frame = ttk.LabelFrame(self.root, text="模型配置", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 输入通道数
        ttk.Label(frame, text="输入通道数:").grid(row=0, column=0, sticky=tk.W)
        self.in_channels = ttk.Entry(frame)
        self.in_channels.insert(0, "3")
        self.in_channels.grid(row=0, column=1, sticky=tk.W)
        
        # 输出通道数
        ttk.Label(frame, text="输出通道数:").grid(row=1, column=0, sticky=tk.W)
        self.out_channels = ttk.Entry(frame)
        self.out_channels.insert(0, "2")
        self.out_channels.grid(row=1, column=1, sticky=tk.W)
        
        # 卷积核大小
        ttk.Label(frame, text="卷积核大小:").grid(row=2, column=0, sticky=tk.W)
        self.kernel_size = ttk.Entry(frame)
        self.kernel_size.insert(0, "3")
        self.kernel_size.grid(row=2, column=1, sticky=tk.W)
        
        # 填充大小
        ttk.Label(frame, text="填充大小:").grid(row=3, column=0, sticky=tk.W)
        self.padding = ttk.Entry(frame)
        self.padding.insert(0, "1")
        self.padding.grid(row=3, column=1, sticky=tk.W)
    
    def create_export_config(self):
        frame = ttk.LabelFrame(self.root, text="ONNX导出配置", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 输出文件名
        ttk.Label(frame, text="输出文件名:").grid(row=0, column=0, sticky=tk.W)
        self.filename = ttk.Entry(frame)
        self.filename.insert(0, "ai_optimizer.onnx")
        self.filename.grid(row=0, column=1, sticky=tk.W)
        
        # 输入名称
        ttk.Label(frame, text="输入名称:").grid(row=1, column=0, sticky=tk.W)
        self.input_names = ttk.Entry(frame)
        self.input_names.insert(0, "input")
        self.input_names.grid(row=1, column=1, sticky=tk.W)
        
        # 输出名称
        ttk.Label(frame, text="输出名称:").grid(row=2, column=0, sticky=tk.W)
        self.output_names = ttk.Entry(frame)
        self.output_names.insert(0, "output")
        self.output_names.grid(row=2, column=1, sticky=tk.W)
        
        # 动态轴
        self.dynamic_axes = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="启用动态batch轴", variable=self.dynamic_axes).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # 验证模型
        self.validate_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="导出后验证模型", variable=self.validate_model).grid(row=4, column=0, columnspan=2, sticky=tk.W)
    
    def create_log_system(self):
        frame = ttk.LabelFrame(self.root, text="日志", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def create_action_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        
        ttk.Button(frame, text="生成模型", command=self.generate_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="清除日志", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)
    
    def init_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("model_generator.log"),
                self.GUIHandler(self)
            ]
        )
        self.logger = logging.getLogger()
    
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)
        self.log_message("日志已清除")
    
    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update()
    
    def generate_model(self):
        try:
            self.log_message("开始生成模型...")
            self.update_progress(10)
            
            # 获取模型配置
            in_channels = int(self.in_channels.get())
            out_channels = int(self.out_channels.get())
            kernel_size = int(self.kernel_size.get())
            padding = int(self.padding.get())
            
            # 创建模型
            model = SimpleAIModel(in_channels, out_channels, kernel_size, padding)
            model.eval()
            self.log_message(f"模型创建成功: 输入通道={in_channels}, 输出通道={out_channels}, 卷积核={kernel_size}, 填充={padding}")
            self.update_progress(30)
            
            # 设置设备
            device = self.device_var.get()
            if device == "dml":
                try:
                    import torch_directml
                    device = torch_directml.device()
                    self.log_message("使用DirectML设备")
                except ImportError:
                    self.log_message("警告: torch-directml未安装, 回退到CPU")
                    device = "cpu"
            elif device == "opencl":
                try:
                    import torch_opencl
                    device = torch_opencl.device()
                    self.log_message("使用OpenCL设备")
                except ImportError:
                    self.log_message("警告: torch-opencl未安装, 回退到CPU")
                    device = "cpu"
            else:
                self.log_message(f"使用{device.upper()}设备")
            
            # 生成输入张量
            dummy_input = torch.randn(1, in_channels, 64, 64)
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
                dummy_input = dummy_input.cuda()
            elif device == "dml":
                model = model.to(device)
                dummy_input = dummy_input.to(device)
            
            self.update_progress(50)
            
            # 准备导出配置
            export_config = {
                "input_names": [self.input_names.get()],
                "output_names": [self.output_names.get()],
            }
            
            if self.dynamic_axes.get():
                export_config["dynamic_axes"] = {
                    self.input_names.get(): {0: "batch_size"},
                    self.output_names.get(): {0: "batch_size"}
                }
                self.log_message("启用动态batch轴")
            
            # 导出模型
            output_file = self.filename.get()
            self.log_message(f"开始导出ONNX模型到 {output_file}")
            
            export(model, dummy_input, output_file, **export_config)
            self.log_message("ONNX模型导出成功!")
            self.update_progress(80)
            
            # 验证模型
            if self.validate_model.get():
                self.log_message("开始验证模型...")
                try:
                    import onnx
                    model = onnx.load(output_file)
                    onnx.checker.check_model(model)
                    self.log_message("模型验证成功!")
                except Exception as e:
                    self.log_message(f"模型验证失败: {str(e)}")
            
            self.update_progress(100)
            messagebox.showinfo("成功", "模型生成并导出成功!")
        
        except Exception as e:
            self.log_message(f"错误: {str(e)}")
            messagebox.showerror("错误", f"模型生成失败: {str(e)}")
        finally:
            self.update_progress(0)
    
    class GUIHandler(logging.Handler):
        def __init__(self, app):
            super().__init__()
            self.app = app
        
        def emit(self, record):
            msg = self.format(record)
            self.app.log_message(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelGeneratorApp(root)
    root.mainloop()