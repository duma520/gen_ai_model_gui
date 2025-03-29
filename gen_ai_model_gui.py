# gen_ai_model_gui.py
__version__ = "2.0.3"

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
        self.root.title(f"AI模型生成工具 v{__version__}")
        self.root.geometry("500x600")
        
        # Create main container frames
        self.create_top_frame()
        self.create_middle_frame()
        self.create_bottom_frame()
        
        # Initialize logging
        self.init_logging()
    
    def create_top_frame(self):
        """Create the top section with device selection and model config"""
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side - Device selection
        device_frame = ttk.LabelFrame(top_frame, text="设备选择", padding=10)
        device_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        devices = [
            ("CPU", "cpu"),
            ("CUDA (NVIDIA GPU)", "cuda"),
            ("DirectML (Windows GPU)", "dml"),
            ("OpenCL (跨平台 GPU)", "opencl")
        ]
        
        self.device_var = tk.StringVar(value="cpu")
        for text, value in devices:
            rb = ttk.Radiobutton(device_frame, text=text, variable=self.device_var, value=value)
            rb.pack(anchor=tk.W)
        
        # Right side - Model configuration
        model_frame = ttk.LabelFrame(top_frame, text="模型配置", padding=10)
        model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create model config entries in a grid
        entries = [
            ("输入通道数:", "in_channels", "3"),
            ("输出通道数:", "out_channels", "2"),
            ("卷积核大小:", "kernel_size", "3"),
            ("填充大小:", "padding", "1")
        ]
        
        for i, (label_text, attr_name, default) in enumerate(entries):
            ttk.Label(model_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(model_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            setattr(self, attr_name, entry)
        
        # Configure grid weights
        model_frame.columnconfigure(1, weight=1)
    
    def create_middle_frame(self):
        """Create the middle section with ONNX export config"""
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side - Export file config
        export_file_frame = ttk.LabelFrame(middle_frame, text="文件配置", padding=10)
        export_file_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        file_entries = [
            ("输出文件名:", "filename", "ai_optimizer.onnx"),
            ("输入名称:", "input_names", "input"),
            ("输出名称:", "output_names", "output")
        ]
        
        for i, (label_text, attr_name, default) in enumerate(file_entries):
            ttk.Label(export_file_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(export_file_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            setattr(self, attr_name, entry)
        
        export_file_frame.columnconfigure(1, weight=1)
        
        # Right side - Export options
        export_opt_frame = ttk.LabelFrame(middle_frame, text="导出选项", padding=10)
        export_opt_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        self.dynamic_axes = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_opt_frame, text="启用动态batch轴", variable=self.dynamic_axes).pack(anchor=tk.W, pady=2)
        
        self.validate_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_opt_frame, text="导出后验证模型", variable=self.validate_model).pack(anchor=tk.W, pady=2)
    
    def create_bottom_frame(self):
        """Create the bottom section with logs and buttons"""
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(bottom_frame, text="日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame with progress bar
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(button_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        ttk.Button(button_frame, text="生成模型", command=self.generate_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除日志", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)
    
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
            try:
                in_channels = int(self.in_channels.get())
                out_channels = int(self.out_channels.get())
                kernel_size = int(self.kernel_size.get())
                padding = int(self.padding.get())
                if any(val <= 0 for val in [in_channels, out_channels, kernel_size, padding]):
                    raise ValueError("所有参数必须为正整数")
            except ValueError as e:
                self.log_message(f"无效参数: {str(e)}")
                messagebox.showerror("错误", f"无效参数: {str(e)}")
                return
            
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
            elif device == "cuda":
                if not torch.cuda.is_available():
                    self.log_message("警告: CUDA不可用, 回退到CPU")
                    device = "cpu"
                else:
                    self.log_message("使用CUDA设备")
            else:
                self.log_message("使用CPU设备")
            
            # 生成输入张量
            dummy_input = torch.randn(1, in_channels, 64, 64)
            if device == "cuda":
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
            
            # 修复路径问题：只有当路径包含目录时才创建目录
            output_dir = os.path.dirname(output_file)
            if output_dir:  # 只有当路径包含目录时才创建
                os.makedirs(output_dir, exist_ok=True)
            
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
