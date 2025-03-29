import onnxruntime as ort
import numpy as np

# 加载模型
sess = ort.InferenceSession("ai_optimizer.onnx")

# 创建一个随机输入 (1, 3, 64, 64)
input_data = np.random.rand(1, 3, 64, 64).astype(np.float32)

# 运行推理
outputs = sess.run(None, {"input": input_data})
print("输出结果 (复杂度, 运动强度):", outputs[0])