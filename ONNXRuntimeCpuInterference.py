# !pip install onnxruntime

import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
dummy_input = np.random.rand(1, 10).astype(np.float32)
output = session.run([output_name], {input_name: dummy_input})
print("✅ Inference result:", output)
