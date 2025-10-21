# -------------------------------------------
# Simple PyTorch → ONNX export example
# Works on CPU (Free-tier friendly)
# -------------------------------------------

import torch
import torch.nn as nn
import torch.onnx

# 1️⃣ Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 2)  # 2 output classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 2️⃣ Create model instance
model = SimpleModel()
model.eval()  # set to evaluation mode

# 3️⃣ Create dummy input (example data shape)
dummy_input = torch.randn(1, 10)  # batch size 1, 10 features

# 4️⃣ Run a forward pass (optional check)
output = model(dummy_input)
print("Model output:", output)

# 5️⃣ Export the model to ONNX format
torch.onnx.export(
    model,                      # model to be exported
    dummy_input,                # dummy input
    "simple_model.onnx",        # output filename
    export_params=True,         # store trained weights
    opset_version=12,           # ONNX version
    do_constant_folding=True,   # optimize constants
    input_names=["input"],      # input node name
    output_names=["output"],    # output node name
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ Model exported successfully to 'simple_model.onnx'")
