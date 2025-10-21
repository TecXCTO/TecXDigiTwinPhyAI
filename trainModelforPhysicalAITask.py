# -----------------------------------------------------
# Physical AI demo: Train & export a lightweight model
# Task: Predict whether a robot should MOVE or STOP
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx

# 1️⃣  Simulated dataset (e.g., distance & light sensor data)
#    distance (cm), light_intensity (0–1), and label (0=STOP, 1=MOVE)
np.random.seed(42)
num_samples = 500
distance = np.random.uniform(0, 100, num_samples)
light = np.random.uniform(0, 1, num_samples)

# Define a simple rule for labeling (the AI will learn this pattern)
labels = np.where((distance > 30) & (light > 0.3), 1, 0)

# Combine into input tensor
X = np.vstack((distance, light)).T.astype(np.float32)
y = labels.astype(np.int64)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# 2️⃣  Define a simple neural network
class PhysicalAIModel(nn.Module):
    def __init__(self):
        super(PhysicalAIModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)  # Output: STOP or MOVE

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = PhysicalAIModel()

# 3️⃣  Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4️⃣  Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5️⃣  Test the trained model
test_input = torch.tensor([[40.0, 0.8]])  # Example: safe distance, good light
pred = torch.argmax(model(test_input)).item()
print("Prediction (0=STOP, 1=MOVE):", pred)

# 6️⃣  Export model to ONNX (for TensorRT or TensorPy)
dummy_input = torch.randn(1, 2)
torch.onnx.export(
    model,
    dummy_input,
    "physical_ai_model.onnx",
    export_params=True,
    opset_version=12,
    input_names=["sensor_input"],
    output_names=["action_output"]
)

print("✅ Model trained and exported successfully → physical_ai_model.onnx")

# 7️⃣  Optional: Simulate physical decision
def robot_action(distance, light):
    data = torch.tensor([[distance, light]], dtype=torch.float32)
    pred = torch.argmax(model(data)).item()
    if pred == 1:
        print(f"🚗 MOVE (distance={distance:.1f}cm, light={light:.2f})")
    else:
        print(f"🛑 STOP (distance={distance:.1f}cm, light={light:.2f})")

# Test robot behavior
robot_action(80, 0.6)
robot_action(10, 0.2)
robot_action(45, 0.8)
