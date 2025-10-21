# ---------------------------------------------
# Physical AI Model Training Example (PyTorch)
# Task: Predict robot action (0=STOP, 1=MOVE)
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1️⃣ Simulated sensor data
np.random.seed(42)
num_samples = 500
distance = np.random.uniform(0, 100, num_samples)
light = np.random.uniform(0, 1, num_samples)
labels = np.where((distance > 30) & (light > 0.3), 1, 0)

# Convert to tensors
X = torch.tensor(np.vstack((distance, light)).T, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# 2️⃣ Define model
class PhysicalAIModel(nn.Module):
    def __init__(self):
        super(PhysicalAIModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)  # Output: STOP/MOVE

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = PhysicalAIModel()

# 3️⃣ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4️⃣ Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5️⃣ Test model
test_input = torch.tensor([[40.0, 0.8]])
pred = torch.argmax(model(test_input)).item()
print("Prediction (0=STOP, 1=MOVE):", pred)
