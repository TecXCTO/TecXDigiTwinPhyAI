# ---------------------------------------------
# Physical AI Model Training Example (TensorFlow)
# Task: Predict robot action (0=STOP, 1=MOVE)
# ---------------------------------------------

import tensorflow as tf
import numpy as np

# 1️⃣ Simulated sensor data
np.random.seed(42)
num_samples = 500
distance = np.random.uniform(0, 100, num_samples)
light = np.random.uniform(0, 1, num_samples)
labels = np.where((distance > 30) & (light > 0.3), 1, 0)

# Combine features
X = np.vstack((distance, light)).T.astype(np.float32)
y = labels.astype(np.int64)

# 2️⃣ Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Output: STOP/MOVE
])

# 3️⃣ Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4️⃣ Train model
model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# 5️⃣ Test model
test_input = np.array([[40.0, 0.8]], dtype=np.float32)
pred = np.argmax(model.predict(test_input))
print("Prediction (0=STOP, 1=MOVE):", pred)
