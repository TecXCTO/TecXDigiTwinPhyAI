import tensorpy as tp
import cv2
import numpy as np

# Load TensorRT engine
engine = tp.Engine("model_fp16.engine")

# Load camera feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_data = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Run inference
    output = engine.infer(input_data)

    # Interpret results (example: object detection)
    if np.argmax(output) == 1:
        print("⚙️ Detected object: Turning motor ON")
        # Send command to IoT device or motor controller
        # (e.g., AWS IoT MQTT publish)
