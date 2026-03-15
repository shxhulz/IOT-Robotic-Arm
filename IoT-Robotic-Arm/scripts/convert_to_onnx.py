from ultralytics import YOLO

model_path = "D:\\RoboticArm\\IOT-Robotic-Arm\\IoT-Robotic-Arm\\scripts\\best.pt"
output_path = "D:\\RoboticArm\\IOT-Robotic-Arm\\IoT-Robotic-Arm\\scripts\\best.onnx"

model = YOLO(model_path)
model.export(format="onnx")
