from ultralytics import YOLO

model = YOLO('models\best.pt')  # load a custom trained model
# Export the model
model.export(format='onnx')