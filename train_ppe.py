from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolo11n.pt')

# Train
results = model.train(
    data='C:/Users/tepea/OneDrive/Documents/GitHub/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch =8,
    name='ppe_detector',
    patience=15,
    device='cpu',
    workers=2,
    project='runs/detect'
)

print("\n✅ Training complete!")
print("📁 Best model: runs/detect/ppe_detector/weights/best.pt")