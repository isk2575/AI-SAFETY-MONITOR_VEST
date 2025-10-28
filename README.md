# 🦺 AI Safety Monitor — Person & Vest Detection

This project uses **YOLO11** models to detect **humans** and **safety vests** in wide-angle security camera footage.  
It can monitor predefined **zones** and automatically log events when people enter without wearing vests.

---

## 🚀 Features

- ✅ Person detection (YOLO11n)
- 🦺 Safety vest detection (custom-trained YOLO model)
- 🧭 Zone drawing & labeling (danger/safe zones)
- ⚡ Real-time FPS display
- 📝 Automatic JSONL event logging
- 📊 Decision system (`allow`, `escalate`, `block`) based on zone severity

---

## 🛠 Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- pathlib, json, uuid, datetime, math

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
