# Face Recognition & Detection System

Python pipeline for face detection and recognition using OpenCV. Haar Cascade for detection, LBPH for recognition.

**Quick start:** `pip install -r requirements.txt` → `python main.py`

---

## Two-Stage Pipeline

1. **Detection** — Haar Cascade locates face regions in each frame using `cv2.CascadeClassifier`.
2. **Recognition** — LBPH identifies each detected face using `cv2.face.LBPHFaceRecognizer`, outputting an ID and confidence value.

**Data flow:** Camera → Grayscale → Haar Cascade detects faces → LBPH predicts ID per face → Map ID to name via `config/users.json` → Display with label.

---

## Algorithms

### Haar Cascade

Pre-trained XML classifier. `detectMultiScale` slides a window over the image at different scales and applies a cascade of tests to reject non-face regions. Returns bounding boxes `(x, y, w, h)` for each face.

### LBPH (Local Binary Pattern Histogram)

Splits each face into cells, computes Local Binary Pattern per cell, builds histograms, and concatenates into one feature vector per face. At prediction: compares query to stored vectors, returns closest ID and distance. Lower distance = better match.

---

## Features

- **Enroll faces** — Collect training images from camera, train LBPH
- **Live face recognition** — Real-time identification from webcam
- **CLI menu** — 3 options: Enroll, Recognize, Exit
- **ID→name mapping** via `config/users.json`

Press `q` to exit camera windows.

---

## Project Structure

```
.
├── main.py              # CLI entry point
├── src/
│   ├── config.py        # Paths, thresholds
│   ├── collect.py       # Camera → face crops → dataset
│   ├── train.py         # Dataset → LBPH model
│   └── recognize.py     # Camera → detection → recognition
├── data/dataset/        # User.{id}.{count}.jpg
├── models/              # trainer.yml (LBPH)
├── config/users.json    # {"id": "name"}
└── assets/              # haarcascade_frontalface_default.xml
```

---

## Run

```bash
pip install -r requirements.txt
python main.py
```

**Prerequisites:** Python 3.6+, webcam. Grant camera permissions on macOS (System Preferences → Security & Privacy → Camera).

Run modules directly:
```bash
python -m src.collect
python -m src.train
python -m src.recognize
```

---

## Dependencies

- **opencv-contrib-python** — Haar cascade, LBPH, video capture
- **numpy** — Array operations
- **pillow** — Image loading in `train.py`
