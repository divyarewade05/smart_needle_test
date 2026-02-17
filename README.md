# Smart Needle 📍

**AI-Powered Face Recognition for Student Attendance. Local. Fast. Private.**

**Smart Needle** is a 100% local, privacy-first facial similarity engine that registers student identities, encodes them into deep face embeddings, and automatically detects who appears in any class or group photo — no cloud, no external APIs, no data leaving your machine.

---

## ✨ Key Features

- **🔒 Privacy First** — All face detection and embedding runs **locally on your device**. Your photos never leave your machine.
- **🧠 ArcFace Embeddings** — Powered by InsightFace's `buffalo_l` model for stable 512-dimensional face vectors with strong identity separation.
- **⚡ Smart Rebuild** — Hash-based change detection means only modified or new identity folders get reprocessed. Adding one student? Only that student gets recalculated.
- **👥 Many-to-Many Detection** — Upload a class photo and get back a structured report of which students are present, with confidence scores and bounding boxes.
- **🎚️ Threshold Slider** — Adjust match sensitivity in real time. Lower it for noisy group photos, raise it for stricter matching.
- **🖼️ Annotated Output** — Every result image is saved with bounding boxes and name labels drawn on detected faces.

---

## 🛠️ Tech Stack

- **Backend**: Python (FastAPI, InsightFace, OpenCV, Uvicorn)
- **Face Model**: ArcFace `buffalo_l` via ONNX Runtime (CPU & GPU)
- **Embedding Storage**: Smart Pickle (hash-tracked, incremental)
- **Frontend**: Single-file HTML/CSS/JS dashboard — no build step required

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9–3.11
- pip or conda

### 1. Clone the repo

```bash
git clone https://github.com/UnityAppSuite/Smart-Needle.git
cd Smart-Needle
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

pip install -r requirements.txt
```

> First run downloads the `buffalo_l` model (~500MB) automatically. Subsequent starts take ~10 seconds.

### 3. Configure your paths

Open `backend/app/core/config.py` and set your base path:

```python
_BACKEND = r"C:\path\to\Smart-Needle\backend"   # Windows
SIMILARITY_THRESHOLD = 0.30
```

### 4. Start the API

```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the UI

In a second terminal, from the project root:

```bash
python -m http.server 5500
```

Open **http://localhost:5500/smart-needle-ui.html** in Chrome or Edge.

---

## 📖 Usage Guide

1. **Add References** — Create one folder per student inside `backend/app/data/reference/`. Name each folder with the student's ID or name and put their photos inside.

2. **Build Embeddings** — Go to **Dashboard → ⚡ Smart Rebuild**. The engine processes only new or changed folders.

3. **Recognize Faces** — Go to **Recognize**, upload any scene image (noisy, multi-face, crowded), and see every detected face matched to an identity.

4. **Search by Person** — Go to **Search**, pick a student name, and find every image in your collection where that student appears.

---



