# 🚀 AI Sign Language AAC & Iron Man Avatar Generator

This project consists of two powerful components: a **Sign Language Communication App** for accessibility and a **Real-time Iron Man Avatar Generator** for video processing.

## 📂 Project Structure

```
├── app.py                          # Main GUI Application (Sign-to-Speech & Video)
├── logic.py                        # Backend logic for AI translation & video stitching
├── requirements.txt                # Python dependencies
├── avatars/
│   ├── face_generator_ironman_body.py  # Iron Man Avatar Generation Script
│   ├── inswapper_128.onnx          # Face swap model (Required)
│   └── real_face_expression/       # Source images for face swapping
├── model/
│   └── keypoint_classifier/        # TFLite/Keras models for hand gesture recognition
├── dataset/                        # Folder containing ASL video clips (WLASL)
└── assets/                         # Images and resources
```

---

## 1️⃣ AI Sign Language AAC (`app.py`)

A comprehensive accessibility tool built with **PyQt6** that bridges the gap between sign language and spoken English.

### ✨ Features
- **📷 Real-time Sign Detection**: Uses MediaPipe & a custom Keypoint Classifier to detect hand signs via webcam.
- **🗣️ Text-to-Speech (TTS)**: Converts typed or detected text into spoken audio using `pyttsx3`.
- **🤖 AI Grammar Correction**: Uses **Ollama (Llama 3 / Qwen 2.5)** to fix broken sign language gloss into proper English sentences.
- **🎥 Text-to-Sign Video**: Translates English text into ASL Gloss and stitches together video clips to demonstrate the signs.
- **⌨️ Predictive Typing**: Includes a virtual keyboard with word suggestions.

### 🚀 How to Run
1. Ensure your webcam is connected.
2. Run the application:
   ```bash
   python app.py
   ```

---

## 2️⃣ Iron Man Avatar Generator (`face_generator_ironman_body.py`)

A sophisticated video processing script that transforms a human subject into an **Iron Man** avatar using procedural drawing and AI face swapping.

### ✨ Features
- **🦾 Procedural Armor**: Draws the Mark 85 armor dynamically on the user's body using MediaPipe Pose landmarks.
- **🎭 AI Face Swapping**: Uses **InsightFace** (`inswapper_128.onnx`) to swap the user's face with a target identity (e.g., Tony Stark).
- **✨ Visual Effects**: Includes glow effects, repulsors, and metallic rendering.
- **🎩 Accessories**: Supports overlaying hats or other headgear.

### 🚀 How to Run
1. Place your input videos in `./dataset/test_wlasl` (or configure `INPUT_FOLDER`).
2. Run the generator:
   ```bash
   python avatars/face_generator_ironman_body.py
   ```
3. Output videos will be saved to `./dataset/custom`.

---

## 🛠️ Installation & Setup

### 1. Install Python Dependencies
Make sure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Install Spacy Model
Required for NLP and text processing.
```bash
python -m spacy download en_core_web_sm
```

### 3. Setup Ollama (For AI Grammar & Translation)
The app uses a local LLM for privacy and speed.
1. Download and install [Ollama](https://ollama.com/).
2. Pull the required model (default is `qwen2.5:7b` or `llama3.2`):
   ```bash
   ollama pull qwen2.5:7b
   ```
   *(Note: You can change the model name in `app.py` variable `MODEL_NAME`)*

### 4. Download Required Models
Ensure the following models are placed in their respective folders:
- **Face Swap Model**: `avatars/inswapper_128.onnx` (Download from InsightFace or HuggingFace).
- **Gesture Classifier**: `model/keypoint_classifier/keypoint_classifier.tflite`.

---

## ⚙️ Configuration

### `app.py` Settings
You can modify these global variables at the top of `app.py`:
- `LLAMA_URL`: URL for the Ollama API (default: `http://localhost:11434/api/generate`).
- `MODEL_NAME`: The LLM model to use.
- `TRIGGER_CHAR`: The character that triggers gesture recognition.

### `face_generator_ironman_body.py` Settings
- `INPUT_FOLDER`: Directory of videos to process.
- `ASSET_HEAD_FOLDER`: Folder containing images of the face you want to swap onto the body.
- `FACE_PUSH_DOWN_AMOUNT`: Adjusts the vertical position of the face/neck.

---

## ⚠️ Troubleshooting

- **"ImportError: DLL load failed"**: You might be missing Visual C++ Redistributables or have a conflict with `opencv-python` and `opencv-contrib-python`. Try uninstalling both and installing only `opencv-python`.
- **"Model missing"**: Ensure `inswapper_128.onnx` is in the `avatars/` folder.
- **Ollama Connection Error**: Make sure the Ollama app is running in the background (`ollama serve`).
