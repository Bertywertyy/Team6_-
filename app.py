import sys
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import threading
import requests
import time
import pyttsx3
import queue
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QTabWidget, QGroupBox, QSizePolicy, 
    QCheckBox, QFrame, QProgressBar
)
from PyQt6.QtCore import (
    Qt, QThread, QObject, pyqtSignal, pyqtSlot, QUrl, QTimer
)
from PyQt6.QtGui import QFont, QImage, QPixmap, QKeyEvent, QAction

# Multimedia Imports
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

import qdarktheme

# ============================================================================
# ⚙️ GLOBAL CONFIGURATION
# ============================================================================
OUTPUT_FOLDER = "./output"
DATASET_FOLDER = "./dataset"
LLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  
TRIGGER_CHAR = "Z"
HOLD_TIME = 0.6

# ============================================================================
# 🔧 IMPORT & SYNC
# ============================================================================
try:
    import logic  
    from utils.cvfpscalc import CvFpsCalc
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
    logic.MODEL_NAME = MODEL_NAME
except ImportError as e:
    print(f"❌ Import Error: {e}")

# ============================================================================
# 🧠 AI PREDICTION WORKER
# ============================================================================
class PredictionWorker(QThread):
    suggestions_ready = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.current_text = ""
        self.running = True

    def update_text(self, text):
        self.current_text = text.strip()
        if not self.isRunning():
            self.start()

    def run(self):
        time.sleep(0.3) 
        text = self.current_text
        if not text:
            self.suggestions_ready.emit([])
            return

        prompt = (
            f"Complete the word or predict the next word for: '{text}'. "
            "Output strictly 3 words separated by commas. No explanations."
        )

        try:
            response = requests.post(LLAMA_URL, json={
                "model": MODEL_NAME, "prompt": prompt, "stream": False, 
                "options": {"temperature": 0.2} 
            }, timeout=3)
            
            if response.status_code == 200:
                raw = response.json().get('response', '').strip()
                clean = raw.replace('"', '').replace('[', '').replace(']', '').replace('.', '')
                words = [w.strip() for w in clean.split(',')]
                self.suggestions_ready.emit(words[:3])
            else:
                self.suggestions_ready.emit([])
        except:
            self.suggestions_ready.emit([])

# ============================================================================
# 🔊 ROBUST SPEAKER ENGINE (Crash Fix)
# ============================================================================
class TTSWorker(QThread):
    """
    Dedicated thread for Text-to-Speech to prevent UI freezes and Crashes.
    Uses a Queue system to process speech requests sequentially.
    """
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.start()

    def speak(self, text):
        self.queue.put(text)

    def run(self):
        # Initialize engine ONCE inside the thread
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS Init Error: {e}")
            return

        while self.running:
            try:
                text = self.queue.get() # Blocks until text is available
                if text is None: break
                
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except Exception as e:
                print(f"TTS Error: {e}")

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.wait()

# ============================================================================
# 📸 CAMERA WORKER (With Cursor Logic)
# ============================================================================
class CameraWorker(QThread):
    image_update = pyqtSignal(QImage)
    text_update = pyqtSignal(str)     
    status_update = pyqtSignal(str) 
    sentence_finalized = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self.running = True
        self.sentence_buffer = []
        self.cursor_pos = 0 # ✅ Tracks where you are typing
        self.labels = []
        self.load_labels()
        
    def load_labels(self):
        path = "model/keypoint_classifier/keypoint_classifier_label.csv"
        if os.path.exists(path):
            import csv
            with open(path, encoding="utf-8-sig") as f:
                self.labels = [row[0] for row in csv.reader(f)]

    def run(self):
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        keypoint_classifier = KeyPointClassifier()

        last_char = ""
        char_start = time.time()
        char_processed = False

        while self.running:
            ret, frame = cap.read()
            if not ret: break

            frame = cv.flip(frame, 1)
            debug_image = copy.deepcopy(frame)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            current_char = ""
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                    landmarks = self.calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed = self.pre_process_landmark(landmarks)
                    
                    idx = keypoint_classifier(pre_processed)
                    if idx < len(self.labels):
                        current_char = self.labels[idx]
                    
                    # VISUALS
                    color = (0, 255, 128) # Spring Green
                    cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), color, 2)
                    cv.putText(debug_image, current_char, (brect[0], brect[1]-10), 
                             cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
                    self.draw_landmarks(debug_image, landmarks)
            
            # LOGIC
            if current_char:
                if current_char == last_char:
                    duration = time.time() - char_start
                    # Progress Bar
                    progress = min(1.0, duration/HOLD_TIME)
                    bar_w = int(progress * 200)
                    col = (0, 255, 255) if progress < 1.0 else (0, 200, 0)
                    cv.rectangle(debug_image, (20, 440), (20+bar_w, 450), col, -1)

                    if duration > HOLD_TIME and not char_processed:
                        if current_char == TRIGGER_CHAR:
                            self.finalize_sentence()
                        else:
                            # ✅ Insert at Cursor
                            self.sentence_buffer.insert(self.cursor_pos, current_char)
                            self.cursor_pos += 1
                            self.emit_text_update()
                        char_processed = True
                else:
                    last_char = current_char
                    char_start = time.time()
                    char_processed = False
            else:
                last_char = ""
                char_start = time.time()

            h, w, ch = debug_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(debug_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_update.emit(qt_image)

        cap.release()

    def emit_text_update(self):
        display = list(self.sentence_buffer)
        # ✅ Show Cursor Visual
        if 0 <= self.cursor_pos <= len(display):
            display.insert(self.cursor_pos, "|")
        self.text_update.emit("".join(display))

    def finalize_sentence(self):
        raw_text = "".join(self.sentence_buffer).replace("  ", " ")
        if not raw_text: return
        
        self.status_update.emit(f"🤖 AI Improving Grammar...")
        threading.Thread(target=self._ai_fix_grammar, args=(raw_text,)).start()
        
        self.sentence_buffer = [] 
        self.cursor_pos = 0
        self.emit_text_update()

    def _ai_fix_grammar(self, text):
        prompt = f"Fix the grammar. Output ONLY the corrected sentence. Input: '{text}'"
        try:
            response = requests.post(LLAMA_URL, json={
                "model": MODEL_NAME, "prompt": prompt, "stream": False
            }, timeout=15)
            
            if response.status_code == 200:
                final = response.json().get('response', '').strip().strip('"')
                self.sentence_finalized.emit(final)
                self.status_update.emit(f"✅ Ready: {final}")
            else:
                self.sentence_finalized.emit(text)
        except:
            self.sentence_finalized.emit(text)

    # --- KEYBOARD CONTROLS (Restored) ---
    def insert_word(self, word):
        # Insert word + space
        word_chars = list(word) + [" "]
        for c in word_chars:
            self.sentence_buffer.insert(self.cursor_pos, c)
            self.cursor_pos += 1
        self.emit_text_update()

    def manual_backspace(self):
        if self.cursor_pos > 0: 
            self.sentence_buffer.pop(self.cursor_pos - 1)
            self.cursor_pos -= 1
            self.emit_text_update()

    def manual_move_left(self):
        if self.cursor_pos > 0:
            self.cursor_pos -= 1
            self.emit_text_update()

    def manual_move_right(self):
        if self.cursor_pos < len(self.sentence_buffer):
            self.cursor_pos += 1
            self.emit_text_update()
            
    def manual_add_space(self):
        self.sentence_buffer.insert(self.cursor_pos, " ")
        self.cursor_pos += 1
        self.emit_text_update()
    
    def stop(self):
        self.running = False
        self.wait()

    # --- CV HELPERS ---
    def calc_bounding_rect(self, image, landmarks):
        w, h = image.shape[1], image.shape[0]
        landmark_array = np.array([[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)] for lm in landmarks.landmark])
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        w, h = image.shape[1], image.shape[0]
        return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)] for lm in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for i in range(len(temp)):
            temp[i][0] -= base_x
            temp[i][1] -= base_y
        flat = list(itertools.chain.from_iterable(temp))
        max_val = max(list(map(abs, flat))) or 1
        return [n / max_val for n in flat]

    def draw_landmarks(self, image, points):
        if len(points) > 0:
            for p in points:
                cv.circle(image, tuple(p), 4, (255, 255, 255), -1)

# ============================================================================
# 🖥️ MAIN UI
# ============================================================================
class UnifiedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 AI Sign Language AAC")
        self.setGeometry(100, 100, 1280, 850)
        
        # ✅ FIX: TTS Worker is now a persistent thread
        self.tts_worker = TTSWorker()
        
        self.camera_worker = None
        self.dataset_loader = None 
        self.predictor = PredictionWorker()
        self.predictor.suggestions_ready.connect(self.update_suggestions)
        
        self.init_ui()
        self.apply_theme()

    def get_loader(self):
        if not self.dataset_loader:
            self.dataset_loader = DatasetLoader(DATASET_FOLDER)
        return self.dataset_loader

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)

        # HEADER
        header_layout = QHBoxLayout()
        title = QLabel("🦾 Sign-to-Speech & Video")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d9ff;")
        header_layout.addWidget(title)
        
        self.status_pill = QLabel(" System Ready ")
        self.status_pill.setStyleSheet("background: #333; color: #aaa; border-radius: 12px; padding: 5px 15px;")
        header_layout.addWidget(self.status_pill, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(header_layout)

        # TAB SYSTEM
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 0px; }
            QTabBar::tab { background: #1e1e2e; color: #888; padding: 12px 25px; font-size: 14px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 5px; }
            QTabBar::tab:selected { background: #2d2d3d; color: #fff; font-weight: bold; border-bottom: 2px solid #00d9ff; }
        """)
        
        self.tab_comm = QWidget()
        self.init_comm_tab()
        self.tabs.addTab(self.tab_comm, "🗣️ Live Communication")

        self.tab_trans = QWidget()
        self.init_trans_tab()
        self.tabs.addTab(self.tab_trans, "🎥 Video Translator")

        main_layout.addWidget(self.tabs)

    # ==========================
    # TAB 1: COMMUNICATE
    # ==========================
    def init_comm_tab(self):
        layout = QHBoxLayout(self.tab_comm)

        # --- LEFT: CAMERA CARD ---
        cam_frame = QFrame()
        cam_frame.setStyleSheet("background: #111; border-radius: 10px; border: 1px solid #333;")
        cam_layout = QVBoxLayout(cam_frame)
        
        self.cam_label = QLabel("Camera Off")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # ✅ FIX: Ignore Size Policy prevents the "Growing UI" bug
        self.cam_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        cam_layout.addWidget(self.cam_label)
        
        ctrls = QHBoxLayout()
        self.btn_cam = QPushButton("Start Camera")
        self.btn_cam.setCheckable(True)
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_cam.setStyleSheet("""
            QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }
            QPushButton:checked { background: #e74c3c; }
        """)
        
        self.check_voice = QCheckBox("🔊 Auto-Speak")
        self.check_voice.setChecked(True)
        
        ctrls.addWidget(self.btn_cam)
        ctrls.addWidget(self.check_voice)
        cam_layout.addLayout(ctrls)

        layout.addWidget(cam_frame, 60)

        # --- RIGHT: INTERFACE CARD ---
        ui_frame = QFrame()
        ui_frame.setStyleSheet("background: #2d2d3d; border-radius: 10px;")
        ui_layout = QVBoxLayout(ui_frame)
        ui_layout.setSpacing(10)

        ui_layout.addWidget(QLabel("🧠 AI Suggestions (F1 - F3):"))
        pred_layout = QHBoxLayout()
        self.pred_btns = []
        for i in range(3):
            btn = QPushButton(f"-")
            btn.setStyleSheet("""
                QPushButton { background: #444; color: #fff; padding: 10px; border-radius: 5px; font-weight: bold; }
                QPushButton:hover { background: #00d9ff; color: #000; }
            """)
            btn.clicked.connect(lambda _, x=i: self.use_suggestion(x))
            self.pred_btns.append(btn)
            pred_layout.addWidget(btn)
        ui_layout.addLayout(pred_layout)

        ui_layout.addWidget(QLabel("📝 Typing Buffer:"))
        self.buffer_display = QLabel("")
        self.buffer_display.setStyleSheet("background: #111; color: #00d9ff; font-family: Monospace; font-size: 28px; padding: 15px; border-radius: 5px;")
        ui_layout.addWidget(self.buffer_display)

        ui_layout.addWidget(QLabel("✅ Final Output (Grammar Fixed):"))
        self.final_display = QLabel("Waiting...")
        self.final_display.setWordWrap(True)
        self.final_display.setStyleSheet("background: #222; color: #fff; font-size: 20px; padding: 15px; border-radius: 5px; border-left: 4px solid #00d9ff;")
        self.final_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        ui_layout.addWidget(self.final_display, 1)

        action_layout = QHBoxLayout()
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_all)
        btn_speak = QPushButton("🔊 Speak Again")
        btn_speak.clicked.connect(lambda: self.tts_worker.speak(self.final_display.text()))
        
        action_layout.addWidget(btn_clear)
        action_layout.addWidget(btn_speak)
        ui_layout.addLayout(action_layout)

        layout.addWidget(ui_frame, 40)

    # ==========================
    # TAB 2: TRANSLATE
    # ==========================
    def init_trans_tab(self):
        layout = QVBoxLayout(self.tab_trans)
        
        input_frame = QFrame()
        input_frame.setStyleSheet("background: #2d2d3d; border-radius: 8px; padding: 15px;")
        h_layout = QHBoxLayout(input_frame)
        
        self.trans_input = QLineEdit()
        self.trans_input.setPlaceholderText("Type a sentence to see it signed...")
        self.trans_input.setStyleSheet("font-size: 16px; padding: 8px; border: 1px solid #555; border-radius: 4px;")
        
        btn_go = QPushButton("Translate ➜")
        btn_go.setMinimumHeight(40)
        btn_go.clicked.connect(self.run_video_translation)
        btn_go.setStyleSheet("background: #00d9ff; color: #000; font-weight: bold; border-radius: 4px; padding: 0 20px;")
        
        h_layout.addWidget(self.trans_input)
        h_layout.addWidget(btn_go)
        layout.addWidget(input_frame)

        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(QAudioOutput())
        
        self.gloss_label = QLabel("Gloss: -")
        self.gloss_label.setStyleSheet("font-size: 18px; color: #51cf66; font-weight: bold; padding: 10px; background: #222; border-radius: 5px;")
        self.gloss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.video_widget)
        layout.addWidget(self.gloss_label)

    # ==========================
    # LOGIC
    # ==========================
    def toggle_camera(self):
        if self.btn_cam.isChecked():
            self.camera_worker = CameraWorker()
            self.camera_worker.image_update.connect(self.update_cam_image)
            self.camera_worker.text_update.connect(self.on_buffer_update)
            self.camera_worker.status_update.connect(lambda s: self.status_pill.setText(s))
            self.camera_worker.sentence_finalized.connect(self.on_sentence_finalized)
            self.camera_worker.start()
            self.btn_cam.setText("Stop Camera")
            self.setFocus()
        else:
            if self.camera_worker:
                self.camera_worker.stop()
                self.camera_worker = None
            self.cam_label.setPixmap(QPixmap())
            self.btn_cam.setText("Start Camera")

    @pyqtSlot(QImage)
    def update_cam_image(self, image):
        pix = QPixmap.fromImage(image)
        self.cam_label.setPixmap(pix.scaled(
            self.cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    @pyqtSlot(str)
    def on_buffer_update(self, text):
        clean_text = text.replace("|", "")
        self.buffer_display.setText(text)
        self.predictor.update_text(clean_text)

    @pyqtSlot(list)
    def update_suggestions(self, words):
        for i, btn in enumerate(self.pred_btns):
            if i < len(words):
                btn.setText(f"F{i+1}: {words[i]}")
                btn.setEnabled(True)
            else:
                btn.setText("-")
                btn.setEnabled(False)

    def use_suggestion(self, index):
        btn = self.pred_btns[index]
        text = btn.text()
        if ":" in text and self.camera_worker:
            word = text.split(": ")[1]
            self.camera_worker.insert_word(word)

    @pyqtSlot(str)
    def on_sentence_finalized(self, text):
        self.final_display.setText(text)
        if self.check_voice.isChecked():
            self.tts_worker.speak(text)
        self.trans_input.setText(text) 

    def clear_all(self):
        self.buffer_display.setText("")
        self.final_display.setText("...")
        if self.camera_worker:
            self.camera_worker.sentence_buffer = []
            self.camera_worker.cursor_pos = 0

    def run_video_translation(self):
        text = self.trans_input.text()
        if not text: return
        threading.Thread(target=self._generate_video_thread, args=(text,)).start()

    def _generate_video_thread(self, text):
        try:
            loader = self.get_loader()
            gloss_list = logic.translate_to_gloss(text)
            gloss_words = " ".join(gloss_list).split()
            
            videos = [loader.get_video(w.strip('.,!?;:')) for w in gloss_words]
            videos = [v for v in videos if v]
            
            video_path = videos[0] if len(videos) == 1 else logic.generate_asl_video(gloss_words)
            if video_path and os.path.exists(video_path):
                self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
                self.media_player.play()
                self.gloss_label.setText(f"Gloss: {' '.join(gloss_list)}")
        except Exception as e:
            print(e)

    # ✅ KEYBOARD EVENTS RESTORED
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        
        # Only process camera keys if we are in the Communicate Tab
        if self.tabs.currentIndex() == 0 and self.camera_worker:
            if key == Qt.Key.Key_Left: self.camera_worker.manual_move_left()
            elif key == Qt.Key.Key_Right: self.camera_worker.manual_move_right()
            elif key == Qt.Key.Key_Backspace: self.camera_worker.manual_backspace()
            elif key == Qt.Key.Key_Space: self.camera_worker.manual_add_space()
            elif key == Qt.Key.Key_Return: self.camera_worker.finalize_sentence()
            
            # Suggestion Hotkeys
            elif key == Qt.Key.Key_F1: self.use_suggestion(0)
            elif key == Qt.Key.Key_F2: self.use_suggestion(1)
            elif key == Qt.Key.Key_F3: self.use_suggestion(2)
        
        super().keyPressEvent(event)

    def apply_theme(self):
        app = QApplication.instance()
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))

    def closeEvent(self, event):
        if self.tts_worker:
            self.tts_worker.stop()
        if self.camera_worker:
            self.camera_worker.stop()
        event.accept()

# ============================================================================
# DATASET LOADER
# ============================================================================
class DatasetLoader:
    def __init__(self, dataset_folder: str):
        self.word_to_video = {}
        if os.path.exists(dataset_folder):
            for root, dirs, files in os.walk(dataset_folder):
                for filename in files:
                    if filename.endswith('.mp4'):
                        self.word_to_video[filename[:-4].lower()] = os.path.join(root, filename)
    def get_video(self, word):
        w = word.lower().strip()
        return self.word_to_video.get(w) or self.word_to_video.get(w.replace(" ", "_"))

def main():
    app = QApplication(sys.argv)
    window = UnifiedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()