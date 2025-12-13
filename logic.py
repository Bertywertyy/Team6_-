import requests
import pyttsx3
import speech_recognition as sr
import os
import random
import spacy
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import time

# --- CONFIGURATION ---
LLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
DATASET_DIR = "dataset"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize modules
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Spacy model not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None 

# ==========================================
# 1. AUDIO LISTENER
# ==========================================
def listen_to_audio():
    """Captures audio from the microphone and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5)
            # Google API requires internet but provides best accuracy
            text = recognizer.recognize_google(audio)
            return text
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            return None
        except sr.RequestError:
            print("❌ Internet required for Speech Recognition.")
            return None

# ==========================================
# 2. AI TRANSLATION
# ==========================================
def translate_to_gloss(text):
    """Uses Llama 3.2 to convert English grammar to ASL Gloss."""
    prompt = (
        "Translate the following English sentence to ASL GLOSS. "
        "Strictly use ALL CAPS, fix word order (Time-Topic-Comment), and remove auxiliary/linking verbs (is, am, are, the, a). "
        "Output ONLY the sequence of gloss words separated by spaces. "
        f"Input: '{text}'"
    )
    
    try:
        response = requests.post(LLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False
        }, timeout=10)
        
        if response.status_code == 200:
            gloss = response.json().get('response', '').strip().upper()
            return gloss.split()
    except Exception as e:
        print(f"⚠️ AI Error: {e}")
    
    # Fallback if AI fails
    return text.upper().split()

# ==========================================
# 3. VIDEO GENERATION
# ==========================================
def get_video_path(word):
    """Finds the video file (supports numbered variations like hello_1.mp4)."""
    word = word.lower()
    if not word: return None

    # Determine subfolder (a, b, c... or numbers)
    if word[0].isdigit():
        folder = "numbers"
    elif word[0].isalpha():
        folder = word[0]
    else:
        return None 
    
    folder_path = os.path.join(DATASET_DIR, folder)
    
    if not os.path.exists(folder_path):
        return None

    # Find matches (Exact match OR variations)
    candidates = []
    for f in os.listdir(folder_path):
        # Match 'hello.mp4'
        if f == f"{word}.mp4":
            candidates.append(f)
        # Match 'hello_1.mp4', but NOT 'hellostorm.mp4'
        elif f.startswith(f"{word}_") and f.endswith(".mp4"):
            candidates.append(f)
            
    if candidates:
        # Pick a random variation to make movement look natural
        return os.path.join(folder_path, random.choice(candidates))
    
    return None

def generate_asl_video(gloss_words):
    """Stitches video clips together with crossfades."""
    clips = []
    
    for word in gloss_words:
        path = get_video_path(word)
        if path:
            try:
                # Resize to standard height to ensure smooth stitching
                clip = VideoFileClip(path).resize(height=480)
                
                # Try Text Overlay (Safe Mode - skips if ImageMagick missing)
                try:
                    txt = TextClip(word, fontsize=50, color='yellow', font='Arial-Bold', stroke_color='black', stroke_width=2)
                    txt = txt.set_position(('center', 0.85), relative=True).set_duration(clip.duration)
                    video = CompositeVideoClip([clip, txt])
                except Exception:
                    video = clip

                # Add crossfade for smooth transition
                if clips: video = video.crossfadein(0.2)
                clips.append(video)
            except Exception as e:
                print(f"❌ Corrupt Video File {path}: {e}")
                
    if clips:
        # Concatenate with negative padding to create the overlap/crossfade
        final = concatenate_videoclips(clips, method="compose", padding=-0.2)
        
        # FIX: Generate Unique Filename to prevent file locking errors
        filename = f"output_{int(time.time())}.mp4"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        final.write_videofile(output_path, fps=24, logger=None)
        return output_path
        
    return None