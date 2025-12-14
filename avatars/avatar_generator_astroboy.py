import cv2 as cv
import mediapipe as mp
import os
import numpy as np

# CONFIGURATION
INPUT_FOLDER = "./dataset/test_wlasl"
OUTPUT_FOLDER = "./dataset/custom"
ASSET_HEAD = "./assets/astro_head.png"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class AvatarGenerator:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.img_head = None
        
        if os.path.exists(ASSET_HEAD):
            loaded_img = cv.imread(ASSET_HEAD, cv.IMREAD_UNCHANGED)
            if loaded_img is not None:
                # Ensure 4 channels (BGRA) for transparency
                if len(loaded_img.shape) == 3 and loaded_img.shape[2] == 3:
                    self.img_head = cv.cvtColor(loaded_img, cv.COLOR_BGR2BGRA)
                elif len(loaded_img.shape) == 3 and loaded_img.shape[2] == 4:
                    self.img_head = loaded_img
                else:
                    self.img_head = None

    def zoom_landmarks(self, landmark_list, scale_factor):
        """ Shrinks landmarks toward the center (0.5, 0.5) to simulate zooming out """
        if not landmark_list: return
        for lm in landmark_list.landmark:
            # Shift center to 0, scale, shift back to 0.5
            lm.x = 0.5 + (lm.x - 0.5) * scale_factor
            lm.y = 0.5 + (lm.y - 0.5) * scale_factor

    def overlay_image(self, bg, overlay, x, y, size_w):
        if overlay is None: return bg
        h, w = overlay.shape[:2]
        aspect_ratio = h / w
        new_w = int(size_w)
        new_h = int(new_w * aspect_ratio)
        try:
            overlay_resized = cv.resize(overlay, (new_w, new_h))
        except: return bg

        x_start = int(x - new_w // 2)
        y_start = int(y - new_h // 2)
        bg_h, bg_w = bg.shape[:2]
        
        if x_start < 0: x_start = 0
        if y_start < 0: y_start = 0
        if x_start + new_w > bg_w: new_w = bg_w - x_start
        if y_start + new_h > bg_h: new_h = bg_h - y_start
        if new_w <= 0 or new_h <= 0: return bg

        overlay_crop = overlay_resized[:new_h, :new_w]
        alpha_s = overlay_crop[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            bg[y_start:y_start+new_h, x_start:x_start+new_w, c] = (
                alpha_s * overlay_crop[:, :, c] +
                alpha_l * bg[y_start:y_start+new_h, x_start:x_start+new_w, c]
            )
        return bg

    def draw_limb(self, canvas, p1, p2, color, thickness):
        h, w, _ = canvas.shape
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        
        # Avoid drawing glitchy points at (0,0)
        if (x1 < 10 and y1 < 10) or (x2 < 10 and y2 < 10): return

        cv.line(canvas, (x1, y1), (x2, y2), color, thickness, cv.LINE_AA)
        cv.circle(canvas, (x1, y1), thickness // 2, color, -1)
        cv.circle(canvas, (x2, y2), thickness // 2, color, -1)

    def process_video(self, input_path, output_path):
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened(): return False

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS)) or 30
        out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # --- ASTRO BOY COLORS ---
        C_SKIN  = (180, 200, 255) 
        C_BLACK = (30, 30, 30)    
        C_RED   = (50, 50, 230)   
        C_GREEN = (80, 200, 80)   

        # ✅ ZOOM SETTING (Lower = Smaller Character)
        ZOOM_LEVEL = 0.7  

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                results = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                
                # 1. Apply Zoom (Shrink Skeleton)
                self.zoom_landmarks(results.pose_landmarks, ZOOM_LEVEL)
                self.zoom_landmarks(results.left_hand_landmarks, ZOOM_LEVEL)
                self.zoom_landmarks(results.right_hand_landmarks, ZOOM_LEVEL)

                # 2. Draw Background
                canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # 3. Calculate Dynamic Thickness
                    shoulders_w = abs(lm[11].x - lm[12].x) * w
                    base_thick = int(shoulders_w * 0.35) 
                    if base_thick < 4: base_thick = 4

                    # --- LEGS ---
                    self.draw_limb(canvas, lm[23], lm[25], C_BLACK, base_thick) # Thighs
                    self.draw_limb(canvas, lm[24], lm[26], C_BLACK, base_thick) 
                    self.draw_limb(canvas, lm[25], lm[27], C_RED,   int(base_thick*1.2)) # Boots
                    self.draw_limb(canvas, lm[26], lm[28], C_RED,   int(base_thick*1.2)) 

                    # --- TORSO (Solid Shape) ---
                    pts = np.array([
                        [int(lm[11].x*w), int(lm[11].y*h)], 
                        [int(lm[12].x*w), int(lm[12].y*h)], 
                        [int(lm[24].x*w), int(lm[24].y*h)], 
                        [int(lm[23].x*w), int(lm[23].y*h)] 
                    ])
                    cv.fillConvexPoly(canvas, pts, C_SKIN)
                    self.draw_limb(canvas, lm[23], lm[24], C_GREEN, base_thick) # Belt

                    # --- ARMS ---
                    self.draw_limb(canvas, lm[11], lm[13], C_SKIN, base_thick) 
                    self.draw_limb(canvas, lm[13], lm[15], C_SKIN, base_thick) 
                    self.draw_limb(canvas, lm[12], lm[14], C_SKIN, base_thick) 
                    self.draw_limb(canvas, lm[14], lm[16], C_SKIN, base_thick) 

                    # --- HEAD ---
                    if self.img_head is not None:
                        nose = lm[0]
                        # Reduced Head Scale from 2.0 to 1.6
                        head_size = shoulders_w * 1.6 
                        cx, cy = int(nose.x * w), int(nose.y * h)
                        canvas = self.overlay_image(canvas, self.img_head, cx, cy - int(head_size*0.1), head_size)

                # --- HANDS ---
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=C_SKIN, thickness=int(base_thick/3), circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=C_SKIN, thickness=int(base_thick/3)))
                
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=C_SKIN, thickness=int(base_thick/3), circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=C_SKIN, thickness=int(base_thick/3)))

                out.write(canvas)

        cap.release()
        out.release()
        return True

    def run_batch(self):
        files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4")]
        print(f"🚀 Astro-Boy-ifying {len(files)} videos...")
        for i, f in enumerate(files):
            print(f"Processing [{i+1}/{len(files)}]: {f}...")
            self.process_video(os.path.join(INPUT_FOLDER, f), os.path.join(OUTPUT_FOLDER, f))
        print("✅ Done! Character Generation Complete.")

if __name__ == "__main__":
    gen = AvatarGenerator()
    gen.run_batch()