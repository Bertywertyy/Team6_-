import os
import cv2
import shutil
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_best"

def get_video_quality_score(video_path):
    """
    Calculate a quality score for a video based on:
    - Resolution (width x height)
    - Frame count
    - File size
    - Frame rate
    
    Returns a tuple: (score, details_dict)
    """
    try:
        # Get file size
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 0, None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Calculate resolution score (pixels)
        resolution = width * height
        
        # Quality score formula:
        # - Resolution is most important (weighted heavily)
        # - File size per second (higher = better quality, more detail)
        # - Frame rate (smoother playback)
        
        bitrate_score = (file_size / duration) if duration > 0 else 0
        
        # Weighted score
        score = (
            resolution * 1.0 +           # Resolution weight
            bitrate_score * 100000 +     # Bitrate weight
            fps * 1000                    # FPS weight
        )
        
        details = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'file_size_mb': file_size,
            'resolution': resolution,
            'score': score
        }
        
        return score, details
        
    except Exception as e:
        print(f"   ⚠️ Error analyzing {video_path}: {e}")
        return 0, None

def get_word_from_filename(filename):
    """
    Extract the word from a filename.
    Examples:
    - 'book.mp4' -> 'book'
    - 'book_1.mp4' -> 'book'
    - 'a lot_3.mp4' -> 'a lot'
    """
    # Remove .mp4 extension
    name = filename.replace('.mp4', '')
    
    # If it ends with _number, remove that part
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    
    return name

def select_best_videos():
    """
    Go through each letter folder, find all unique words,
    and select the best quality video for each word.
    """
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("🎬 Starting Best Video Selection...")
    print(f"📂 Source: {DATASET_DIR}")
    print(f"📂 Destination: {OUTPUT_DIR}")
    print("-" * 60)
    
    total_words_processed = 0
    total_videos_selected = 0
    
    # Process each letter folder (a-z)
    for letter in os.listdir(DATASET_DIR):
        letter_path = os.path.join(DATASET_DIR, letter)
        
        if not os.path.isdir(letter_path):
            continue
        
        print(f"\n📁 Processing folder: {letter}/")
        
        # Group videos by word
        word_videos = {}
        
        for filename in os.listdir(letter_path):
            if not filename.endswith('.mp4'):
                continue
            
            word = get_word_from_filename(filename)
            video_path = os.path.join(letter_path, filename)
            
            if word not in word_videos:
                word_videos[word] = []
            
            word_videos[word].append(video_path)
        
        # Create output letter folder
        output_letter_path = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(output_letter_path, exist_ok=True)
        
        # Process each word
        for word, video_paths in word_videos.items():
            if len(video_paths) == 1:
                # Only one video, just copy it
                src = video_paths[0]
                dst = os.path.join(output_letter_path, f"{word}.mp4")
                shutil.copy2(src, dst)
                print(f"   ✅ {word}: Only 1 video, copied.")
                total_videos_selected += 1
            else:
                # Multiple videos, analyze and select best
                print(f"   🔍 {word}: Analyzing {len(video_paths)} videos...")
                
                best_video = None
                best_score = 0
                best_details = None
                
                for video_path in video_paths:
                    score, details = get_video_quality_score(video_path)
                    
                    if score > best_score:
                        best_score = score
                        best_video = video_path
                        best_details = details
                
                if best_video:
                    # Copy the best video
                    dst = os.path.join(output_letter_path, f"{word}.mp4")
                    shutil.copy2(best_video, dst)
                    
                    if best_details:
                        print(f"   ✅ {word}: Selected {os.path.basename(best_video)}")
                        print(f"      Resolution: {best_details['width']}x{best_details['height']}, "
                              f"FPS: {best_details['fps']:.1f}, "
                              f"Size: {best_details['file_size_mb']:.2f}MB")
                    else:
                        print(f"   ✅ {word}: Selected {os.path.basename(best_video)}")
                    
                    total_videos_selected += 1
            
            total_words_processed += 1
    
    print("\n" + "=" * 60)
    print("🎉 Best Video Selection Complete!")
    print(f"📚 Total unique words processed: {total_words_processed}")
    print(f"🎬 Total videos selected: {total_videos_selected}")
    print(f"📂 Output directory: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    select_best_videos()
