import json
import os
import shutil

# --- CONFIGURATION ---
JSON_FILE = "WLASL_v0.3.json"
SOURCE_DIR = "raw_videos"   # Where your numbered .mp4 files are currently
TARGET_DIR = "dataset"      # Where we want them to go (organized)

def setup_dataset():
    # 1. Load the JSON Map
    print(f"📖 Reading {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {JSON_FILE}. Please put it in this folder.")
        return

    total_definitions = len(data)
    print(f"📚 Total words defined in JSON: {total_definitions}")

    # 2. Create Target Directory
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    print("🚀 Starting Organization...")
    
    total_videos_count = 0
    unique_words_count = 0
    missing_instances = 0

    # 3. Loop through every word in the dictionary
    for entry in data:
        word = entry['gloss']  # e.g., "book"
        instances = entry['instances'] # List of videos for this word
        
        found_any_for_this_word = False

        for i, vid in enumerate(instances):
            video_id = vid['video_id'] # e.g., "69241"
            
            # The file currently sits as "raw_videos/69241.mp4"
            src_filename = f"{video_id}.mp4"
            src_path = os.path.join(SOURCE_DIR, src_filename)

            if os.path.exists(src_path):
                # Determine destination folder (A-Z)
                first_letter = word[0].lower()
                if not first_letter.isalpha():
                    first_letter = "numbers"
                
                dest_folder = os.path.join(TARGET_DIR, first_letter)
                os.makedirs(dest_folder, exist_ok=True)

                # Determine new filename
                # Instance 0 -> "book.mp4"
                # Instance 1 -> "book_1.mp4"
                if i == 0:
                    new_filename = f"{word}.mp4"
                else:
                    new_filename = f"{word}_{i}.mp4"
                
                dest_path = os.path.join(dest_folder, new_filename)

                # Copy/Move the file
                shutil.copy2(src_path, dest_path)
                
                print(f"✅ Moved: {src_filename} -> {first_letter}/{new_filename}")
                
                total_videos_count += 1
                found_any_for_this_word = True
            else:
                missing_instances += 1
        
        if found_any_for_this_word:
            unique_words_count += 1

    print("-" * 40)
    print(f"🎉 Organization Complete!")
    print(f"📚 Total Words in Dictionary:  {total_definitions}")
    print(f"✅ Unique Words Found:         {unique_words_count}")
    print(f"🎬 Total Video Clips Moved:    {total_videos_count}")
    print("-" * 40)

if __name__ == "__main__":
    setup_dataset()