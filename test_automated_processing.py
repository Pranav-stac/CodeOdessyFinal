"""
Test script for automated video processing
"""

import os
import shutil
from pathlib import Path
from automated_video_processor import AutomatedVideoProcessor

def test_automated_processing():
    """Test the automated video processing system"""
    print("🧪 Testing Automated Video Processing System")
    print("=" * 50)
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Check folder structure
    today_folder = processor.get_today_folder()
    print(f"📁 Today's folder: {today_folder}")
    
    # Check if test video exists
    test_video = "test.mp4"
    if not os.path.exists(test_video):
        print(f"❌ Test video '{test_video}' not found!")
        return False
    
    # Copy test video to input folder
    input_folder = today_folder / "input"
    input_folder.mkdir(exist_ok=True)
    
    test_video_dest = input_folder / test_video
    if not test_video_dest.exists():
        shutil.copy2(test_video, test_video_dest)
        print(f"📹 Copied test video to: {test_video_dest}")
    
    # Check videos
    videos = processor.get_videos_for_date(today_folder.name)
    print(f"🎬 Found {len(videos)} videos to process")
    
    for video in videos:
        print(f"  - {video['name']} ({video['size']} bytes)")
    
    # Test processing (without actually running it)
    print(f"\n✅ Automated processing system ready!")
    print(f"📁 Add videos to: {input_folder}")
    print(f"📊 Results will be saved to: {processor.analysis_folder}")
    
    return True

if __name__ == "__main__":
    test_automated_processing()

