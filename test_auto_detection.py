#!/usr/bin/env python3
"""
Test script for auto-detection of videos in input folder
"""

import os
import sys
from pathlib import Path
from datetime import date

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_video_processor import AutomatedVideoProcessor

def test_auto_detection():
    """Test auto-detection functionality"""
    print("🧪 Testing Auto-Detection of Videos")
    print("=" * 50)
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Get today's folder
    today = date.today().strftime("%Y-%m-%d")
    input_folder = processor.base_folder / today / "input"
    
    print(f"📁 Checking input folder: {input_folder}")
    
    # Check if folder exists
    if not input_folder.exists():
        print(f"❌ Input folder does not exist: {input_folder}")
        print("📝 Creating input folder...")
        input_folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created input folder: {input_folder}")
    
    # Check for videos
    videos = processor.get_videos_for_date(today)
    print(f"🎬 Found {len(videos)} video(s) in input folder")
    
    if videos:
        print("📋 Video details:")
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video['name']}")
            print(f"     Size: {video['size'] / (1024*1024):.2f} MB")
            print(f"     Modified: {video['modified']}")
            print(f"     Path: {video['path']}")
    
    # Test auto-detection
    print(f"\n🔍 Testing auto-detection...")
    success = processor.auto_detect_and_process(show_footage=False, realtime_display=False)
    
    if success:
        print("✅ Auto-detection and processing completed successfully!")
    else:
        print("❌ Auto-detection failed or no videos to process")
    
    # Check processed videos
    processed_videos = processor.get_processed_videos(today)
    print(f"\n📊 Processed videos: {len(processed_videos)}")
    
    if processed_videos:
        print("📋 Processed video details:")
        for i, video in enumerate(processed_videos, 1):
            print(f"  {i}. {video['video_name']}")
            print(f"     Status: {video['status']}")
            print(f"     Processed at: {video['processed_at']}")
            print(f"     Face count: {video.get('face_count', 0)}")
            print(f"     Student count: {video.get('student_count', 0)}")
    
    print("\n✅ Auto-detection test completed!")

if __name__ == "__main__":
    test_auto_detection()

