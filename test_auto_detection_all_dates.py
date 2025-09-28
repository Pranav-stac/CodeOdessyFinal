#!/usr/bin/env python3
"""
Test script for auto-detection of videos in all date folders
"""

import os
import sys
from pathlib import Path
from datetime import date

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_video_processor import AutomatedVideoProcessor

def test_auto_detection_all_dates():
    """Test auto-detection functionality across all date folders"""
    print("🧪 Testing Auto-Detection Across All Date Folders")
    print("=" * 60)
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Get all available date folders
    available_dates = processor.get_available_dates()
    print(f"📅 Available date folders: {available_dates}")
    
    if not available_dates:
        print("❌ No date folders found")
        return
    
    total_videos_to_process = 0
    videos_by_date = {}
    
    # Check each date folder for unprocessed videos
    for date_str in available_dates:
        print(f"\n📁 Checking {date_str}...")
        
        # Get videos and processed videos for this date
        videos = processor.get_videos_for_date(date_str)
        processed_videos = processor.get_processed_videos(date_str)
        processed_names = {p['video_name'] for p in processed_videos}
        
        print(f"  🎬 Total videos: {len(videos)}")
        print(f"  ✅ Processed videos: {len(processed_videos)}")
        
        # Filter out already processed videos
        unprocessed_videos = [v for v in videos if v['name'] not in processed_names]
        
        if unprocessed_videos:
            videos_by_date[date_str] = unprocessed_videos
            total_videos_to_process += len(unprocessed_videos)
            print(f"  📹 Unprocessed videos: {len(unprocessed_videos)}")
            for video in unprocessed_videos:
                print(f"    - {video['name']} ({video['size'] / (1024*1024):.2f} MB)")
        else:
            print(f"  ✅ All videos already processed")
    
    print(f"\n📊 Summary:")
    print(f"  📅 Date folders checked: {len(available_dates)}")
    print(f"  📹 Total unprocessed videos: {total_videos_to_process}")
    print(f"  📁 Date folders with unprocessed videos: {len(videos_by_date)}")
    
    if total_videos_to_process == 0:
        print("\n✅ No unprocessed videos found in any date folder")
        return
    
    print(f"\n🚀 Would process {total_videos_to_process} video(s) across {len(videos_by_date)} date folder(s)")
    
    # Show what would be processed
    for date_str, videos in videos_by_date.items():
        print(f"\n📅 {date_str}:")
        for video in videos:
            print(f"  🎬 {video['name']} - {video['size'] / (1024*1024):.2f} MB")
    
    print("\n✅ Auto-detection test completed!")

if __name__ == "__main__":
    test_auto_detection_all_dates()

