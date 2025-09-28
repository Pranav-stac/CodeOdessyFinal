#!/usr/bin/env python3
"""
Test script to verify auto-detection fix
"""

import os
import sys
from pathlib import Path
from datetime import date

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_video_processor import AutomatedVideoProcessor

def test_auto_detection_fix():
    """Test that auto-detection can process videos without GUI attributes"""
    print("🧪 Testing Auto-Detection Fix")
    print("=" * 40)
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Get all available date folders
    available_dates = processor.get_available_dates()
    print(f"📅 Available date folders: {available_dates}")
    
    if not available_dates:
        print("❌ No date folders found")
        return
    
    # Check for unprocessed videos
    total_videos_to_process = 0
    videos_by_date = {}
    
    for date_str in available_dates:
        print(f"\n📁 Checking {date_str}...")
        
        videos = processor.get_videos_for_date(date_str)
        processed_videos = processor.get_processed_videos(date_str)
        processed_names = {p['video_name'] for p in processed_videos}
        
        unprocessed_videos = [v for v in videos if v['name'] not in processed_names]
        
        if unprocessed_videos:
            videos_by_date[date_str] = unprocessed_videos
            total_videos_to_process += len(unprocessed_videos)
            print(f"  📹 Unprocessed videos: {len(unprocessed_videos)}")
            for video in unprocessed_videos:
                print(f"    - {video['name']} ({video['size'] / (1024*1024):.2f} MB)")
        else:
            print(f"  ✅ All videos already processed")
    
    if total_videos_to_process == 0:
        print("\n✅ No unprocessed videos found")
        return
    
    print(f"\n🚀 Testing processing of {total_videos_to_process} video(s)...")
    
    # Test processing with default values
    total_processed = 0
    total_failed = 0
    
    for date_str, videos in videos_by_date.items():
        print(f"\n📅 Processing videos for {date_str}...")
        
        for video in videos:
            try:
                print(f"🎬 Processing: {video['name']}")
                
                # Process with default values (no GUI attributes)
                output_dir = processor.analysis_folder / date_str / Path(video['name']).stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                success = processor.process_video(
                    video['path'], 
                    str(output_dir), 
                    video['name'],
                    False,  # show_footage = False
                    True    # realtime_display = True
                )
                
                if success:
                    total_processed += 1
                    print(f"✅ Processed: {video['name']}")
                else:
                    total_failed += 1
                    print(f"❌ Failed: {video['name']}")
                    
            except Exception as e:
                total_failed += 1
                print(f"❌ Error processing {video['name']}: {e}")
    
    print(f"\n📊 Processing completed!")
    print(f"✅ Successfully processed: {total_processed} videos")
    print(f"❌ Failed: {total_failed} videos")
    
    print("\n✅ Auto-detection fix test completed!")

if __name__ == "__main__":
    test_auto_detection_fix()

