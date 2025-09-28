"""
Simple test for automated video processing
"""

import os
import shutil
from pathlib import Path
from automated_video_processor import AutomatedVideoProcessor

def test_simple_processing():
    """Test simple automated processing"""
    print("🧪 Testing Simple Automated Processing")
    print("=" * 40)
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Check if test video exists
    test_video = "test.mp4"
    if not os.path.exists(test_video):
        print(f"❌ Test video '{test_video}' not found!")
        return False
    
    # Copy test video to input folder
    today_folder = processor.get_today_folder()
    input_folder = today_folder / "input"
    input_folder.mkdir(exist_ok=True)
    
    test_video_dest = input_folder / test_video
    if test_video_dest.exists():
        test_video_dest.unlink()  # Remove existing
    
    shutil.copy2(test_video, test_video_dest)
    print(f"📹 Copied test video to: {test_video_dest}")
    
    # Test processing
    print("🚀 Starting automated processing...")
    try:
        success = processor.process_all_videos()
        if success:
            print("✅ Automated processing completed successfully!")
            
            # Check if results were created
            analysis_folder = processor.analysis_folder / today_folder.name
            if analysis_folder.exists():
                analysis_dirs = [d for d in analysis_folder.iterdir() if d.is_dir()]
                print(f"📊 Found {len(analysis_dirs)} analysis directories")
                
                for analysis_dir in analysis_dirs:
                    comprehensive_file = analysis_dir / "comprehensive_analysis_report.json"
                    if comprehensive_file.exists():
                        print(f"✅ Found comprehensive report: {comprehensive_file}")
                    else:
                        print(f"⚠️ No comprehensive report in: {analysis_dir}")
            else:
                print("⚠️ No analysis folder created")
            
            return True
        else:
            print("❌ Automated processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_processing()

