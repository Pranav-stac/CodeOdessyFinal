#!/usr/bin/env python3
"""
Test the progress display functionality
"""

from automated_video_processor import AutomatedVideoProcessor

def test_progress_callback(progress, message):
    """Test progress callback function"""
    print(f"[{progress:5.1f}%] {message}")

def test_auto_processing():
    """Test automated processing with progress callbacks"""
    print("🧪 Testing Automated Processing with Progress Display...")
    
    # Initialize processor
    processor = AutomatedVideoProcessor()
    
    # Test progress callback
    print("📊 Testing progress callback...")
    test_progress_callback(0, "Starting test")
    test_progress_callback(50, "Halfway done")
    test_progress_callback(100, "Complete")
    
    # Test getting available dates
    print("\n📅 Testing available dates...")
    dates = processor.get_available_dates()
    print(f"Available dates: {dates}")
    
    # Test processing with progress callback
    print("\n🚀 Testing process_all_available_dates with progress callback...")
    try:
        result = processor.process_all_available_dates(
            progress_callback=test_progress_callback,
            show_footage=False,
            realtime_display=False
        )
        print(f"✅ Processing result: {result}")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auto_processing()

