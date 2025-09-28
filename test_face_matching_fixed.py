#!/usr/bin/env python3
"""
Test the fixed face matching functionality
"""

import json
import os
from enhanced_face_matcher_simple import SimpleEnhancedFaceMatcher

def test_face_matching():
    """Test face matching with the fixed implementation"""
    print("🧪 Testing Fixed Face Matching...")
    
    # Initialize face matcher
    face_matcher = SimpleEnhancedFaceMatcher()
    
    # Load analysis data
    analysis_file = "realtime_analysis/comprehensive_analysis_report.json"
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return False
    
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    print(f"📊 Loaded analysis data with {len(analysis_data.get('faces', {}))} faces")
    
    # Test face matching
    try:
        results = face_matcher.process_video_faces(analysis_data, "test_video")
        
        print(f"✅ Face matching completed!")
        print(f"📊 Results type: {type(results)}")
        print(f"📊 Results: {results}")
        
        if isinstance(results, dict):
            print(f"👥 Matched faces: {results.get('matched', 0)}")
            print(f"➕ New faces: {results.get('new', 0)}")
            print(f"📈 Total processed: {results.get('total_processed', 0)}")
            print(f"✅ Success: {results.get('success', False)}")
        else:
            print(f"⚠️ Unexpected results format: {results}")
        
        # Test attendance summary
        attendance = face_matcher.get_attendance_summary()
        print(f"📊 Attendance summary: {attendance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face matching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_face_matching()
    if success:
        print("✅ Face matching test passed!")
    else:
        print("❌ Face matching test failed!")

