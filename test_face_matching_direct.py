"""
Direct test of face matching functionality
"""

import json
import os
from enhanced_face_matcher import EnhancedFaceMatcher

def test_face_matching():
    """Test face matching directly"""
    print("🧪 Testing Face Matching Directly")
    print("=" * 40)
    
    # Initialize face matcher
    face_matcher = EnhancedFaceMatcher()
    
    # Load analysis data
    analysis_file = "analysis_results/comprehensive_analysis_report.json"
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return
    
    try:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"✅ Loaded analysis data with {len(analysis_data.get('faces', {}))} faces")
        
        # Test face matching
        print("🔍 Testing face matching...")
        matching_results = face_matcher.process_video_faces(analysis_data, "test_video")
        
        print(f"📊 Matching results type: {type(matching_results)}")
        print(f"📊 Matching results: {matching_results}")
        
        # Test attendance summary
        print("📋 Testing attendance summary...")
        attendance_summary = face_matcher.get_attendance_summary()
        
        print(f"📊 Attendance summary type: {type(attendance_summary)}")
        print(f"📊 Attendance summary: {attendance_summary}")
        
        print("✅ Face matching test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during face matching test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_matching()

