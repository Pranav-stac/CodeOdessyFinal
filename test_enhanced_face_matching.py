"""
Test the enhanced face matcher with real data
"""

import os
import json
from enhanced_face_matcher import EnhancedFaceMatcher

def test_enhanced_face_matching():
    """Test enhanced face matching with real analysis data"""
    print("🧪 Testing Enhanced Face Matching...")
    
    # Initialize matcher
    matcher = EnhancedFaceMatcher()
    
    # Test with real analysis data
    analysis_file = "realtime_analysis/comprehensive_analysis_report.json"
    
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return False
    
    print(f"📄 Loading analysis data from: {analysis_file}")
    
    try:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"✅ Loaded analysis data")
        print(f"📊 Faces found: {len(analysis_data.get('faces', {}))}")
        
        # Process faces
        print("\n🔍 Processing faces...")
        results = matcher.process_video_faces(analysis_data, "test_video")
        
        print(f"\n📊 Results:")
        print(f"  Total processed: {results['total_processed']}")
        print(f"  Matched: {results['matched']}")
        print(f"  New: {results['new']}")
        
        # Get attendance summary
        attendance = matcher.get_attendance_summary()
        print(f"\n👥 Attendance Summary:")
        print(f"  Total people: {len(attendance)}")
        
        for person_id, person_data in attendance.items():
            print(f"  {person_id}: {person_data['total_appearances']} appearances")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_face_matching()
    if success:
        print("\n🎉 Enhanced face matching test passed!")
    else:
        print("\n❌ Enhanced face matching test failed!")

