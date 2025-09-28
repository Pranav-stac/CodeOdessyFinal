#!/usr/bin/env python3
"""
Test the vector face matcher
"""

import json
import numpy as np
from vector_face_matcher import VectorFaceMatcher

def test_vector_face_matcher():
    """Test the vector face matcher with real data"""
    print("🧪 Testing Vector Face Matcher...")
    
    # Initialize matcher
    matcher = VectorFaceMatcher()
    
    # Load analysis data
    with open("realtime_analysis/comprehensive_analysis_report.json", 'r') as f:
        analysis_data = json.load(f)
    
    print(f"📊 Loaded analysis data with {len(analysis_data.get('faces', {}))} faces")
    
    # Test with first few faces only
    faces = analysis_data.get('faces', {})
    test_faces = dict(list(faces.items())[:5])  # Only test first 5 faces
    
    # Create test analysis data
    test_analysis = analysis_data.copy()
    test_analysis['faces'] = test_faces
    
    print(f"🎬 Testing with {len(test_faces)} faces...")
    
    # Process faces
    try:
        results = matcher.process_video_faces(test_analysis, "test_video")
        
        print(f"✅ Face matching completed!")
        print(f"📊 Results: {results}")
        
        if isinstance(results, dict):
            print(f"👥 Matched faces: {results.get('matched', 0)}")
            print(f"➕ New faces: {results.get('new', 0)}")
            print(f"📈 Total processed: {results.get('total_processed', 0)}")
            print(f"✅ Success: {results.get('success', False)}")
        
        # Test attendance summary
        attendance = matcher.get_attendance_summary()
        print(f"📊 Attendance summary: {attendance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector face matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_face_matcher()
    if success:
        print("✅ Vector face matcher test passed!")
    else:
        print("❌ Vector face matcher test failed!")

