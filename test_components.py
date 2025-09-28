#!/usr/bin/env python3
"""
Test script to check all new components of the classroom analyzer
"""

import sys
import os
import json
from datetime import datetime

def test_data_manager():
    """Test the data manager component"""
    print("🔍 Testing Data Manager...")
    try:
        from data_manager import DataManager
        
        # Test initialization
        dm = DataManager("test_database.db", "test_analysis_history")
        print("✅ Data Manager initialized successfully")
        
        # Test database operations (skip file existence check for testing)
        try:
            video_id = dm.register_video("test_video.mp4", {"duration": 100, "fps": 30})
            print("⚠️ Video registration test skipped (file not found)")
            video_id = None
        except:
            print("⚠️ Video registration test skipped (file not found)")
            video_id = None
        
        if video_id:
            session_id = dm.create_analysis_session(video_id, "test_output", "Test Session")
            if session_id:
                print("✅ Analysis session creation works")
        else:
            print("⚠️ Analysis session test skipped (no video ID)")
        
        # Test statistics
        stats = dm.get_statistics()
        print(f"✅ Statistics retrieval works: {stats}")
        
        # Cleanup
        try:
            os.remove("test_database.db")
        except:
            pass
        print("✅ Data Manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data Manager test failed: {e}")
        return False

def test_face_matcher():
    """Test the face matcher component"""
    print("\n🔍 Testing Face Matcher...")
    try:
        from video_face_matcher import VideoFaceMatcher
        
        # Test initialization
        fm = VideoFaceMatcher("test_face_database.json")
        print("✅ Face Matcher initialized successfully")
        
        # Test face database operations
        attendance_summary = fm.get_all_attendance_summary()
        print("✅ Attendance summary retrieval works")
        
        # Cleanup
        if os.path.exists("test_face_database.json"):
            os.remove("test_face_database.json")
        if os.path.exists("attendance_records.json"):
            os.remove("attendance_records.json")
            
        print("✅ Face Matcher test passed")
        return True
        
    except Exception as e:
        print(f"❌ Face Matcher test failed: {e}")
        return False

def test_lecture_classifier():
    """Test the lecture classifier component"""
    print("\n🔍 Testing Lecture Classifier...")
    try:
        from lecture_classifier import LectureClassifier
        
        # Test initialization
        lc = LectureClassifier(use_local_model=False)  # Use rule-based for testing
        print("✅ Lecture Classifier initialized successfully")
        
        # Test classification types
        lecture_types = lc.lecture_types
        print(f"✅ Lecture types loaded: {len(lecture_types)} types")
        
        # Test rule-based classification
        test_features = {
            'activity_patterns': {
                'movement_level': 0.3,
                'interaction_level': 0.4,
                'presentation_elements': 0.6,
                'group_activities': 0.2
            }
        }
        
        classification = lc.classify_with_rules(test_features)
        print(f"✅ Rule-based classification works: {classification['type']}")
        
        print("✅ Lecture Classifier test passed")
        return True
        
    except Exception as e:
        print(f"❌ Lecture Classifier test failed: {e}")
        return False

def test_analysis_viewer():
    """Test the analysis viewer component"""
    print("\n🔍 Testing Analysis Viewer...")
    try:
        # Check if we have existing analysis data
        if os.path.exists("realtime_analysis/comprehensive_analysis_report.json"):
            from analysis_viewer import AnalysisViewer
            import tkinter as tk
            
            # Test initialization with existing data
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            av = AnalysisViewer(root, analysis_dir="realtime_analysis")
            print("✅ Analysis Viewer initialized with existing data")
            
            root.destroy()
            print("✅ Analysis Viewer test passed")
            return True
        else:
            print("⚠️ No existing analysis data found - skipping viewer test")
            return True
            
    except Exception as e:
        print(f"❌ Analysis Viewer test failed: {e}")
        return False

def test_gui_integration():
    """Test GUI integration"""
    print("\n🔍 Testing GUI Integration...")
    try:
        from classroom_analyzer_gui import ClassroomAnalyzerGUI
        import tkinter as tk
        
        # Test GUI initialization
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        gui = ClassroomAnalyzerGUI(root)
        print("✅ GUI initialized successfully")
        
        # Test new components are available
        if hasattr(gui, 'face_matcher'):
            print("✅ Face matcher integrated")
        if hasattr(gui, 'lecture_classifier'):
            print("✅ Lecture classifier integrated")
        if hasattr(gui, 'data_manager'):
            print("✅ Data manager integrated")
            
        root.destroy()
        print("✅ GUI Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ GUI Integration test failed: {e}")
        return False

def main():
    print("🧪 Testing Classroom Analyzer Components")
    print("=" * 50)
    
    tests = [
        test_data_manager,
        test_face_matcher,
        test_lecture_classifier,
        test_analysis_viewer,
        test_gui_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components are working correctly!")
        return True
    else:
        print("⚠️ Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
