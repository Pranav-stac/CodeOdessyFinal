"""
Complete system test for all new features
"""

import os
import json
import cv2
from video_face_matcher import VideoFaceMatcher
from vision_lecture_classifier import VisionLectureClassifier
from data_manager import DataManager

def test_face_matching():
    """Test face matching system"""
    print("🧪 Testing Face Matching System...")
    
    matcher = VideoFaceMatcher()
    
    # Test with available face images
    face_dir = "realtime_analysis/face_images"
    if not os.path.exists(face_dir):
        print("❌ Face images directory not found")
        return False
    
    face_files = [f for f in os.listdir(face_dir) if f.endswith("_best.jpg")][:10]
    print(f"📸 Testing with {len(face_files)} face images")
    
    successful_encodings = 0
    for face_file in face_files:
        face_path = os.path.join(face_dir, face_file)
        img = cv2.imread(face_path)
        if img is not None:
            encoding = matcher.extract_face_encoding(img)
            if encoding is not None:
                successful_encodings += 1
                print(f"  ✅ {face_file}: Encoding extracted")
            else:
                print(f"  ⚠️ {face_file}: No encoding")
    
    print(f"📊 Success rate: {successful_encodings}/{len(face_files)} ({successful_encodings/len(face_files)*100:.1f}%)")
    return successful_encodings > 0

def test_vision_classifier():
    """Test vision-based lecture classifier"""
    print("\n🧪 Testing Vision Lecture Classifier...")
    
    classifier = VisionLectureClassifier()
    
    # Test with available video
    test_video = "test.mp4"
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    print(f"🎬 Testing with video: {test_video}")
    result = classifier.classify_video_frame(test_video)
    
    if result:
        print(f"✅ Classification successful:")
        print(f"  Type: {result['lecture_type']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result['method']}")
        return True
    else:
        print("❌ Classification failed")
        return False

def test_data_manager():
    """Test data management system"""
    print("\n🧪 Testing Data Manager...")
    
    # Use test database
    dm = DataManager("test_system.db", "test_analysis_history")
    
    # Test basic operations
    try:
        stats = dm.get_statistics()
        print(f"✅ Statistics: {stats}")
        
        # Test video registration
        video_id = dm.register_video("test_video.mp4", {"duration": 100, "fps": 30})
        if video_id:
            print(f"✅ Video registered: {video_id}")
            
            # Test session creation
            session_id = dm.create_analysis_session(video_id, "test_output", "Test Session")
            if session_id:
                print(f"✅ Session created: {session_id}")
                return True
            else:
                print("❌ Session creation failed")
        else:
            print("❌ Video registration failed")
        
        return False
        
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def test_gui_components():
    """Test GUI component integration"""
    print("\n🧪 Testing GUI Component Integration...")
    
    try:
        from classroom_analyzer_gui import ClassroomAnalyzerGUI
        import tkinter as tk
        
        # Create a test root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Initialize GUI
        gui = ClassroomAnalyzerGUI(root)
        
        # Check if all components are initialized
        components = [
            ("Face Matcher", hasattr(gui, 'face_matcher')),
            ("Lecture Classifier", hasattr(gui, 'lecture_classifier')),
            ("Vision Classifier", hasattr(gui, 'vision_classifier')),
            ("Data Manager", hasattr(gui, 'data_manager')),
            ("Analysis Viewer", hasattr(gui, 'analysis_viewer'))
        ]
        
        all_working = True
        for name, status in components:
            if status:
                print(f"  ✅ {name}: Initialized")
            else:
                print(f"  ❌ {name}: Not initialized")
                all_working = False
        
        # Check buttons
        buttons = [
            ("View Analysis", hasattr(gui, 'view_analysis_button')),
            ("Match Faces", hasattr(gui, 'match_faces_button')),
            ("Classify Lecture", hasattr(gui, 'classify_lecture_button')),
            ("Attendance Report", hasattr(gui, 'attendance_button'))
        ]
        
        for name, status in buttons:
            if status:
                print(f"  ✅ {name} Button: Present")
            else:
                print(f"  ❌ {name} Button: Missing")
                all_working = False
        
        root.destroy()
        return all_working
        
    except Exception as e:
        print(f"❌ GUI component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Complete System Test")
    print("=" * 50)
    
    tests = [
        ("Face Matching", test_face_matching),
        ("Vision Classifier", test_vision_classifier),
        ("Data Manager", test_data_manager),
        ("GUI Components", test_gui_components)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("🎉 All systems are working correctly!")
        print("🚀 Ready to use the Classroom Analyzer!")
    else:
        print("⚠️ Some systems need attention")
        print("💡 Check the failed tests above for details")
    
    # Cleanup
    try:
        if os.path.exists("test_system.db"):
            os.remove("test_system.db")
        if os.path.exists("test_analysis_history"):
            import shutil
            shutil.rmtree("test_analysis_history")
    except:
        pass

if __name__ == "__main__":
    main()

