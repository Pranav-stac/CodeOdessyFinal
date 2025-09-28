#!/usr/bin/env python3
"""
Test script to verify all components work before building
"""

import sys
import importlib

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def test_project_modules():
    """Test all project modules"""
    print("🧪 Testing Project Modules...")
    print("=" * 40)
    
    modules = [
        ("realtime_classroom_analyzer", "Real-time Classroom Analyzer"),
        ("analysis_viewer", "Analysis Viewer"),
        ("video_face_matcher", "Video Face Matcher"),
        ("vector_face_matcher", "Vector Face Matcher"),
        ("lecture_classifier", "Lecture Classifier"),
        ("lightweight_vision_classifier", "Lightweight Vision Classifier"),
        ("data_manager", "Data Manager"),
        ("automated_video_processor", "Automated Video Processor"),
        ("firebase_sync", "Firebase Sync"),
        ("classroom_analyzer_gui", "Main GUI"),
    ]
    
    success_count = 0
    total_count = len(modules)
    
    for module_name, description in modules:
        if test_import(module_name, description):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} modules imported successfully")
    return success_count == total_count

def test_dependencies():
    """Test all external dependencies"""
    print("\n🔧 Testing Dependencies...")
    print("=" * 40)
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("ultralytics", "Ultralytics YOLO"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("face_recognition", "Face Recognition"),
        ("firebase_admin", "Firebase Admin"),
        ("tkinter", "Tkinter"),
        ("matplotlib", "Matplotlib"),
        ("json", "JSON"),
        ("base64", "Base64"),
        ("pickle", "Pickle"),
        ("sqlite3", "SQLite3"),
        ("threading", "Threading"),
        ("datetime", "DateTime"),
        ("pathlib", "Pathlib"),
        ("shutil", "Shutil"),
        ("io", "IO"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, description in dependencies:
        if test_import(module_name, description):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} dependencies available")
    return success_count == total_count

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing File Structure...")
    print("=" * 40)
    
    required_files = [
        "classroom_analyzer_gui.py",
        "realtime_classroom_analyzer.py",
        "analysis_viewer.py",
        "video_face_matcher.py",
        "vector_face_matcher.py",
        "lecture_classifier.py",
        "lightweight_vision_classifier.py",
        "data_manager.py",
        "automated_video_processor.py",
        "firebase_sync.py",
        "classroom_labels.json",
        "classroom_icon.ico",
        "classroom_icon.png",
        "firebase_service_account.json",
        "AI_Model_Weights/AI_Model_Weights/yolov8s.pt",
        "AI_Model_Weights/AI_Model_Weights/yolov8n-pose.pt",
        "AI_Model_Weights/AI_Model_Weights/yolov12s-face.pt",
    ]
    
    success_count = 0
    total_count = len(required_files)
    
    for file_path in required_files:
        try:
            with open(file_path, 'r') as f:
                print(f"✅ {file_path}")
                success_count += 1
        except FileNotFoundError:
            print(f"❌ {file_path}")
        except Exception as e:
            print(f"⚠️  {file_path} - {e}")
    
    print(f"\n📊 Results: {success_count}/{total_count} files found")
    return success_count == total_count

def main():
    """Main test function"""
    print("🏗️  Classroom Analyzer - Build Test")
    print("=" * 50)
    
    # Test project modules
    modules_ok = test_project_modules()
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test file structure
    files_ok = test_file_structure()
    
    print("\n" + "=" * 50)
    print("📋 BUILD READINESS SUMMARY")
    print("=" * 50)
    print(f"Project Modules: {'✅ PASS' if modules_ok else '❌ FAIL'}")
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"File Structure: {'✅ PASS' if files_ok else '❌ FAIL'}")
    
    if modules_ok and deps_ok and files_ok:
        print("\n🎉 All tests passed! Ready for PyInstaller build.")
        return True
    else:
        print("\n❌ Some tests failed. Please fix issues before building.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
