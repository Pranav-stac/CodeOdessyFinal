#!/usr/bin/env python3
"""
Test script to check all dependencies for the advanced classroom analyzer
"""

import sys

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description} - Available")
        return True
    except ImportError as e:
        print(f"❌ {description} - Not available: {e}")
        return False

def main():
    print("🔍 Testing Classroom Analyzer Dependencies")
    print("=" * 50)
    
    # Core libraries
    core_available = True
    core_available &= test_import("cv2", "OpenCV")
    core_available &= test_import("numpy", "NumPy")
    core_available &= test_import("json", "JSON")
    test_import("os", "OS")
    core_available &= test_import("tkinter", "Tkinter GUI")
    core_available &= test_import("PIL", "Pillow (PIL)")
    core_available &= test_import("matplotlib", "Matplotlib")
    
    # AI/ML libraries
    ai_available = True
    ai_available &= test_import("ultralytics", "Ultralytics YOLO")
    ai_available &= test_import("face_recognition", "Face Recognition")
    
    # Optional AI libraries
    transformers_available = test_import("transformers", "Transformers (LLM)")
    torch_available = test_import("torch", "PyTorch")
    
    # Database
    db_available = test_import("sqlite3", "SQLite3")
    
    print("\n📊 Summary:")
    print(f"Core libraries: {'✅' if core_available else '❌'}")
    print(f"AI/ML libraries: {'✅' if ai_available else '❌'}")
    print(f"LLM support: {'✅' if transformers_available and torch_available else '⚠️ (rule-based fallback)'}")
    print(f"Database support: {'✅' if db_available else '❌'}")
    
    if core_available and ai_available and db_available:
        print("\n🎉 All essential dependencies are available!")
        print("✅ Classroom Analyzer should work properly")
        return True
    else:
        print("\n⚠️ Some dependencies are missing")
        print("❌ Please install missing packages")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
