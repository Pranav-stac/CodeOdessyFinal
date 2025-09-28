#!/usr/bin/env python3
"""
Test the main GUI application
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def test_gui_launch():
    """Test if the GUI launches without errors"""
    print("🚀 Testing GUI Launch...")
    
    try:
        # Import and create GUI
        from classroom_analyzer_gui import ClassroomAnalyzerGUI
        
        root = tk.Tk()
        root.title("Test - Classroom Analyzer")
        root.geometry("800x600")
        
        # Create GUI instance
        app = ClassroomAnalyzerGUI(root)
        
        print("✅ GUI created successfully")
        
        # Test basic functionality
        print("🔍 Testing GUI components...")
        
        # Check if all new buttons exist
        required_buttons = [
            'view_analysis_button',
            'match_faces_button', 
            'classify_lecture_button',
            'attendance_button'
        ]
        
        for button_name in required_buttons:
            if hasattr(app, button_name):
                print(f"✅ {button_name} exists")
            else:
                print(f"❌ {button_name} missing")
                return False
        
        # Check if new components exist
        required_components = [
            'face_matcher',
            'lecture_classifier',
            'data_manager'
        ]
        
        for component_name in required_components:
            if hasattr(app, component_name):
                print(f"✅ {component_name} integrated")
            else:
                print(f"❌ {component_name} missing")
                return False
        
        print("✅ All GUI components present")
        
        # Close GUI
        root.destroy()
        print("✅ GUI test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_viewer():
    """Test the analysis viewer with existing data"""
    print("\n📊 Testing Analysis Viewer...")
    
    try:
        from analysis_viewer import AnalysisViewer
        
        if not os.path.exists("realtime_analysis/comprehensive_analysis_report.json"):
            print("⚠️ No existing analysis data found")
            return True
        
        root = tk.Tk()
        root.title("Test - Analysis Viewer")
        root.geometry("1000x700")
        
        # Create analysis viewer
        av = AnalysisViewer(root, analysis_dir="realtime_analysis")
        
        print("✅ Analysis Viewer created successfully")
        
        # Test data loading
        if av.analysis_data:
            print(f"✅ Analysis data loaded: {len(av.analysis_data.get('students', {}))} students")
        
        root.destroy()
        print("✅ Analysis Viewer test completed")
        return True
        
    except Exception as e:
        print(f"❌ Analysis Viewer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing Classroom Analyzer GUI")
    print("=" * 40)
    
    tests = [
        test_gui_launch,
        test_analysis_viewer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 GUI Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GUI is working correctly!")
        print("✅ Ready to run: python classroom_analyzer_gui.py")
        return True
    else:
        print("⚠️ GUI needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

