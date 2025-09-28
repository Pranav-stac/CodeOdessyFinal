"""
Test script to check analysis data structure
"""

import json
import os
from pathlib import Path

def test_analysis_data():
    """Test the structure of analysis data"""
    print("🧪 Testing Analysis Data Structure")
    print("=" * 40)
    
    # Look for analysis results
    analysis_dirs = [
        "analysis_results",
        "analysis_history/2025-09-28",
        "realtime_analysis"
    ]
    
    for analysis_dir in analysis_dirs:
        if os.path.exists(analysis_dir):
            print(f"📁 Found analysis directory: {analysis_dir}")
            
            # Look for comprehensive analysis report
            comprehensive_file = os.path.join(analysis_dir, "comprehensive_analysis_report.json")
            if os.path.exists(comprehensive_file):
                print(f"✅ Found comprehensive report: {comprehensive_file}")
                
                try:
                    with open(comprehensive_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f"📊 Data keys: {list(data.keys())}")
                    
                    if 'faces' in data:
                        faces = data['faces']
                        print(f"👥 Faces data type: {type(faces)}")
                        print(f"👥 Number of faces: {len(faces)}")
                        
                        if faces:
                            first_face_key = list(faces.keys())[0]
                            first_face_data = faces[first_face_key]
                            print(f"👤 First face key: {first_face_key}")
                            print(f"👤 First face data type: {type(first_face_data)}")
                            print(f"👤 First face data keys: {list(first_face_data.keys()) if isinstance(first_face_data, dict) else 'Not a dict'}")
                            
                            if isinstance(first_face_data, dict) and 'best_image_info' in first_face_data:
                                best_image_info = first_face_data['best_image_info']
                                print(f"🖼️ Best image info type: {type(best_image_info)}")
                                print(f"🖼️ Best image info keys: {list(best_image_info.keys()) if isinstance(best_image_info, dict) else 'Not a dict'}")
                    else:
                        print("⚠️ No 'faces' key found in analysis data")
                        
                except Exception as e:
                    print(f"❌ Error reading analysis data: {e}")
            else:
                print(f"⚠️ No comprehensive report found in {analysis_dir}")
        else:
            print(f"❌ Analysis directory not found: {analysis_dir}")

if __name__ == "__main__":
    test_analysis_data()

