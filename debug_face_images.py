#!/usr/bin/env python3
"""
Debug face images to understand why encoding is failing
"""

import json
import base64
import numpy as np
from PIL import Image
import io
import cv2

def debug_face_images():
    """Debug the face images from the analysis data"""
    print("🔍 Debugging face images...")
    
    # Load analysis data
    with open("realtime_analysis/comprehensive_analysis_report.json", 'r') as f:
        analysis_data = json.load(f)
    
    faces = analysis_data.get('faces', {})
    print(f"📊 Total faces in analysis: {len(faces)}")
    
    # Check first few faces
    face_count = 0
    for face_id, face_data in faces.items():
        if face_count >= 5:  # Only check first 5 faces
            break
            
        print(f"\n🔍 Analyzing {face_id}:")
        print(f"   - Keys: {list(face_data.keys())}")
        
        # Check if there's a face_image directly
        if 'face_image' in face_data:
            face_image = face_data['face_image']
            print(f"   - Direct face_image type: {type(face_image)}")
            if isinstance(face_image, list):
                print(f"   - Face image shape: {len(face_image)} elements")
        
        # Check best_image_info
        if 'best_image_info' in face_data:
            best_info = face_data['best_image_info']
            print(f"   - Best image info keys: {list(best_info.keys())}")
            
            if 'base64_image' in best_info:
                try:
                    # Decode base64 image
                    base64_data = best_info['base64_image']
                    image_data = base64.b64decode(base64_data)
                    pil_image = Image.open(io.BytesIO(image_data))
                    face_array = np.array(pil_image)
                    
                    print(f"   - Decoded image shape: {face_array.shape}")
                    print(f"   - Image dtype: {face_array.dtype}")
                    print(f"   - Image min/max: {face_array.min()}/{face_array.max()}")
                    
                    # Check if image is too small
                    height, width = face_array.shape[:2]
                    print(f"   - Image size: {width}x{height}")
                    
                    if height < 50 or width < 50:
                        print(f"   ⚠️ Image is very small!")
                    
                    # Try to detect faces using OpenCV
                    gray = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY) if len(face_array.shape) == 3 else face_array
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)
                    print(f"   - OpenCV faces detected: {len(faces_detected)}")
                    
                except Exception as e:
                    print(f"   ❌ Error decoding image: {e}")
        
        face_count += 1

if __name__ == "__main__":
    debug_face_images()

