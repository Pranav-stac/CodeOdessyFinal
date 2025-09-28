"""
Test face encoding extraction with improved error handling
"""

import cv2
import numpy as np
from video_face_matcher import VideoFaceMatcher

def test_face_encoding():
    """Test face encoding extraction"""
    print("🧪 Testing Face Encoding Extraction...")
    
    # Initialize face matcher
    matcher = VideoFaceMatcher()
    
    # Test with a sample image if available
    test_image_path = "realtime_analysis/face_images/face_1001_best.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found: {test_image_path}")
        print("📝 Creating a dummy test image...")
        
        # Create a dummy face-like image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Add some face-like features
        cv2.rectangle(dummy_image, (30, 30), (70, 70), (255, 255, 255), -1)  # Face area
        cv2.circle(dummy_image, (40, 45), 3, (0, 0, 0), -1)  # Left eye
        cv2.circle(dummy_image, (60, 45), 3, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(dummy_image, (45, 55), (55, 60), (0, 0, 0), -1)  # Nose
        cv2.rectangle(dummy_image, (40, 65), (60, 70), (0, 0, 0), -1)  # Mouth
        
        test_image = dummy_image
        print("✅ Created dummy test image")
    else:
        print(f"📸 Loading test image: {test_image_path}")
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print("❌ Could not load test image")
            return False
    
    print(f"📏 Image shape: {test_image.shape}")
    print(f"📊 Image dtype: {test_image.dtype}")
    
    # Test face encoding extraction
    print("\n🔍 Testing face encoding extraction...")
    encoding = matcher.extract_face_encoding(test_image)
    
    if encoding is not None:
        print(f"✅ Face encoding extracted successfully!")
        print(f"📊 Encoding length: {len(encoding)}")
        print(f"📈 First 5 values: {encoding[:5]}")
        return True
    else:
        print("❌ Face encoding extraction failed")
        return False

if __name__ == "__main__":
    import os
    success = test_face_encoding()
    if success:
        print("\n🎉 Face encoding test passed!")
    else:
        print("\n❌ Face encoding test failed!")

