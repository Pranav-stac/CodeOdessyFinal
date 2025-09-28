"""
Test face detection and encoding with various approaches
"""

import cv2
import numpy as np
import face_recognition
import os

def test_face_detection_methods():
    """Test different face detection methods"""
    print("🧪 Testing Face Detection Methods...")
    
    # Test with a real face image
    test_image_path = "realtime_analysis/face_images/face_1001_best.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    # Load image
    img = cv2.imread(test_image_path)
    if img is None:
        print("❌ Could not load image")
        return False
    
    print(f"📏 Image shape: {img.shape}")
    print(f"📊 Image dtype: {img.dtype}")
    
    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Test 1: Face locations detection
    print("\n🔍 Testing face location detection...")
    try:
        face_locations = face_recognition.face_locations(rgb_img)
        print(f"📍 Face locations found: {len(face_locations)}")
        for i, location in enumerate(face_locations):
            print(f"  Face {i+1}: {location}")
    except Exception as e:
        print(f"❌ Face location detection failed: {e}")
    
    # Test 2: Face encodings with different models
    print("\n🔍 Testing face encodings...")
    
    models = ['hog', 'cnn']
    for model in models:
        try:
            print(f"  Testing with {model} model...")
            encodings = face_recognition.face_encodings(rgb_img, model=model)
            print(f"  ✅ {model} model: {len(encodings)} encodings found")
            if encodings:
                print(f"  📊 Encoding length: {len(encodings[0])}")
        except Exception as e:
            print(f"  ❌ {model} model failed: {e}")
    
    # Test 3: Try with resized image
    print("\n🔍 Testing with resized image...")
    try:
        # Resize to larger size
        height, width = rgb_img.shape[:2]
        if height < 200 or width < 200:
            scale = max(200/height, 200/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_img = cv2.resize(rgb_img, (new_width, new_height))
            print(f"📏 Resized from {height}x{width} to {new_height}x{new_width}")
            
            encodings = face_recognition.face_encodings(resized_img)
            print(f"✅ Resized image: {len(encodings)} encodings found")
        else:
            print("📏 Image already large enough")
    except Exception as e:
        print(f"❌ Resized image test failed: {e}")
    
    # Test 4: Try with different face images
    print("\n🔍 Testing with different face images...")
    face_files = [f for f in os.listdir("realtime_analysis/face_images") if f.endswith("_best.jpg")][:5]
    
    for face_file in face_files:
        face_path = os.path.join("realtime_analysis/face_images", face_file)
        try:
            face_img = cv2.imread(face_path)
            if face_img is not None:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)
                print(f"  {face_file}: {len(encodings)} encodings")
        except Exception as e:
            print(f"  {face_file}: Error - {e}")
    
    return True

def create_test_face_image():
    """Create a test face image that should work with face_recognition"""
    print("\n🎨 Creating test face image...")
    
    # Create a larger image with a clear face
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a simple face
    # Face outline
    cv2.ellipse(img, (100, 100), (80, 100), 0, 0, 360, (220, 180, 140), -1)
    
    # Eyes
    cv2.circle(img, (80, 80), 8, (0, 0, 0), -1)
    cv2.circle(img, (120, 80), 8, (0, 0, 0), -1)
    
    # Nose
    cv2.ellipse(img, (100, 100), (5, 15), 0, 0, 360, (200, 160, 120), -1)
    
    # Mouth
    cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite("test_face.jpg", img)
    print("✅ Test face image created: test_face.jpg")
    
    # Test with this image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    print(f"📊 Test face encodings: {len(encodings)}")
    
    return len(encodings) > 0

if __name__ == "__main__":
    print("🧪 Comprehensive Face Detection Test")
    print("=" * 50)
    
    # Test existing images
    success1 = test_face_detection_methods()
    
    # Create and test synthetic face
    success2 = create_test_face_image()
    
    print("\n📊 Test Results:")
    print(f"  Existing images: {'✅' if success1 else '❌'}")
    print(f"  Synthetic face: {'✅' if success2 else '❌'}")
    
    if success2:
        print("\n🎉 Face detection is working with synthetic images!")
        print("💡 The issue might be with the quality/size of real face images")
    else:
        print("\n❌ Face detection is not working properly")

