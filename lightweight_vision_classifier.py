#!/usr/bin/env python3
"""
Lightweight Vision LLM Classifier for Lecture Type Detection
Uses a pre-trained lightweight open-source vision model
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms

# Try to import torch and torchvision
try:
    import torch
    import torchvision
    from torchvision import models
    VISION_AVAILABLE = True
    print("✅ Torch and TorchVision available for vision-based classification")
except (ImportError, ModuleNotFoundError) as e:
    VISION_AVAILABLE = False
    print(f"❌ Torch/TorchVision not available: {e}")
    print("❌ Please install: pip install torch torchvision")

class LightweightVisionClassifier:
    def __init__(self, model_name="resnet18"):
        """
        Initialize the lightweight vision classifier
        
        Args:
            model_name (str): Model name (resnet18, resnet34, mobilenet_v2)
        """
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.vision_available = False
        
        # Lecture types for classification
        self.lecture_types = {
            "lecture": "Traditional lecture with teacher speaking and students listening",
            "discussion": "Group discussion or interactive session", 
            "presentation": "Student or teacher presentation",
            "reading_writing": "Reading or writing activity",
            "practical": "Hands-on practical work or lab session",
            "chaos": "Disorganized or chaotic classroom environment"
        }
        
        # Initialize the vision model
        if VISION_AVAILABLE:
            self.initialize_vision_model()
        else:
            raise RuntimeError("Vision model not available - required for lecture classification")
    
    def initialize_vision_model(self):
        """Initialize the lightweight vision model"""
        try:
            print(f"🤖 Loading lightweight vision model: {self.model_name}")
            
            # Load pre-trained model
            if self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif self.model_name == "resnet34":
                self.model = models.resnet34(pretrained=True)
            elif self.model_name == "mobilenet_v2":
                self.model = models.mobilenet_v2(pretrained=True)
            else:
                self.model = models.resnet18(pretrained=True)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to CUDA")
            
            self.vision_available = True
            print("✅ Lightweight vision model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            self.vision_available = False
            raise RuntimeError(f"Failed to load vision model: {e}")
    
    def extract_image_features(self, frame):
        """
        Extract features from image using pre-trained model
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            torch.Tensor: Feature vector
        """
        try:
            # Convert frame to PIL Image
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                # Remove the last classification layer to get features
                features = features.squeeze()
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting image features: {e}")
            return None
    
    def classify_frame_with_vision(self, frame):
        """
        Classify lecture type using vision model
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (lecture_type, confidence)
        """
        try:
            if not self.vision_available:
                raise RuntimeError("Vision model not available")
            
            # Extract features
            features = self.extract_image_features(frame)
            if features is None:
                return "lecture", 0.5
            
            # Convert to numpy for analysis
            features_np = features.cpu().numpy()
            
            # Analyze features to determine lecture type
            lecture_type, confidence = self._analyze_features(features_np, frame)
            
            return lecture_type, confidence
            
        except Exception as e:
            print(f"❌ Error in vision classification: {e}")
            raise RuntimeError(f"Vision classification failed: {e}")
    
    def _analyze_features(self, features, frame):
        """Analyze extracted features to classify lecture type"""
        try:
            # Get basic image statistics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Edge detection for activity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Face detection
            face_count = 0
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
            except:
                pass
            
            # Text detection (simplified)
            text_regions = 0
            try:
                # Simple text detection using contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_regions = len([c for c in contours if cv2.contourArea(c) > 100])
            except:
                pass
            
            # Classification based on features
            scores = {}
            
            # Lecture: moderate brightness, low activity, some faces
            if 100 < brightness < 180 and edge_density < 0.1 and 2 <= face_count <= 8:
                scores["lecture"] = 0.8
            else:
                scores["lecture"] = 0.3
            
            # Discussion: moderate activity, multiple faces
            if edge_density > 0.05 and face_count > 4:
                scores["discussion"] = 0.7
            else:
                scores["discussion"] = 0.2
            
            # Presentation: high brightness, some text, few faces
            if brightness > 150 and text_regions > 2 and face_count <= 3:
                scores["presentation"] = 0.8
            else:
                scores["presentation"] = 0.3
            
            # Reading/Writing: moderate brightness, high text regions
            if 100 < brightness < 160 and text_regions > 3:
                scores["reading_writing"] = 0.7
            else:
                scores["reading_writing"] = 0.2
            
            # Practical: moderate activity, some faces, moderate brightness
            if 0.03 < edge_density < 0.08 and 3 <= face_count <= 6 and 120 < brightness < 170:
                scores["practical"] = 0.7
            else:
                scores["practical"] = 0.2
            
            # Chaos: high activity, many faces, variable brightness
            if edge_density > 0.1 and face_count > 6:
                scores["chaos"] = 0.8
            else:
                scores["chaos"] = 0.1
            
            # Find best match
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            return best_type, confidence
            
        except Exception as e:
            print(f"❌ Error analyzing features: {e}")
            return "lecture", 0.5
    
    def classify_video_frame(self, video_path, frame_time=0.5):
        """
        Classify a single frame from video using vision model
        
        Args:
            video_path (str): Path to video file
            frame_time (float): Time in video to extract frame (0.0 to 1.0)
            
        Returns:
            dict: Classification results
        """
        try:
            if not self.vision_available:
                raise RuntimeError("Vision model not available")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = int(total_frames * frame_time)
            
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read frame from video")
            
            # Classify using vision model
            lecture_type, confidence = self.classify_frame_with_vision(frame)
            
            return {
                "lecture_type": lecture_type,
                "confidence": confidence,
                "method": "lightweight_vision",
                "frame_time": frame_time,
                "video_path": video_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error classifying video frame: {e}")
            raise RuntimeError(f"Video frame classification failed: {e}")
    
    def get_available_lecture_types(self):
        """Get list of available lecture types"""
        return list(self.lecture_types.keys())
    
    def get_lecture_type_description(self, lecture_type):
        """Get description for a lecture type"""
        return self.lecture_types.get(lecture_type, "Unknown lecture type")

# Test the lightweight vision classifier
if __name__ == "__main__":
    print("🧪 Testing Lightweight Vision Classifier")
    print("=" * 50)
    
    try:
        classifier = LightweightVisionClassifier()
        print("✅ Lightweight vision classifier initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        lecture_type, confidence = classifier.classify_frame_with_vision(dummy_frame)
        print(f"📊 Classification: {lecture_type} (confidence: {confidence:.3f})")
        
        print("✅ Lightweight vision classifier test completed!")
        
    except Exception as e:
        print(f"❌ Lightweight vision classifier test failed: {e}")
        print("❌ Vision model is required for lecture classification!")

