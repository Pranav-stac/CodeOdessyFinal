#!/usr/bin/env python3
"""
Simple Vision Classifier for Lecture Type Detection
Uses CLIP model to classify classroom scenes from video frames
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import torch

# Try to import CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
    VISION_AVAILABLE = True
    print("✅ CLIP model available for vision-based classification")
except (ImportError, ModuleNotFoundError) as e:
    VISION_AVAILABLE = False
    print(f"❌ CLIP not available: {e}")
    print("❌ Please install: pip install transformers torch torchvision")

class SimpleVisionClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the simple vision classifier
        
        Args:
            model_name (str): Hugging Face model name for CLIP model
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
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
        
        # Text prompts for each lecture type
        self.text_prompts = [
            "a traditional lecture with teacher speaking and students listening",
            "a group discussion or interactive session",
            "a student or teacher presentation",
            "students reading or writing activity",
            "hands-on practical work or lab session",
            "a disorganized or chaotic classroom environment"
        ]
        
        # Initialize the vision model
        if VISION_AVAILABLE:
            self.initialize_vision_model()
        else:
            raise RuntimeError("Vision model not available - required for lecture classification")
    
    def initialize_vision_model(self):
        """Initialize the CLIP model"""
        try:
            print(f"🤖 Loading CLIP model: {self.model_name}")
            
            # Load CLIP model and processor
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to CUDA")
            
            self.vision_available = True
            print("✅ CLIP model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            self.vision_available = False
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def classify_frame_with_vision(self, frame):
        """
        Classify lecture type using CLIP vision model
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (lecture_type, confidence)
        """
        try:
            if not self.vision_available:
                raise RuntimeError("Vision model not available")
            
            # Convert frame to PIL Image
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            pil_image = Image.fromarray(frame_rgb)
            
            # Process image and text prompts
            inputs = self.processor(
                text=self.text_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get best match
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            
            # Map to lecture types
            lecture_type_map = ["lecture", "discussion", "presentation", "reading_writing", "practical", "chaos"]
            lecture_type = lecture_type_map[best_idx]
            
            return lecture_type, confidence
            
        except Exception as e:
            print(f"❌ Error in vision classification: {e}")
            raise RuntimeError(f"Vision classification failed: {e}")
    
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
                "method": "clip_vision",
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

# Test the simple vision classifier
if __name__ == "__main__":
    print("🧪 Testing Simple Vision Classifier")
    print("=" * 50)
    
    try:
        classifier = SimpleVisionClassifier()
        print("✅ Simple vision classifier initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        lecture_type, confidence = classifier.classify_frame_with_vision(dummy_frame)
        print(f"📊 Classification: {lecture_type} (confidence: {confidence:.3f})")
        
        print("✅ Simple vision classifier test completed!")
        
    except Exception as e:
        print(f"❌ Simple vision classifier test failed: {e}")
        print("❌ Vision model is required for lecture classification!")

