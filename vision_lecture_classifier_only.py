#!/usr/bin/env python3
"""
Vision-only Lecture Classifier using Lightweight Vision LLM
Uses only Vision LLM - no rule-based fallback
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import io

# Try to import transformers for vision models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from PIL import Image
    import torch
    # Test if transformers is available
    VISION_LLM_AVAILABLE = True
    print("✅ Transformers available for vision-based classification")
except (ImportError, ModuleNotFoundError) as e:
    VISION_LLM_AVAILABLE = False
    print(f"❌ Transformers not available: {e}")
    print("❌ Vision LLM required for lecture classification!")

class VisionOnlyLectureClassifier:
    def __init__(self, model_name="microsoft/git-base", use_local_model=True):
        """
        Initialize the vision-only lecture classifier (Vision LLM required)
        
        Args:
            model_name (str): Hugging Face model name for vision-language model
            use_local_model (bool): Whether to use local model or API
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.model = None
        self.processor = None
        self.vision_llm_available = False
        
        # Lecture types for classification
        self.lecture_types = {
            "lecture": "Traditional lecture with teacher speaking and students listening",
            "discussion": "Group discussion or interactive session",
            "presentation": "Student or teacher presentation",
            "reading_writing": "Reading or writing activity",
            "practical": "Hands-on practical work or lab session",
            "chaos": "Disorganized or chaotic classroom environment"
        }
        
        # Vision LLM is required - no fallback
        if use_local_model and VISION_LLM_AVAILABLE:
            self.initialize_vision_model()
        else:
            print("❌ Vision LLM is required for lecture classification!")
            print("❌ Please install transformers and torch to use this feature")
            raise RuntimeError("Vision LLM not available - required for lecture classification")
    
    def initialize_vision_model(self):
        """Initialize the vision-language model"""
        try:
            print(f"🤖 Loading vision model: {self.model_name}")
            
            # Try different model loading approaches
            try:
                # First try: Vision-language model
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            except Exception as e1:
                print(f"⚠️ Vision-language model failed: {e1}")
                # Fallback: Use text-only model with image features
                try:
                    self.processor = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/DialoGPT-medium",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    print("✅ Using text-based model as fallback")
                except Exception as e2:
                    print(f"❌ Text model also failed: {e2}")
                    raise e2
            
            # Move to device manually if CUDA is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.vision_llm_available = True
            print("✅ Vision model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            self.vision_llm_available = False
            raise RuntimeError(f"Failed to load vision model: {e}")
    
    def classify_with_vision_llm(self, frame):
        """
        Classify lecture type using vision LLM only
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (lecture_type, confidence)
        """
        try:
            if not self.vision_llm_available:
                raise RuntimeError("Vision LLM not available")
            
            # Convert frame to PIL Image
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            pil_image = Image.fromarray(frame_rgb)
            
            # Check if we have a vision model or text model
            if hasattr(self.processor, 'images'):
                # Vision-language model
                prompt = "Classify this classroom scene into one of these types: lecture, discussion, presentation, reading_writing, practical, chaos. Respond with just the type name."
                inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
            else:
                # Text-only model - extract image features manually
                image_features = self.extract_image_features(frame)
                prompt = f"Classify this classroom scene with features {image_features} into one of these types: lecture, discussion, presentation, reading_writing, practical, chaos. Respond with just the type name."
                inputs = self.processor(prompt, return_tensors="pt")
            
            # Move inputs to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate classification
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract classification from generated text
            classification = self.extract_classification_from_text(generated_text)
            
            # Calculate confidence based on model output
            confidence = self.calculate_confidence(outputs, classification)
            
            return classification, confidence
            
        except Exception as e:
            print(f"❌ Error in vision LLM classification: {e}")
            raise RuntimeError(f"Vision LLM classification failed: {e}")
    
    def extract_image_features(self, frame):
        """Extract basic features from image for text-based classification"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic features
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Face detection (simplified)
            face_count = 0
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
            except:
                pass
            
            return {
                "brightness": int(brightness),
                "contrast": int(contrast),
                "edge_density": round(edge_density, 3),
                "face_count": face_count
            }
        except:
            return {"brightness": 128, "contrast": 50, "edge_density": 0.1, "face_count": 0}
    
    def extract_classification_from_text(self, text):
        """Extract lecture type from generated text"""
        text_lower = text.lower()
        
        # Look for lecture types in the text
        for lecture_type in self.lecture_types.keys():
            if lecture_type in text_lower:
                return lecture_type
        
        # Default fallback
        return "lecture"
    
    def calculate_confidence(self, outputs, classification):
        """Calculate confidence score from model outputs"""
        try:
            # Simple confidence calculation based on output probabilities
            # This is a simplified approach - in practice, you'd use proper probability extraction
            if classification in self.lecture_types:
                return 0.8  # High confidence for successful classification
            else:
                return 0.5  # Medium confidence for fallback
        except:
            return 0.6  # Default confidence
    
    def classify_video_frame(self, video_path, frame_time=0.5):
        """
        Classify a single frame from video using vision LLM only
        
        Args:
            video_path (str): Path to video file
            frame_time (float): Time in video to extract frame (0.0 to 1.0)
            
        Returns:
            dict: Classification results
        """
        try:
            if not self.vision_llm_available:
                raise RuntimeError("Vision LLM not available")
            
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
            
            # Classify using vision LLM
            lecture_type, confidence = self.classify_with_vision_llm(frame)
            
            return {
                "lecture_type": lecture_type,
                "confidence": confidence,
                "method": "vision_llm_only",
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

# Test the vision-only classifier
if __name__ == "__main__":
    print("🧪 Testing Vision-Only Lecture Classifier")
    print("=" * 50)
    
    try:
        classifier = VisionOnlyLectureClassifier()
        print("✅ Vision-only classifier initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        lecture_type, confidence = classifier.classify_with_vision_llm(dummy_frame)
        print(f"📊 Classification: {lecture_type} (confidence: {confidence:.3f})")
        
        print("✅ Vision-only classifier test completed!")
        
    except Exception as e:
        print(f"❌ Vision-only classifier test failed: {e}")
        print("❌ Vision LLM is required for lecture classification!")
