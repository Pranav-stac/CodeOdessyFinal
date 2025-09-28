#!/usr/bin/env python3
"""
Vision LLM Classifier for Lecture Type Detection
Uses a proper vision-language model to classify classroom scenes from video frames
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import torch

# Try to import vision transformers
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    VISION_LLM_AVAILABLE = True
    print("✅ Vision LLM available for frame-based classification")
except (ImportError, ModuleNotFoundError) as e:
    VISION_LLM_AVAILABLE = False
    print(f"❌ Vision LLM not available: {e}")
    print("❌ Please install: pip install transformers[vision] torch torchvision")

class VisionLLMClassifier:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the vision LLM classifier
        
        Args:
            model_name (str): Hugging Face model name for vision-language model
        """
        self.model_name = model_name
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
        
        # Initialize the vision model
        if VISION_LLM_AVAILABLE:
            self.initialize_vision_model()
        else:
            raise RuntimeError("Vision LLM not available - required for lecture classification")
    
    def initialize_vision_model(self):
        """Initialize the vision-language model"""
        try:
            print(f"🤖 Loading vision model: {self.model_name}")
            
            # Try BLIP model first (good for image captioning)
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✅ BLIP model loaded successfully")
            except Exception as e1:
                print(f"⚠️ BLIP model failed: {e1}")
                # Try CLIP model as fallback
                try:
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    print("✅ CLIP model loaded successfully")
                except Exception as e2:
                    print(f"❌ CLIP model also failed: {e2}")
                    raise e2
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to CUDA")
            
            self.vision_llm_available = True
            print("✅ Vision model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            self.vision_llm_available = False
            raise RuntimeError(f"Failed to load vision model: {e}")
    
    def classify_frame_with_vision_llm(self, frame):
        """
        Classify lecture type using vision LLM
        
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
            
            # Check model type and process accordingly
            if "blip" in self.model_name.lower():
                return self._classify_with_blip(pil_image)
            elif "clip" in str(type(self.model)).lower():
                return self._classify_with_clip(pil_image)
            else:
                raise RuntimeError("Unknown model type")
            
        except Exception as e:
            print(f"❌ Error in vision LLM classification: {e}")
            raise RuntimeError(f"Vision LLM classification failed: {e}")
    
    def _classify_with_blip(self, image):
        """Classify using BLIP model"""
        try:
            # Generate caption for the image
            inputs = self.processor(image, return_tensors="pt")
            
            # Move to device if CUDA is available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=3)
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Classify based on caption
            lecture_type, confidence = self._classify_from_caption(caption)
            
            return lecture_type, confidence
            
        except Exception as e:
            print(f"❌ Error in BLIP classification: {e}")
            return "lecture", 0.5
    
    def _classify_with_clip(self, image):
        """Classify using CLIP model"""
        try:
            # Prepare text prompts for each lecture type
            text_prompts = [
                "a traditional lecture with teacher speaking and students listening",
                "a group discussion or interactive session",
                "a student or teacher presentation",
                "students reading or writing activity",
                "hands-on practical work or lab session",
                "a disorganized or chaotic classroom environment"
            ]
            
            # Process image and text
            inputs = self.processor(
                text=text_prompts,
                images=image,
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
            print(f"❌ Error in CLIP classification: {e}")
            return "lecture", 0.5
    
    def _classify_from_caption(self, caption):
        """Classify lecture type from image caption"""
        caption_lower = caption.lower()
        
        # Define keywords for each lecture type
        keywords = {
            "lecture": ["teacher", "speaking", "lecture", "explaining", "board", "presentation", "classroom", "students listening"],
            "discussion": ["discussion", "talking", "group", "conversation", "interaction", "debate"],
            "presentation": ["presentation", "slides", "screen", "projector", "presenting", "showing"],
            "reading_writing": ["reading", "writing", "books", "papers", "notebooks", "pens", "text"],
            "practical": ["computer", "lab", "equipment", "tools", "hands-on", "experiment", "working"],
            "chaos": ["noise", "movement", "disorganized", "chaos", "confusion", "messy", "crowded"]
        }
        
        # Calculate scores for each lecture type
        scores = {}
        for lecture_type, words in keywords.items():
            score = sum(1 for word in words if word in caption_lower)
            scores[lecture_type] = score
        
        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # Calculate confidence based on keyword matches
            total_keywords = len(keywords[best_type])
            confidence = min(best_score / total_keywords, 1.0) if total_keywords > 0 else 0.5
            
            return best_type, confidence
        else:
            return "lecture", 0.5
    
    def classify_video_frame(self, video_path, frame_time=0.5):
        """
        Classify a single frame from video using vision LLM
        
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
            lecture_type, confidence = self.classify_frame_with_vision_llm(frame)
            
            return {
                "lecture_type": lecture_type,
                "confidence": confidence,
                "method": "vision_llm",
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

# Test the vision LLM classifier
if __name__ == "__main__":
    print("🧪 Testing Vision LLM Classifier")
    print("=" * 50)
    
    try:
        classifier = VisionLLMClassifier()
        print("✅ Vision LLM classifier initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        lecture_type, confidence = classifier.classify_frame_with_vision_llm(dummy_frame)
        print(f"📊 Classification: {lecture_type} (confidence: {confidence:.3f})")
        
        print("✅ Vision LLM classifier test completed!")
        
    except Exception as e:
        print(f"❌ Vision LLM classifier test failed: {e}")
        print("❌ Vision LLM is required for lecture classification!")

