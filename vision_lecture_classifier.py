"""
Vision-based Lecture Classifier using Lightweight Vision LLM
Uses a single frame from video to classify lecture type
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
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    # Test if the specific model can be loaded
    try:
        # Try to load a lightweight model to test availability
        test_processor = AutoProcessor.from_pretrained("microsoft/git-base")
        VISION_LLM_AVAILABLE = True
        print("✅ Vision LLM available for frame-based classification")
    except Exception as model_error:
        print(f"⚠️ Vision LLM model loading failed: {model_error}")
        VISION_LLM_AVAILABLE = False
        print("📝 Will use rule-based classification instead")
except (ImportError, ModuleNotFoundError) as e:
    VISION_LLM_AVAILABLE = False
    print(f"⚠️ Vision LLM not available: {e}")
    print("📝 Will use rule-based classification instead")

class VisionLectureClassifier:
    def __init__(self, model_name="microsoft/git-base", use_local_model=True):
        """
        Initialize the vision-based lecture classifier
        
        Args:
            model_name (str): Hugging Face model name for vision-language model
            use_local_model (bool): Whether to use local model or API
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.model = None
        self.processor = None
        
        # Lecture types for classification
        self.lecture_types = {
            "lecture": "Traditional lecture with teacher speaking and students listening",
            "discussion": "Group discussion or interactive session",
            "presentation": "Student or teacher presentation",
            "reading_writing": "Reading or writing activity",
            "practical": "Hands-on practical work or lab session",
            "chaos": "Disorganized or chaotic classroom environment"
        }
        
        # Rule-based classification fallback
        self.rule_based_rules = {
            "lecture": {
                "keywords": ["teacher", "speaking", "lecture", "explaining", "board", "presentation"],
                "min_confidence": 0.6
            },
            "discussion": {
                "keywords": ["discussion", "talking", "group", "conversation", "interaction"],
                "min_confidence": 0.5
            },
            "presentation": {
                "keywords": ["presentation", "slides", "screen", "projector", "presenting"],
                "min_confidence": 0.6
            },
            "reading_writing": {
                "keywords": ["reading", "writing", "books", "papers", "notebooks", "pens"],
                "min_confidence": 0.5
            },
            "practical": {
                "keywords": ["computer", "lab", "equipment", "tools", "hands-on", "experiment"],
                "min_confidence": 0.5
            },
            "chaos": {
                "keywords": ["noise", "movement", "disorganized", "chaos", "confusion"],
                "min_confidence": 0.4
            }
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
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 to avoid device issues
                low_cpu_mem_usage=True
            )
            # Move to device manually if CUDA is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            print("✅ Vision model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            print("📝 Falling back to rule-based classification")
            self.model = None
            self.processor = None
    
    def extract_frame_features(self, frame):
        """
        Extract visual features from a single frame
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            dict: Extracted features
        """
        try:
            # Convert to PIL Image
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            pil_image = Image.fromarray(rgb_frame)
            
            # Basic image analysis
            features = {
                "image_size": pil_image.size,
                "brightness": np.mean(rgb_frame),
                "contrast": np.std(rgb_frame),
                "dominant_colors": self.get_dominant_colors(rgb_frame),
                "face_count": self.count_faces_in_frame(frame),
                "text_regions": self.detect_text_regions(frame),
                "movement_indicators": self.detect_movement_indicators(frame)
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting frame features: {e}")
            return {}
    
    def get_dominant_colors(self, image, k=5):
        """Get dominant colors in the image"""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
            
        except Exception as e:
            print(f"⚠️ Could not extract dominant colors: {e}")
            return []
    
    def count_faces_in_frame(self, frame):
        """Count faces in the frame"""
        try:
            import face_recognition
            face_locations = face_recognition.face_locations(frame)
            return len(face_locations)
        except Exception as e:
            print(f"⚠️ Could not count faces: {e}")
            return 0
    
    def detect_text_regions(self, frame):
        """Detect text regions in the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Text-like characteristics
                if area > 100 and 0.1 < aspect_ratio < 10:
                    text_regions.append((x, y, w, h))
            
            return len(text_regions)
            
        except Exception as e:
            print(f"⚠️ Could not detect text regions: {e}")
            return 0
    
    def detect_movement_indicators(self, frame):
        """Detect indicators of movement or activity"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradient magnitude (edge detection)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradient indicates edges/movement
            movement_score = np.mean(gradient_magnitude)
            
            return float(movement_score)
            
        except Exception as e:
            print(f"⚠️ Could not detect movement indicators: {e}")
            return 0.0
    
    def classify_with_vision_llm(self, frame):
        """
        Classify lecture type using vision LLM
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (lecture_type, confidence)
        """
        if not self.model or not self.processor:
            return None, 0.0
        
        try:
            # Convert frame to PIL Image
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            pil_image = Image.fromarray(rgb_frame)
            
            # Prepare prompt for classification
            prompt = "Classify this classroom scene. Choose from: lecture, discussion, presentation, reading_writing, practical, chaos. Describe what you see and then give the classification."
            
            # Process image and text
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract classification from response
            response_lower = response.lower()
            for lecture_type in self.lecture_types.keys():
                if lecture_type in response_lower:
                    # Calculate confidence based on response quality
                    confidence = min(0.9, len(response) / 100)  # Simple confidence metric
                    return lecture_type, confidence
            
            # Default fallback
            return "lecture", 0.5
            
        except Exception as e:
            print(f"❌ Error in vision LLM classification: {e}")
            return None, 0.0
    
    def classify_with_rules(self, frame):
        """
        Classify lecture type using rule-based approach
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (lecture_type, confidence)
        """
        try:
            # Extract features
            features = self.extract_frame_features(frame)
            
            # Calculate scores for each lecture type
            scores = {}
            
            for lecture_type, rules in self.rule_based_rules.items():
                score = 0.0
                total_checks = 0
                
                # Check face count
                face_count = features.get("face_count", 0)
                if lecture_type == "discussion" and face_count > 5:
                    score += 0.3
                elif lecture_type == "lecture" and 3 <= face_count <= 8:
                    score += 0.2
                elif lecture_type == "presentation" and face_count > 2:
                    score += 0.2
                
                total_checks += 1
                
                # Check text regions (for reading/writing)
                text_regions = features.get("text_regions", 0)
                if lecture_type == "reading_writing" and text_regions > 3:
                    score += 0.4
                elif lecture_type == "presentation" and text_regions > 2:
                    score += 0.3
                
                total_checks += 1
                
                # Check movement indicators
                movement = features.get("movement_indicators", 0)
                if lecture_type == "chaos" and movement > 50:
                    score += 0.4
                elif lecture_type == "discussion" and movement > 30:
                    score += 0.3
                elif lecture_type == "lecture" and movement < 20:
                    score += 0.2
                
                total_checks += 1
                
                # Check brightness (for presentations)
                brightness = features.get("brightness", 128)
                if lecture_type == "presentation" and brightness > 150:
                    score += 0.2
                elif lecture_type == "reading_writing" and 100 < brightness < 180:
                    score += 0.1
                
                total_checks += 1
                
                # Normalize score
                if total_checks > 0:
                    scores[lecture_type] = score / total_checks
                else:
                    scores[lecture_type] = 0.0
            
            # Find best match
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # Apply minimum confidence threshold
            if best_score >= self.rule_based_rules[best_type]["min_confidence"]:
                return best_type, best_score
            else:
                # Default to lecture if no clear match
                return "lecture", 0.5
                
        except Exception as e:
            print(f"❌ Error in rule-based classification: {e}")
            return "lecture", 0.5
    
    def classify_video_frame(self, video_path, frame_time=0.5):
        """
        Classify lecture type from a single frame of video
        
        Args:
            video_path (str): Path to video file
            frame_time (float): Time in seconds to extract frame (0.5 = middle of video)
            
        Returns:
            dict: Classification results
        """
        try:
            print(f"🎬 Extracting frame from: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame number for the specified time
            target_frame = int(frame_time * fps) if fps > 0 else total_frames // 2
            target_frame = min(target_frame, total_frames - 1)
            
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Could not read frame at time {frame_time}s")
                cap.release()
                return None
            
            cap.release()
            
            # Try vision LLM first, fallback to rules
            if self.model and self.processor:
                lecture_type, confidence = self.classify_with_vision_llm(frame)
                method = "vision_llm"
            else:
                lecture_type, confidence = self.classify_with_rules(frame)
                method = "rule_based"
            
            # Prepare results
            results = {
                "video_path": video_path,
                "frame_time": frame_time,
                "lecture_type": lecture_type,
                "confidence": confidence,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "video_duration": duration,
                "frame_number": target_frame
            }
            
            print(f"✅ Classification: {lecture_type} (confidence: {confidence:.3f})")
            return results
            
        except Exception as e:
            print(f"❌ Error classifying video frame: {e}")
            return None
    
    def get_lecture_types(self):
        """Get available lecture types"""
        return list(self.lecture_types.keys())
    
    def get_classification_method(self):
        """Get the current classification method"""
        if self.model and self.processor:
            return "vision_llm"
        else:
            return "rule_based"

# Test the classifier
if __name__ == "__main__":
    print("🧪 Testing Vision Lecture Classifier...")
    
    classifier = VisionLectureClassifier()
    print(f"📊 Available lecture types: {classifier.get_lecture_types()}")
    print(f"🔧 Classification method: {classifier.get_classification_method()}")
    
    # Test with a sample video if available
    test_video = "test.mp4"
    if os.path.exists(test_video):
        print(f"\n🎬 Testing with video: {test_video}")
        results = classifier.classify_video_frame(test_video)
        if results:
            print(f"📋 Results: {json.dumps(results, indent=2)}")
    else:
        print(f"⚠️ Test video {test_video} not found")
