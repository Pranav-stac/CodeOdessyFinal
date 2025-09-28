"""
Lecture Type Classifier using Local Vision LLM
Classifies classroom videos into different lecture types (lecture, group discussion, etc.)
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import base64
from PIL import Image
import io

# Try to import transformers for local LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for LLM-based classification")
except (ImportError, ModuleNotFoundError) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ Transformers not fully available: {e}")
    print("📝 Will use rule-based classification instead")

# Try to import alternative lightweight models
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for deep learning models")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

class LectureClassifier:
    def __init__(self, model_name="microsoft/DialoGPT-medium", use_local_model=True):
        """
        Initialize the lecture classifier
        
        Args:
            model_name (str): Name of the model to use
            use_local_model (bool): Whether to use local model or fallback to rule-based
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Lecture type categories
        self.lecture_types = {
            'lecture': {
                'description': 'Traditional lecture with instructor presenting to students',
                'keywords': ['lecture', 'presentation', 'instructor', 'teaching', 'whiteboard', 'slides'],
                'visual_cues': ['instructor_standing', 'students_listening', 'presentation_screen', 'whiteboard']
            },
            'group_discussion': {
                'description': 'Interactive group discussion or collaborative learning',
                'keywords': ['discussion', 'group', 'collaborative', 'interactive', 'debate'],
                'visual_cues': ['students_facing_each_other', 'multiple_groups', 'hands_raised', 'interaction']
            },
            'hands_on_activity': {
                'description': 'Practical hands-on activity or lab work',
                'keywords': ['hands-on', 'practical', 'lab', 'experiment', 'activity', 'workshop'],
                'visual_cues': ['students_working', 'materials', 'tools', 'practical_work']
            },
            'presentation': {
                'description': 'Student presentations or demonstrations',
                'keywords': ['presentation', 'student_presenting', 'demonstration', 'showcase'],
                'visual_cues': ['student_standing', 'presentation_screen', 'audience_facing_presenter']
            },
            'question_answer': {
                'description': 'Q&A session or interactive questioning',
                'keywords': ['question', 'answer', 'q&a', 'interactive', 'questioning'],
                'visual_cues': ['hands_raised', 'instructor_asking', 'students_responding']
            },
            'reading_writing': {
                'description': 'Silent reading or writing activity',
                'keywords': ['reading', 'writing', 'silent', 'individual', 'study'],
                'visual_cues': ['students_reading', 'students_writing', 'quiet_activity']
            }
        }
        
        # Initialize model if available
        if use_local_model and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self.initialize_model()
        else:
            print("📝 Using rule-based classification (LLM model not loaded)")
    
    def initialize_model(self):
        """Initialize the local LLM model"""
        try:
            print(f"🤖 Loading model: {self.model_name}")
            
            # Use a lightweight model suitable for text classification
            # Note: For vision tasks, we'd need a vision-language model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📝 Falling back to rule-based classification")
            self.model = None
            self.tokenizer = None
    
    def extract_video_features(self, video_path: str, sample_frames: int = 10) -> Dict:
        """
        Extract features from video for classification
        
        Args:
            video_path (str): Path to video file
            sample_frames (int): Number of frames to sample for analysis
            
        Returns:
            dict: Extracted video features
        """
        print(f"🎬 Extracting features from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        features = {
            'video_info': {
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'total_frames': total_frames
            },
            'frame_features': [],
            'activity_patterns': {
                'movement_level': 0.0,
                'interaction_level': 0.0,
                'presentation_elements': 0.0,
                'group_activities': 0.0
            }
        }
        
        prev_frame = None
        movement_scores = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract frame features
            frame_features = self.extract_frame_features(frame, i)
            features['frame_features'].append(frame_features)
            
            # Calculate movement between frames
            if prev_frame is not None:
                movement = self.calculate_movement(prev_frame, frame)
                movement_scores.append(movement)
            
            prev_frame = frame.copy()
        
        # Calculate overall activity patterns
        if movement_scores:
            features['activity_patterns']['movement_level'] = np.mean(movement_scores)
        
        cap.release()
        
        # Aggregate frame features
        self.aggregate_frame_features(features)
        
        return features
    
    def extract_frame_features(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Extract features from a single frame
        
        Args:
            frame (np.ndarray): Video frame
            frame_idx (int): Frame index
            
        Returns:
            dict: Frame features
        """
        height, width = frame.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Basic image statistics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge detection for activity level
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Color analysis
        color_variance = np.var(frame.reshape(-1, 3), axis=0).mean()
        
        # Detect potential presentation elements (white/light areas)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_ratio = np.sum(white_mask > 0) / (height * width)
        
        # Detect potential screens/projectors (rectangular bright areas)
        screen_score = self.detect_screen_elements(gray)
        
        return {
            'frame_idx': frame_idx,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'color_variance': float(color_variance),
            'white_ratio': float(white_ratio),
            'screen_score': float(screen_score)
        }
    
    def detect_screen_elements(self, gray_frame: np.ndarray) -> float:
        """
        Detect potential screen/projection elements in frame
        
        Args:
            gray_frame (np.ndarray): Grayscale frame
            
        Returns:
            float: Screen detection score (0-1)
        """
        # Find bright rectangular regions (potential screens)
        _, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        screen_score = 0.0
        frame_area = gray_frame.shape[0] * gray_frame.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check if contour is large enough and roughly rectangular
            if area > frame_area * 0.05:  # At least 5% of frame
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Screen-like aspect ratios (wide rectangles)
                if 1.2 <= aspect_ratio <= 3.0:
                    screen_score += area / frame_area
        
        return min(screen_score, 1.0)
    
    def calculate_movement(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate movement between two frames
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            float: Movement score (0-1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate movement score
        movement_pixels = np.sum(diff > 30)  # Threshold for significant movement
        total_pixels = diff.shape[0] * diff.shape[1]
        
        return movement_pixels / total_pixels
    
    def aggregate_frame_features(self, features: Dict):
        """
        Aggregate frame features to get overall video characteristics
        
        Args:
            features (dict): Video features with frame_features
        """
        frame_features = features['frame_features']
        
        if not frame_features:
            return
        
        # Calculate averages
        avg_brightness = np.mean([f['brightness'] for f in frame_features])
        avg_contrast = np.mean([f['contrast'] for f in frame_features])
        avg_edge_density = np.mean([f['edge_density'] for f in frame_features])
        avg_white_ratio = np.mean([f['white_ratio'] for f in frame_features])
        avg_screen_score = np.mean([f['screen_score'] for f in frame_features])
        
        # Update activity patterns based on aggregated features
        features['activity_patterns'].update({
            'interaction_level': avg_edge_density,
            'presentation_elements': avg_screen_score + avg_white_ratio,
            'group_activities': min(avg_contrast / 50.0, 1.0)  # Higher contrast suggests more people
        })
    
    def classify_with_llm(self, features: Dict, video_description: str = "") -> Dict:
        """
        Classify lecture type using LLM (if available)
        
        Args:
            features (dict): Video features
            video_description (str): Additional description
            
        Returns:
            dict: Classification results
        """
        if not self.model or not self.tokenizer:
            return self.classify_with_rules(features)
        
        try:
            # Create prompt for classification
            prompt = self.create_classification_prompt(features, video_description)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse LLM response
            classification = self.parse_llm_response(response)
            
            return classification
            
        except Exception as e:
            print(f"❌ Error in LLM classification: {e}")
            return self.classify_with_rules(features)
    
    def create_classification_prompt(self, features: Dict, video_description: str) -> str:
        """
        Create prompt for LLM classification
        
        Args:
            features (dict): Video features
            video_description (str): Additional description
            
        Returns:
            str: Classification prompt
        """
        video_info = features.get('video_info', {})
        activity_patterns = features.get('activity_patterns', {})
        
        prompt = f"""
        Classify this classroom video into one of these lecture types:
        
        1. lecture - Traditional lecture with instructor presenting
        2. group_discussion - Interactive group discussion
        3. hands_on_activity - Practical hands-on activity or lab work
        4. presentation - Student presentations or demonstrations
        5. question_answer - Q&A session or interactive questioning
        6. reading_writing - Silent reading or writing activity
        
        Video characteristics:
        - Duration: {video_info.get('duration', 0):.1f} seconds
        - Movement level: {activity_patterns.get('movement_level', 0):.3f}
        - Interaction level: {activity_patterns.get('interaction_level', 0):.3f}
        - Presentation elements: {activity_patterns.get('presentation_elements', 0):.3f}
        - Group activities: {activity_patterns.get('group_activities', 0):.3f}
        
        Additional context: {video_description}
        
        Based on these features, classify the video and provide confidence (0-1).
        Format: type: [type], confidence: [score], reasoning: [brief explanation]
        """
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response to extract classification
        
        Args:
            response (str): LLM response
            
        Returns:
            dict: Parsed classification
        """
        # Simple parsing - look for type and confidence
        lines = response.split('\n')
        
        classification_type = 'lecture'  # default
        confidence = 0.5
        reasoning = "LLM classification"
        
        for line in lines:
            line = line.lower().strip()
            
            # Look for type
            for lecture_type in self.lecture_types.keys():
                if lecture_type in line:
                    classification_type = lecture_type
                    break
            
            # Look for confidence
            if 'confidence:' in line:
                try:
                    confidence_str = line.split('confidence:')[1].strip()
                    confidence = float(confidence_str.split()[0])
                except:
                    pass
        
        return {
            'type': classification_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'method': 'llm'
        }
    
    def classify_with_rules(self, features: Dict) -> Dict:
        """
        Classify lecture type using rule-based approach
        
        Args:
            features (dict): Video features
            
        Returns:
            dict: Classification results
        """
        activity_patterns = features.get('activity_patterns', {})
        
        movement_level = activity_patterns.get('movement_level', 0)
        interaction_level = activity_patterns.get('interaction_level', 0)
        presentation_elements = activity_patterns.get('presentation_elements', 0)
        group_activities = activity_patterns.get('group_activities', 0)
        
        # Rule-based classification
        scores = {}
        
        # Lecture classification
        lecture_score = presentation_elements * 0.4 + (1 - movement_level) * 0.3 + (1 - interaction_level) * 0.3
        
        # Group discussion classification
        discussion_score = interaction_level * 0.4 + group_activities * 0.3 + movement_level * 0.3
        
        # Hands-on activity classification
        hands_on_score = movement_level * 0.4 + interaction_level * 0.3 + group_activities * 0.3
        
        # Presentation classification
        presentation_score = presentation_elements * 0.5 + (1 - movement_level) * 0.3 + group_activities * 0.2
        
        # Q&A classification
        qa_score = interaction_level * 0.5 + group_activities * 0.3 + movement_level * 0.2
        
        # Reading/writing classification
        reading_score = (1 - movement_level) * 0.4 + (1 - interaction_level) * 0.4 + (1 - group_activities) * 0.2
        
        scores = {
            'lecture': lecture_score,
            'group_discussion': discussion_score,
            'hands_on_activity': hands_on_score,
            'presentation': presentation_score,
            'question_answer': qa_score,
            'reading_writing': reading_score
        }
        
        # Find best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Generate reasoning
        reasoning = self.generate_reasoning(best_type, activity_patterns, scores)
        
        return {
            'type': best_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'method': 'rule_based',
            'all_scores': scores
        }
    
    def generate_reasoning(self, classification_type: str, activity_patterns: Dict, scores: Dict) -> str:
        """
        Generate reasoning for classification
        
        Args:
            classification_type (str): Selected classification type
            activity_patterns (dict): Activity pattern scores
            scores (dict): All classification scores
            
        Returns:
            str: Reasoning text
        """
        movement = activity_patterns.get('movement_level', 0)
        interaction = activity_patterns.get('interaction_level', 0)
        presentation = activity_patterns.get('presentation_elements', 0)
        group = activity_patterns.get('group_activities', 0)
        
        reasoning_parts = []
        
        if classification_type == 'lecture':
            if presentation > 0.3:
                reasoning_parts.append("High presentation elements detected")
            if movement < 0.3:
                reasoning_parts.append("Low movement suggests structured presentation")
            if interaction < 0.4:
                reasoning_parts.append("Low interaction indicates one-way communication")
        
        elif classification_type == 'group_discussion':
            if interaction > 0.4:
                reasoning_parts.append("High interaction level detected")
            if group > 0.3:
                reasoning_parts.append("Group activities observed")
            if movement > 0.2:
                reasoning_parts.append("Active movement suggests discussion")
        
        elif classification_type == 'hands_on_activity':
            if movement > 0.4:
                reasoning_parts.append("High movement indicates hands-on work")
            if interaction > 0.3:
                reasoning_parts.append("Active interaction suggests collaborative work")
        
        elif classification_type == 'presentation':
            if presentation > 0.4:
                reasoning_parts.append("Strong presentation elements detected")
            if group > 0.2:
                reasoning_parts.append("Audience engagement observed")
        
        elif classification_type == 'question_answer':
            if interaction > 0.5:
                reasoning_parts.append("Very high interaction level")
            if group > 0.3:
                reasoning_parts.append("Group engagement patterns")
        
        elif classification_type == 'reading_writing':
            if movement < 0.2:
                reasoning_parts.append("Low movement suggests quiet activity")
            if interaction < 0.3:
                reasoning_parts.append("Low interaction indicates individual work")
        
        if not reasoning_parts:
            reasoning_parts.append("Based on overall activity patterns")
        
        return "; ".join(reasoning_parts)
    
    def classify_video(self, video_path: str, video_description: str = "") -> Dict:
        """
        Main method to classify a video
        
        Args:
            video_path (str): Path to video file
            video_description (str): Additional description for classification
            
        Returns:
            dict: Classification results
        """
        print(f"🎯 Classifying video: {video_path}")
        
        try:
            # Extract video features
            features = self.extract_video_features(video_path)
            
            # Classify using available method
            if self.use_local_model and self.model:
                classification = self.classify_with_llm(features, video_description)
            else:
                classification = self.classify_with_rules(features)
            
            # Add metadata
            classification.update({
                'video_path': video_path,
                'classified_at': datetime.now().isoformat(),
                'video_features': features,
                'lecture_type_info': self.lecture_types.get(classification['type'], {})
            })
            
            print(f"✅ Classification: {classification['type']} (confidence: {classification['confidence']:.3f})")
            
            return classification
            
        except Exception as e:
            print(f"❌ Error classifying video: {e}")
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'reasoning': f"Classification failed: {str(e)}",
                'method': 'error',
                'video_path': video_path,
                'classified_at': datetime.now().isoformat()
            }
    
    def batch_classify_videos(self, video_paths: List[str]) -> List[Dict]:
        """
        Classify multiple videos
        
        Args:
            video_paths (List[str]): List of video file paths
            
        Returns:
            List[Dict]: List of classification results
        """
        results = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"🎬 Processing video {i}/{len(video_paths)}: {os.path.basename(video_path)}")
            
            classification = self.classify_video(video_path)
            results.append(classification)
        
        return results
    
    def save_classification_results(self, results: List[Dict], output_file: str = None):
        """
        Save classification results to file
        
        Args:
            results (List[Dict]): Classification results
            output_file (str): Output file path
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"lecture_classifications_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Classification results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Test the lecture classifier"""
    classifier = LectureClassifier()
    
    print("🎓 Lecture Type Classifier Test")
    print("=" * 40)
    
    # Test with a sample video (if available)
    test_video = "Activity.mp4"  # Replace with actual video path
    
    if os.path.exists(test_video):
        result = classifier.classify_video(test_video)
        print(f"📊 Classification Result:")
        print(f"   Type: {result['type']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Method: {result['method']}")
        print(f"   Reasoning: {result['reasoning']}")
    else:
        print(f"⚠️ Test video not found: {test_video}")
        print("💡 Place a video file named 'Activity.mp4' in the current directory to test")

if __name__ == "__main__":
    main()
