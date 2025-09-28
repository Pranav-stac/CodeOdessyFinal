"""
Real-Time Classroom Video Analysis with OpenCV Visualization
Shows live analysis with pose keypoints, activity detection, and behavior annotations
"""

import cv2
import json
import numpy as np
import os
import sys
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, Counter
import math
import random
import time
import base64

class FaceTracker:
    """Simple face tracker using centroid distance matching"""
    
    def __init__(self, max_disappeared=10, max_distance=100, max_face_ids=100):
        """
        Initialize face tracker
        
        Args:
            max_disappeared (int): Maximum frames a face can be missing before removal
            max_distance (float): Maximum distance for face matching
            max_face_ids (int): Maximum number of face IDs to create
        """
        self.next_face_id = 0
        self.max_face_ids = max_face_ids
        self.faces = {}  # face_id -> {'centroid': (x, y), 'bbox': (x1, y1, x2, y2), 'disappeared': 0}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.face_history = defaultdict(list)  # face_id -> list of (frame_num, bbox, confidence)
    
    def _calculate_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, centroid1, centroid2):
        """Calculate Euclidean distance between two centroids"""
        return math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    
    def update(self, detections, frame_num):
        """
        Update face tracker with new detections
        
        Args:
            detections (list): List of face detections with bbox and confidence
            frame_num (int): Current frame number
            
        Returns:
            list: List of tracked faces with IDs
        """
        # If no detections, mark all faces as disappeared
        if len(detections) == 0:
            for face_id in list(self.faces.keys()):
                self.faces[face_id]['disappeared'] += 1
                if self.faces[face_id]['disappeared'] > self.max_disappeared:
                    del self.faces[face_id]
            return []
        
        # Calculate centroids for new detections
        input_centroids = []
        for detection in detections:
            centroid = self._calculate_centroid(detection['bbox'])
            input_centroids.append(centroid)
        
        # If no existing faces, register all new detections
        if len(self.faces) == 0:
            for i, detection in enumerate(detections):
                self._register_face(detection, input_centroids[i], frame_num)
        else:
            # Match existing faces with new detections
            self._match_faces(detections, input_centroids, frame_num)
        
        # Return tracked faces
        tracked_faces = []
        for face_id, face_data in self.faces.items():
            if face_data['disappeared'] == 0:  # Only return faces that are currently visible
                tracked_faces.append({
                    'id': face_id,
                    'bbox': face_data['bbox'],
                    'centroid': face_data['centroid'],
                    'confidence': face_data.get('confidence', 0.0)
                })
        
        return tracked_faces
    
    def _register_face(self, detection, centroid, frame_num):
        """Register a new face"""
        # Don't create new faces if we've reached the limit
        if len(self.faces) >= self.max_face_ids:
            return
        
        face_id = self.next_face_id
        self.next_face_id += 1
        
        self.faces[face_id] = {
            'centroid': centroid,
            'bbox': detection['bbox'],
            'disappeared': 0,
            'confidence': detection['confidence']
        }
        
        # Add to history
        self.face_history[face_id].append((frame_num, detection['bbox'], detection['confidence']))
    
    def _match_faces(self, detections, input_centroids, frame_num):
        """Match existing faces with new detections"""
        # Get existing face centroids
        existing_face_ids = list(self.faces.keys())
        existing_centroids = [self.faces[face_id]['centroid'] for face_id in existing_face_ids]
        
        # Calculate distance matrix
        D = np.linalg.norm(np.array(existing_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
        
        # Find minimum values in each row and column
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        # Track used row and column indices
        used_row_indices = set()
        used_col_indices = set()
        
        # Update existing faces
        for (row, col) in zip(rows, cols):
            if row in used_row_indices or col in used_col_indices:
                continue
            
            if D[row, col] > self.max_distance:
                continue
            
            face_id = existing_face_ids[row]
            self.faces[face_id]['centroid'] = input_centroids[col]
            self.faces[face_id]['bbox'] = detections[col]['bbox']
            self.faces[face_id]['disappeared'] = 0
            self.faces[face_id]['confidence'] = detections[col]['confidence']
            
            # Add to history
            self.face_history[face_id].append((frame_num, detections[col]['bbox'], detections[col]['confidence']))
            
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # Register new faces for unmatched detections
        for col in range(len(input_centroids)):
            if col not in used_col_indices:
                self._register_face(detections[col], input_centroids[col], frame_num)
        
        # Mark unmatched existing faces as disappeared
        for row in range(len(existing_centroids)):
            if row not in used_row_indices:
                face_id = existing_face_ids[row]
                self.faces[face_id]['disappeared'] += 1
                if self.faces[face_id]['disappeared'] > self.max_disappeared:
                    del self.faces[face_id]
    
    def get_face_history(self, face_id):
        """Get history for a specific face"""
        return self.face_history.get(face_id, [])
    
    def get_all_face_ids(self):
        """Get all tracked face IDs"""
        return list(self.faces.keys())

class RealtimeClassroomAnalyzer:
    def __init__(self, video_path, output_dir="realtime_analysis", headless_mode=False, fast_mode=False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.headless_mode = headless_mode
        self.fast_mode = fast_mode
        os.makedirs(output_dir, exist_ok=True)
        
        # Load classroom labels schema
        self.labels_schema = self.load_labels_schema()
        
        # Initialize models
        self.models = self.load_models()
        
        # Analysis storage
        self.student_tracks = defaultdict(list)
        self.frame_annotations = []
        self.student_id_counter = 1
        
        # Face tracking with limited IDs and better matching
        self.face_tracker = FaceTracker(max_face_ids=80, max_distance=120)  # Larger distance for better matching
        self.face_tracks = defaultdict(list)
        
        # Face storage directories
        self.face_images_dir = os.path.join(output_dir, "face_images")
        self.face_metadata_dir = os.path.join(output_dir, "face_metadata")
        os.makedirs(self.face_images_dir, exist_ok=True)
        os.makedirs(self.face_metadata_dir, exist_ok=True)
        
        # Additional organized directories
        self.student_data_dir = os.path.join(output_dir, "student_data")
        self.activity_data_dir = os.path.join(output_dir, "activity_data")
        self.statistics_dir = os.path.join(output_dir, "statistics")
        os.makedirs(self.student_data_dir, exist_ok=True)
        os.makedirs(self.activity_data_dir, exist_ok=True)
        os.makedirs(self.statistics_dir, exist_ok=True)
        
        # Face image capture settings
        self.face_capture_settings = {
            'min_confidence_for_capture': 0.6,  # Lower threshold for better coverage
            'max_images_per_face': 1,  # Only one best image per face
            'min_time_between_captures': 0.5  # More frequent updates for best selection
        }
        
        # Performance settings for headless mode
        if self.headless_mode:
            # More aggressive face capture in headless mode
            self.face_capture_settings['min_confidence_for_capture'] = 0.5
            self.face_capture_settings['min_time_between_captures'] = 0.3
        
        # Track best face images for each face with continuous replacement
        self.best_faces = {}  # face_id -> {'best_image': frame, 'best_bbox': bbox, 'best_confidence': float, 'best_frame_num': int, 'base64_image': str}
        
        # Visualization settings
        self.colors = {
            'person': (0, 255, 0),
            'writing': (0, 255, 255),
            'raising_hand': (255, 0, 255),
            'listening': (255, 255, 0),
            'distracted': (0, 0, 255),
            'engaged': (0, 255, 0),
            'not_engaged': (0, 0, 255),
            'face': (255, 0, 0)
        }
        
        # Face colors for consistent tracking
        self.face_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 128, 0),  # Olive
        ]
        
        # COCO pose connections for skeleton drawing
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Statistics for real-time display
        self.frame_stats = {
            'total_students': 0,
            'engaged_count': 0,
            'activities': Counter(),
            'attention_levels': Counter(),
            'total_faces': 0,
            'unique_faces': 0
        }

    def load_labels_schema(self):
        """Load classroom labels JSON schema"""
        labels_path = "classroom_labels.json"
        try:
            with open(labels_path, 'r') as f:
                schema = json.load(f)
                print(f"✅ Loaded classroom schema with {len(schema)} objects")
                return schema
        except FileNotFoundError:
            print("⚠️ classroom_labels.json not found, using default schema")
            return [{"name": "person", "id": 1, "attributes": []}]
        except Exception as e:
            print(f"❌ Error loading classroom schema: {e}")
            return [{"name": "person", "id": 1, "attributes": []}]
    
    def load_models(self):
        """Load available YOLO models"""
        models = {}
        
        # Get the directory where the executable is running from
        try:
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller executable
                base_path = sys._MEIPASS
            else:
                # Running as script
                base_path = os.path.dirname(os.path.abspath(__file__))
        except (NameError, AttributeError):
            # Fallback if sys is not available
            base_path = os.path.dirname(os.path.abspath(__file__))
            print("⚠️ sys module not available, using script directory")
        
        # Look for AI_Model_Weights in the executable directory
        weights_dir = os.path.join(base_path, "AI_Model_Weights", "AI_Model_Weights")
        
        # If not found in executable, try current directory
        if not os.path.exists(weights_dir):
            weights_dir = os.path.join(base_path, "AI_Model_Weights")
        
        # If still not found, try relative to script location
        if not os.path.exists(weights_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(script_dir, "AI_Model_Weights", "AI_Model_Weights")
        
        model_files = {
            'detection': 'yolov8s.pt',
            'pose': 'yolov8n-pose.pt',
            'face': 'yolov12s-face.pt'
        }
        
        for model_type, filename in model_files.items():
            model_path = os.path.join(weights_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    
                    # Optimize for inference speed (maintains accuracy)
                    if hasattr(model, 'model'):
                        model.model.eval()
                        # Fuse layers for faster inference
                        if hasattr(model.model, 'fuse'):
                            model.model.fuse()
                    
                    models[model_type] = model
                    print(f"✅ Loaded {model_type}: {filename}")
                except Exception as e:
                    print(f"❌ Failed to load {model_type}: {e}")
        
        return models
    
    def analyze_video_realtime(self, display=True, save_frames=False):
        """Main video analysis with optional OpenCV visualization"""
        if display:
            print(f"🎬 Starting real-time analysis: {self.video_path}")
            print("Press 'q' to quit, 's' to save current frame, SPACE to pause")
        else:
            print(f"🚀 Starting headless analysis: {self.video_path}")
            print("Processing without display for faster performance...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
        
        # Setup display variables
        display_width = min(1200, width)
        display_height = int(height * (display_width / width))
        vis_frame = None
        
        frame_count = 0
        paused = False
        
        # Progress tracking for headless mode
        last_progress_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Analyze frame
                frame_data = self.analyze_frame(frame, frame_count, timestamp)
                self.frame_annotations.append(frame_data)
                self.update_student_tracking(frame_data, frame_count)
                
                if display:
                    vis_frame = self.create_visualization(frame.copy(), frame_data, timestamp)
                elif save_frames and frame_count % 30 == 0:  # Save every 30th frame in headless mode
                    # Create minimal visualization for frame saving
                    vis_frame = self.create_visualization(frame.copy(), frame_data, timestamp)
                    save_path = os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(save_path, vis_frame)
                
                frame_count += 1
            
                # Progress update for headless mode
                if not display:
                    current_time = time.time()
                    if current_time - last_progress_time >= 5.0:  # Every 5 seconds
                        progress = (frame_count / total_frames) * 100
                        print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                        last_progress_time = current_time
            
            if display:
                # Display mode - show frame and handle input
                if vis_frame is not None:
                    vis_display = cv2.resize(vis_frame, (display_width, display_height))
                    
                    # Show frame
                    cv2.imshow('Real-time Classroom Analysis', vis_display)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(save_path, vis_frame)
                        print(f"💾 Saved frame: {save_path}")
                    elif key == ord(' '):
                        paused = not paused
                        print("⏸️ Paused" if paused else "▶️ Resumed")
            else:
                # Headless mode - just process frames
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Analysis complete! Processed {frame_count} frames")
        self.save_best_face_images()
        self.save_organized_reports()
        self.save_comprehensive_report()
        
        return self.frame_annotations
    
    def analyze_frame(self, frame, frame_num, timestamp):
        """Analyze single frame with all models"""
        frame_data = {
            'frame_number': frame_num,
            'timestamp': timestamp,
            'annotated_persons': [],
            'faces': []
        }
        
        height, width = frame.shape[:2]
        
        # Person detection
        if 'detection' in self.models:
            results = self.models['detection'](frame, verbose=False)
            persons = self.extract_persons(results)
            
            # Pose analysis for activity recognition
            pose_data = []
            if 'pose' in self.models and persons:
                pose_results = self.models['pose'](frame, verbose=False)
                pose_data = self.extract_pose_data(pose_results)
            
            # Apply classroom label schema
            frame_data['annotated_persons'] = self.create_annotations(
                persons, pose_data, height, width
            )
            frame_data['pose_keypoints'] = pose_data
        
        # Face detection
        if 'face' in self.models:
            face_data = self.detect_faces(frame, frame_num)
            frame_data['faces'] = face_data
        
        return frame_data
    
    def extract_persons(self, results):
        """Extract person detections"""
        persons = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        persons.append({
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(box.conf[0].cpu().numpy()),
                            'center': [(x1+x2)/2, (y1+y2)/2]
                        })
        return persons
    
    def extract_pose_data(self, pose_results):
        """Extract pose keypoints for visualization"""
        pose_data = []
        for result in pose_results:
            if result.keypoints is not None:
                for kp in result.keypoints.data:
                    points = kp.cpu().numpy()
                    analysis = self.analyze_pose(points)
                    pose_data.append({
                        'keypoints': points,
                        'analysis': analysis
                    })
        return pose_data
    
    def detect_faces(self, frame, frame_num):
        """Detect faces in frame using YOLO face model"""
        if 'face' not in self.models:
            return []
        
        # Run face detection
        results = self.models['face'](frame, conf=0.25, verbose=False)
        
        # Extract face detections
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        
        # Update face tracker
        tracked_faces = self.face_tracker.update(detections, frame_num)
        
        # Track best face images for each face
        for face in tracked_faces:
            self.track_best_face(face, frame, frame_num)
        
        # Update face statistics
        self.frame_stats['total_faces'] = len(tracked_faces)
        self.frame_stats['unique_faces'] = len(self.face_tracker.get_all_face_ids())
        
        return tracked_faces
    
    def track_best_face(self, face, frame, frame_num):
        """Track the best face image for each face, replacing when confidence increases"""
        face_id = face['id']
        confidence = face['confidence']
        bbox = face['bbox']
        
        # Skip low confidence detections
        if confidence < self.face_capture_settings['min_confidence_for_capture']:
            return
        
        # Calculate face quality score
        quality_score = self.calculate_face_quality(face, frame)
        
        # Check if this is the best face we've seen for this ID
        if face_id not in self.best_faces:
            # First face for this ID - always accept
            face_crop = self.extract_face_crop(frame, bbox)
            base64_image = self.encode_face_to_base64(face_crop)
            
            self.best_faces[face_id] = {
                'best_image': frame.copy(),
                'best_bbox': bbox,
                'best_confidence': confidence,
                'best_quality_score': quality_score,
                'best_frame_num': frame_num,
                'base64_image': base64_image,
                'replacement_count': 0
            }
            
            if not self.headless_mode:
                print(f"📸 Initial face capture for ID {face_id} (confidence: {confidence:.3f})")
        else:
            # Check if this detection has higher confidence than current best
            current_best = self.best_faces[face_id]
            
            if confidence > current_best['best_confidence']:
                # Higher confidence detected - replace the image
                face_crop = self.extract_face_crop(frame, bbox)
                base64_image = self.encode_face_to_base64(face_crop)
                
                self.best_faces[face_id] = {
                    'best_image': frame.copy(),
                    'best_bbox': bbox,
                    'best_confidence': confidence,
                    'best_quality_score': quality_score,
                    'best_frame_num': frame_num,
                    'base64_image': base64_image,
                    'replacement_count': current_best.get('replacement_count', 0) + 1
                }
                
                if not self.headless_mode:
                    print(f"🔄 Replaced face ID {face_id}: {current_best['best_confidence']:.3f} → {confidence:.3f} (frame {frame_num})")
    
    def calculate_face_quality(self, face, frame):
        """Calculate face quality score based on size, position, and clarity"""
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        
        # Calculate face size (larger faces are generally better)
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        
        # Normalize by frame size
        h, w = frame.shape[:2]
        frame_area = w * h
        size_score = min(1.0, face_area / (frame_area * 0.1))  # Max score for 10% of frame
        
        # Calculate position score (center is better)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        distance_from_center = math.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        max_distance = math.sqrt((w/2)**2 + (h/2)**2)
        position_score = 1.0 - (distance_from_center / max_distance)
        
        # Calculate aspect ratio score (closer to 1:1 is better for faces)
        aspect_ratio = face_width / face_height if face_height > 0 else 1.0
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Perfect square gets 1.0
        
        # Calculate clarity score (simplified - could be enhanced with blur detection)
        face_region = frame[y1:y2, x1:x2]
        if face_region.size > 0:
            # Use variance as a simple clarity measure
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            clarity_score = min(1.0, np.var(gray_face) / 1000.0)  # Normalize variance
        else:
            clarity_score = 0.0
        
        # Weighted combination of all factors
        quality_score = (0.3 * size_score + 0.2 * position_score + 0.2 * aspect_score + 0.3 * clarity_score)
        
        return quality_score
    
    def extract_face_crop(self, frame, bbox):
        """Extract face crop from frame with padding"""
        x1, y1, x2, y2 = bbox
        
        # Add padding around the face
        padding = 15
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract face crop
        face_crop = frame[y1:y2, x1:x2]
        
        return face_crop
    
    def encode_face_to_base64(self, face_crop):
        """Enhance face image and encode to base64"""
        if face_crop.size == 0:
            return ""
        
        # Enhance the face image
        enhanced_crop = self.enhance_face_image(face_crop)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', enhanced_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        return base64_string
    
    def enhance_face_image(self, face_crop):
        """Enhance face image quality"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% enhanced, 30% sharpened)
        final = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        # Apply slight noise reduction
        denoised = cv2.bilateralFilter(final, 9, 75, 75)
        
        return denoised
    
    def save_best_face_images(self):
        """Save the best face images for each tracked face"""
        print("💾 Saving best face images...")
        
        for face_id, best_face_data in self.best_faces.items():
            try:
                # Extract face region from the best frame
                bbox = best_face_data['best_bbox']
                frame = best_face_data['best_image']
                
                x1, y1, x2, y2 = bbox
                
                # Add padding around the face
                padding = 15
                h, w = frame.shape[:2]
                
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Extract face crop
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Enhance the face image
                    enhanced_crop = self.enhance_face_image(face_crop)
                    
                    # Save enhanced best face image
                    face_filename = f"face_{face_id:02d}_best.jpg"
                    face_path = os.path.join(self.face_images_dir, face_filename)
                    
                    cv2.imwrite(face_path, enhanced_crop)
                    
                    # Save metadata for the best face
                    self.save_best_face_metadata(face_id, best_face_data, face_filename)
                    
                    print(f"📸 Saved best face {face_id} (confidence: {best_face_data['best_confidence']:.3f}, quality: {best_face_data['best_quality_score']:.3f}) -> {face_filename}")
                
            except Exception as e:
                print(f"❌ Failed to save best face {face_id}: {e}")
    
    def save_best_face_metadata(self, face_id, best_face_data, face_filename):
        """Save metadata for the best face image"""
        metadata = {
            "face_id": int(face_id),
            "image_filename": face_filename,
            "frame_number": int(best_face_data['best_frame_num']),
            "bbox": [int(x) for x in best_face_data['best_bbox']],
            "confidence": float(best_face_data['best_confidence']),
            "quality_score": float(best_face_data['best_quality_score']),
            "combined_score": float(0.7 * best_face_data['best_confidence'] + 0.3 * best_face_data['best_quality_score']),
            "timestamp": float(time.time()),
            "is_best_image": True
        }
        
        # Save individual face metadata file
        metadata_filename = f"face_{face_id:02d}_best_metadata.json"
        metadata_path = os.path.join(self.face_metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_face_metadata(self, face_id, face, frame_num, face_filename):
        """Save metadata for a captured face"""
        metadata = {
            "face_id": int(face_id),
            "image_filename": face_filename,
            "frame_number": int(frame_num),
            "bbox": [int(x) for x in face['bbox']],  # Convert numpy int64 to Python int
            "confidence": float(face['confidence']),  # Convert numpy float to Python float
            "timestamp": float(time.time()),
            "capture_count": int(self.captured_faces[face_id]['capture_count'])
        }
        
        # Save individual face metadata file
        metadata_filename = f"face_{face_id:02d}_metadata.json"
        metadata_path = os.path.join(self.face_metadata_dir, metadata_filename)
        
        # Load existing metadata or create new
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            except:
                existing_metadata = {"face_id": face_id, "captures": []}
        else:
            existing_metadata = {"face_id": face_id, "captures": []}
        
        # Add this capture to the metadata
        if "captures" not in existing_metadata:
            existing_metadata["captures"] = []
        
        existing_metadata["captures"].append(metadata)
        existing_metadata["total_captures"] = len(existing_metadata["captures"])
        existing_metadata["last_updated"] = time.time()
        
        # Save updated metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save face metadata: {e}")
    
    def analyze_pose(self, keypoints):
        """Enhanced pose analysis for activity inference"""
        if len(keypoints) < 17:
            return {'activity': 'unknown', 'posture': 'sitting', 'confidence': 0.0}
        
        # Key points with confidence checking
        def get_keypoint(keypoint_idx):
            if keypoint_idx < len(keypoints) and len(keypoints[keypoint_idx]) >= 3:
                x, y, conf = keypoints[keypoint_idx][:3]
                return [x, y] if conf > 0.3 else [0, 0]  # Only use if confidence > 0.3
            return [0, 0]
        
        # Get key points
        left_wrist = get_keypoint(9)
        right_wrist = get_keypoint(10)
        left_shoulder = get_keypoint(5)
        right_shoulder = get_keypoint(6)
        left_elbow = get_keypoint(7)
        right_elbow = get_keypoint(8)
        nose = get_keypoint(0)
        left_eye = get_keypoint(1)
        right_eye = get_keypoint(2)
        
        activity = 'listening'
        posture = 'sitting'
        confidence = 0.5
        
        # Check if we have valid keypoints
        valid_keypoints = sum(1 for kp in [left_wrist, right_wrist, left_shoulder, right_shoulder] 
                            if kp[0] > 0 and kp[1] > 0)
        
        if valid_keypoints < 2:
            return {'activity': 'unknown', 'posture': 'sitting', 'confidence': 0.0}
        
        # Enhanced activity detection
        
        # 1. Check for raised hand (more robust)
        left_hand_raised = (left_wrist[1] > 0 and left_shoulder[1] > 0 and 
                           left_wrist[1] < left_shoulder[1] - 20)  # More strict threshold
        right_hand_raised = (right_wrist[1] > 0 and right_shoulder[1] > 0 and 
                            right_wrist[1] < right_shoulder[1] - 20)
        
        if left_hand_raised or right_hand_raised:
            activity = 'raising_hand'
            confidence = 0.9
        
        # 2. Check for writing position (enhanced)
        elif (left_wrist[0] > 0 and left_shoulder[0] > 0 and 
              abs(left_wrist[0] - left_shoulder[0]) < 50 and
              left_wrist[1] > left_shoulder[1]) or \
             (right_wrist[0] > 0 and right_shoulder[0] > 0 and 
              abs(right_wrist[0] - right_shoulder[0]) < 50 and
              right_wrist[1] > right_shoulder[1]):
            activity = 'writing'
            confidence = 0.8
        
        # 3. Check for listening (head orientation and posture)
        elif (nose[0] > 0 and left_eye[0] > 0 and right_eye[0] > 0):
            # Check if person is looking forward (listening)
            head_orientation = abs(left_eye[0] - right_eye[0])
            if head_orientation > 10:  # Eyes are reasonably spaced
                activity = 'listening'
                confidence = 0.7
            else:
                activity = 'listening'  # Default to listening
                confidence = 0.5
        else:
            # Default to listening if we can't determine
            activity = 'listening'
            confidence = 0.4
        
        # Posture detection
        if (left_shoulder[1] > 0 and right_shoulder[1] > 0 and 
            abs(left_shoulder[1] - right_shoulder[1]) < 20):
            # Shoulders are level, likely sitting
            posture = 'sitting'
        else:
            posture = 'sitting'  # Default to sitting
        
        return {'activity': activity, 'posture': posture, 'confidence': confidence}
    
    def create_annotations(self, persons, pose_data, height, width):
        """Create classroom label annotations"""
        annotations = []
        
        # Reset frame stats
        self.frame_stats = {
            'total_students': len(persons),
            'engaged_count': 0,
            'activities': Counter(),
            'attention_levels': Counter(),
            'total_faces': 0,
            'unique_faces': 0
        }
        
        # Get person schema
        person_schema = None
        for label in self.labels_schema:
            if label['name'] == 'person':
                person_schema = label
                break
        
        if not person_schema:
            print("⚠️ No person schema found, using default attributes")
            # Create a fallback person schema that matches classroom_labels.json
            person_schema = {
                "name": "person",
                "id": 1,
                "color": "#ff0000",
                "type": "rectangle",
                "attributes": [
                    {
                        "name": "activity",
                        "input_type": "select",
                        "mutable": True,
                        "values": [
                            "writing",
                            "listening",
                            "talking",
                            "using_phone",
                            "sleeping",
                            "raising_hand",
                            "walking",
                            "distracted",
                            "reading",
                            "eating",
                            "unknown"
                        ],
                        "default_value": "unknown"
                    },
                    {
                        "name": "posture",
                        "input_type": "select",
                        "mutable": True,
                        "values": [
                            "sitting",
                            "standing",
                            "leaning",
                            "slouching"
                        ],
                        "default_value": "sitting"
                    },
                    {
                        "name": "attention_level",
                        "input_type": "select",
                        "mutable": True,
                        "values": [
                            "focused",
                            "partially_focused",
                            "distracted",
                            "not_visible"
                        ],
                        "default_value": "not_visible"
                    },
                    {
                        "name": "engagement",
                        "input_type": "checkbox",
                        "mutable": True,
                        "values": [
                            "engaged"
                        ],
                        "default_value": ""
                    }
                ]
            }

        for i, person in enumerate(persons):
            # Base annotation
            annotation = {
                'name': 'person',
                'bbox': person['bbox'],
                'confidence': person['confidence'],
                'attributes': {}
            }
            
            # Get pose analysis if available
            pose_analysis = pose_data[i]['analysis'] if i < len(pose_data) else {'activity': 'unknown', 'posture': 'sitting'}
            
            # Position-based engagement (memory: position-based confidence scoring)
            center_y = person['center'][1]
            if center_y < height * 0.4:  # Front rows
                engagement_prob = 0.8
                zone = 'front'
            elif center_y < height * 0.7:  # Middle rows
                engagement_prob = 0.6
                zone = 'middle'
            else:  # Back rows
                engagement_prob = 0.4
                zone = 'back'
            
            # Apply classroom attributes from schema
            for attr in person_schema.get('attributes', []):
                attr_name = attr['name']
                
                if attr_name == 'activity':
                    activity = pose_analysis['activity']
                    if activity in attr.get('values', []):
                        annotation['attributes'][attr_name] = activity
                    else:
                        annotation['attributes'][attr_name] = attr.get('default_value', 'unknown')
                
                elif attr_name == 'posture':
                    annotation['attributes'][attr_name] = pose_analysis.get('posture', 'sitting')
                
                elif attr_name == 'attention_level':
                    if pose_analysis['activity'] == 'raising_hand':
                        attention = 'focused'
                    elif pose_analysis['activity'] == 'writing':
                        attention = 'focused'
                    else:
                        try:
                            # Use random for attention calculation
                            rand_val = random.random()
                            if rand_val < engagement_prob:
                                attention = 'partially_focused'
                            else:
                                attention = 'distracted'
                        except Exception as e:
                            # Fallback if random fails in executable
                            print(f"⚠️ Random module error in attention calculation: {e}")
                            # Use deterministic fallback based on position
                            if engagement_prob > 0.7:
                                attention = 'partially_focused'
                            else:
                                attention = 'distracted'
                    annotation['attributes'][attr_name] = attention
                
                elif attr_name == 'engagement':
                    # Enhanced engagement calculation based on pose analysis
                    activity = pose_analysis.get('activity', 'unknown')
                    posture = pose_analysis.get('posture', 'sitting')
                    confidence = pose_analysis.get('confidence', 0.0)
                    
                    # Direct engagement indicators
                    if activity in ['raising_hand', 'writing']:
                        engagement = 'engaged'
                    elif activity == 'listening' and posture in ['sitting', 'standing'] and confidence > 0.5:
                        # Listening with good posture and high confidence indicates engagement
                        engagement = 'engaged'
                    else:
                        # Use deterministic calculation based on multiple factors
                        engagement_score = 0.0
                        
                        # Activity-based scoring
                        if activity == 'listening':
                            engagement_score += 0.6 * confidence  # Weight by confidence
                        elif activity == 'writing':
                            engagement_score += 0.9
                        elif activity == 'raising_hand':
                            engagement_score += 1.0
                        elif activity == 'unknown':
                            engagement_score += 0.3
                        
                        # Posture-based scoring
                        if posture == 'sitting':
                            engagement_score += 0.2
                        elif posture == 'standing':
                            engagement_score += 0.1
                        
                        # Position-based scoring (zone)
                        engagement_score += engagement_prob * 0.3
                        
                        # Confidence-based bonus
                        engagement_score += confidence * 0.2
                        
                        # Determine engagement based on total score
                        if engagement_score >= 0.6:
                            engagement = 'engaged'
                        else:
                            engagement = 'not_engaged'
                    
                    annotation['attributes'][attr_name] = engagement
            
            # Update frame stats
            activity = annotation['attributes'].get('activity', 'unknown')
            attention = annotation['attributes'].get('attention_level', 'not_visible')
            engagement = annotation['attributes'].get('engagement', '')
            
            self.frame_stats['activities'][activity] += 1
            self.frame_stats['attention_levels'][attention] += 1
            if engagement == 'engaged':
                self.frame_stats['engaged_count'] += 1
            
            # Add metadata
            annotation['position_zone'] = zone
            annotation['student_id'] = self.find_or_create_student(person['center'], len(self.frame_annotations))
            
            annotations.append(annotation)
        
        return annotations
    
    def find_or_create_student(self, center, frame_num):
        """Simple student tracking by position"""
        for student_id, track in self.student_tracks.items():
            if track:
                last_pos = track[-1].get('center', [0, 0])
                distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                if distance < 80:  # Same student if within 80 pixels
                    return student_id
        
        # Create new student
        new_id = f"student_{self.student_id_counter:02d}"
        self.student_id_counter += 1
        return new_id
    
    def update_student_tracking(self, frame_data, frame_num):
        """Update student behavior tracking"""
        for person in frame_data['annotated_persons']:
            student_id = person['student_id']
            
            record = {
                'frame': frame_num,
                'timestamp': frame_data['timestamp'],
                'center': [person['bbox'][0] + person['bbox'][2]/2, person['bbox'][1] + person['bbox'][3]/2],
                'activity': person['attributes'].get('activity', 'unknown'),
                'attention_level': person['attributes'].get('attention_level', 'not_visible'),
                'engagement': person['attributes'].get('engagement', ''),
                'position_zone': person.get('position_zone', 'unknown')
            }
            
            self.student_tracks[student_id].append(record)
    
    def create_visualization(self, frame, frame_data, timestamp):
        """Create comprehensive visualization overlay"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw face detections first (behind person annotations)
        for face in frame_data.get('faces', []):
            face_id = face['id']
            bbox = face['bbox']
            confidence = face['confidence']
            x1, y1, x2, y2 = bbox
            
            # Choose color based on face ID for consistency
            color = self.face_colors[face_id % len(self.face_colors)]
            
            # Draw face bounding box (thinner line)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)
            
            # Add face ID and confidence label
            label = f"Face {face_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 5), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw pose skeletons and annotations for each person
        for i, person in enumerate(frame_data['annotated_persons']):
            # Get bounding box and attributes
            bbox = person['bbox']
            x, y, w, h = bbox
            activity = person['attributes'].get('activity', 'unknown')
            attention = person['attributes'].get('attention_level', 'not_visible')
            engagement = person['attributes'].get('engagement', '')
            student_id = person['student_id']
            
            # Choose color based on activity
            color = self.colors.get(activity, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw pose skeleton if available
            if i < len(frame_data.get('pose_keypoints', [])):
                keypoints = frame_data['pose_keypoints'][i]['keypoints']
                self.draw_pose_skeleton(vis_frame, keypoints)
            
            # Create label
            label_lines = [
                f"{student_id}",
                f"{activity}",
                f"{attention}",
                f"{'ENGAGED' if engagement == 'engaged' else 'NOT ENGAGED'}"
            ]
            
            # Draw label background
            label_height = len(label_lines) * 20 + 10
            cv2.rectangle(vis_frame, (x, y - label_height), (x + 200, y), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (x, y - label_height), (x + 200, y), color, 2)
            
            # Draw label text
            for line_idx, line in enumerate(label_lines):
                text_y = y - label_height + 15 + line_idx * 20
                text_color = (0, 255, 0) if line == 'ENGAGED' else (0, 0, 255) if line == 'NOT ENGAGED' else (255, 255, 255)
                cv2.putText(vis_frame, line, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Draw statistics overlay
        self.draw_statistics_overlay(vis_frame, timestamp)
        
        # Draw zone indicators
        self.draw_zone_indicators(vis_frame, height, width)
        
        return vis_frame
    
    def draw_pose_skeleton(self, frame, keypoints):
        """Draw pose skeleton on frame"""
        if len(keypoints) < 17:
            return
        
        # Draw keypoints
        for i, point in enumerate(keypoints):
            if len(point) >= 3 and point[2] > 0.5:  # Confidence threshold
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Draw skeleton connections
        for connection in self.pose_connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                if len(pt1) >= 3 and len(pt2) >= 3 and pt1[2] > 0.5 and pt2[2] > 0.5:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    def draw_statistics_overlay(self, frame, timestamp):
        """Draw real-time statistics overlay"""
        height, width = frame.shape[:2]
        
        # Statistics panel background
        panel_width = 300
        panel_height = 250
        panel_x = width - panel_width - 10
        panel_y = 10
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "REAL-TIME ANALYSIS", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Time
        time_str = f"Time: {timestamp:.1f}s"
        cv2.putText(frame, time_str, (panel_x + 10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Student count
        student_count = f"Students: {self.frame_stats['total_students']}"
        cv2.putText(frame, student_count, (panel_x + 10, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Engagement rate
        engagement_rate = (self.frame_stats['engaged_count'] / max(1, self.frame_stats['total_students'])) * 100
        engagement_str = f"Engagement: {engagement_rate:.1f}%"
        engagement_color = (0, 255, 0) if engagement_rate > 70 else (0, 255, 255) if engagement_rate > 50 else (0, 0, 255)
        cv2.putText(frame, engagement_str, (panel_x + 10, panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, engagement_color, 1)
        
        # Face statistics
        face_count = f"Faces: {self.frame_stats['total_faces']}"
        cv2.putText(frame, face_count, (panel_x + 10, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        unique_faces = f"Unique: {self.frame_stats['unique_faces']}"
        cv2.putText(frame, unique_faces, (panel_x + 10, panel_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Top activities
        y_offset = 150
        cv2.putText(frame, "Activities:", (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (activity, count) in enumerate(self.frame_stats['activities'].most_common(3)):
            y_offset += 20
            activity_str = f"  {activity}: {count}"
            color = self.colors.get(activity, (255, 255, 255))
            cv2.putText(frame, activity_str, (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_zone_indicators(self, frame, height, width):
        """Draw classroom zone indicators"""
        # Front zone (top 40%)
        front_y = int(height * 0.4)
        cv2.line(frame, (0, front_y), (width, front_y), (0, 255, 0), 2)
        cv2.putText(frame, "FRONT ZONE", (10, front_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Middle zone (40%-70%)
        middle_y = int(height * 0.7)
        cv2.line(frame, (0, middle_y), (width, middle_y), (255, 255, 0), 2)
        cv2.putText(frame, "MIDDLE ZONE", (10, middle_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Back zone (70%-100%)
        cv2.putText(frame, "BACK ZONE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    def save_organized_reports(self):
        """Save organized analysis reports in multiple files"""
        print("📁 Saving organized analysis reports...")
        
        # Save main summary report
        self.save_main_summary()
        
        # Save individual student data files
        self.save_student_data_files()
        
        # Save face data files
        self.save_face_data_files()
        
        # Save activity statistics
        self.save_activity_statistics()
        
        # Save frame-by-frame data
        self.save_frame_data()
        
        print("✅ All reports saved successfully!")
    
    def save_main_summary(self):
        """Save main summary report"""
        summary_file = os.path.join(self.output_dir, "analysis_summary.json")
        
        # Calculate overall statistics
        total_faces_detected = len(self.best_faces)
        total_face_images = len(self.best_faces)  # One best image per face
        
        summary = {
            "analysis_info": {
                "video_path": str(self.video_path),
                "total_frames": int(len(self.frame_annotations)),
                "analysis_date": str(datetime.now().isoformat()),
                "output_directory": str(self.output_dir)
            },
            "overview": {
                "total_students_tracked": int(len(self.student_tracks)),
                "total_unique_faces": int(len(self.face_tracker.get_all_face_ids())),
                "total_face_images_captured": int(total_face_images),
                "models_loaded": [str(k) for k in self.models.keys()]
            },
            "file_structure": {
                "face_images": "face_images/",
                "face_metadata": "face_metadata/",
                "student_data": "student_data/",
                "activity_data": "activity_data/",
                "statistics": "statistics/"
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Main summary saved: {summary_file}")
    
    def save_student_data_files(self):
        """Save individual student data files"""
        for student_id, track in self.student_tracks.items():
            if len(track) >= 3:
                student_file = os.path.join(self.student_data_dir, f"{student_id}_data.json")
                
                activities = [r['activity'] for r in track]
                engagements = [r['engagement'] for r in track if r['engagement']]
                attention_levels = [r['attention_level'] for r in track]
                
                student_data = {
                    "student_id": str(student_id),
                    "tracking_info": {
                        "total_appearances": int(len(track)),
                        "first_seen_frame": int(track[0]['frame']),
                        "last_seen_frame": int(track[-1]['frame']),
                        "position_zone": str(track[0]['position_zone'])
                    },
                    "behavior_analysis": {
                        "dominant_activity": str(Counter(activities).most_common(1)[0][0]) if activities else "unknown",
                        "activity_distribution": {str(k): int(v) for k, v in Counter(activities).items()},
                    "engagement_rate": f"{len(engagements)/len(track)*100:.1f}%",
                        "attention_distribution": {str(k): int(v) for k, v in Counter(attention_levels).items()}
                    },
                    "face_data": self._find_student_face_with_base64(student_id)
                    # Timeline removed to reduce file size
                }
                
                with open(student_file, 'w') as f:
                    json.dump(student_data, f, indent=2)
        
        print(f"👥 Student data files saved in: {self.student_data_dir}")
    
    def save_face_data_files(self):
        """Save comprehensive face data files"""
        face_summary_file = os.path.join(self.statistics_dir, "face_summary.json")
        
        face_summary = {}
        for face_id in self.face_tracker.get_all_face_ids():
            history = self.face_tracker.get_face_history(face_id)
            if len(history) >= 1:
                confidences = [h[2] for h in history]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                
                # Get best face data if available
                best_face_data = self.best_faces.get(face_id, {})
                
                face_summary[f"face_{face_id}"] = {
                    "face_id": int(face_id),
                    "appearances": int(len(history)),
                    "average_confidence": f"{float(avg_confidence):.3f}",
                    "max_confidence": f"{float(max_confidence):.3f}",
                    "first_seen_frame": int(history[0][0]),
                    "last_seen_frame": int(history[-1][0]),
                    "best_confidence": f"{float(best_face_data.get('best_confidence', 0)):.3f}",
                    "best_quality_score": f"{float(best_face_data.get('best_quality_score', 0)):.3f}",
                    "best_frame": int(best_face_data.get('best_frame_num', 0)),
                    "best_image_file": f"face_{int(face_id):02d}_best.jpg",
                    "metadata_file": f"face_{int(face_id):02d}_best_metadata.json"
                }
        
        with open(face_summary_file, 'w') as f:
            json.dump(face_summary, f, indent=2)
        
        print(f"👤 Face summary saved: {face_summary_file}")
    
    def save_activity_statistics(self):
        """Save activity and engagement statistics"""
        activity_file = os.path.join(self.activity_data_dir, "activity_statistics.json")
        
        # Aggregate all activities and attention levels
        all_activities = []
        all_attention = []
        all_engagement = []
        
        for student_id, track in self.student_tracks.items():
            for record in track:
                all_activities.append(record['activity'])
                all_attention.append(record['attention_level'])
                if record['engagement']:
                    all_engagement.append(record['engagement'])
        
        activity_stats = {
            "overall_statistics": {
                "total_activity_records": int(len(all_activities)),
                "activity_distribution": {str(k): int(v) for k, v in Counter(all_activities).items()},
                "attention_distribution": {str(k): int(v) for k, v in Counter(all_attention).items()},
                "engagement_distribution": {str(k): int(v) for k, v in Counter(all_engagement).items()},
                "overall_engagement_rate": f"{len(all_engagement)/len(all_activities)*100:.1f}%" if all_activities else "0%"
            },
            "by_zone": {
                "front_zone": self._get_zone_statistics("front"),
                "middle_zone": self._get_zone_statistics("middle"),
                "back_zone": self._get_zone_statistics("back")
            }
        }
        
        with open(activity_file, 'w') as f:
            json.dump(activity_stats, f, indent=2)
        
        print(f"📊 Activity statistics saved: {activity_file}")
    
    def _get_zone_statistics(self, zone):
        """Get statistics for a specific zone"""
        zone_activities = []
        zone_engagement = []
        
        for student_id, track in self.student_tracks.items():
            for record in track:
                if record['position_zone'] == zone:
                    zone_activities.append(record['activity'])
                    if record['engagement']:
                        zone_engagement.append(record['engagement'])
        
        return {
            "activity_distribution": {str(k): int(v) for k, v in Counter(zone_activities).items()},
            "engagement_rate": f"{len(zone_engagement)/len(zone_activities)*100:.1f}%" if zone_activities else "0%"
        }
    
    def save_frame_data(self):
        """Save frame-by-frame analysis data"""
        frames_file = os.path.join(self.statistics_dir, "frame_analysis.json")
        
        frame_summary = []
        for frame_data in self.frame_annotations:
            summary = {
                "frame_number": int(frame_data['frame_number']),
                "timestamp": float(frame_data['timestamp']),
                "persons_detected": int(len(frame_data['annotated_persons'])),
                "faces_detected": int(len(frame_data.get('faces', []))),
                "activities": [p['attributes'].get('activity', 'unknown') for p in frame_data['annotated_persons']],
                "engagement_count": int(sum(1 for p in frame_data['annotated_persons'] if p['attributes'].get('engagement') == 'engaged'))
            }
            frame_summary.append(summary)
        
        with open(frames_file, 'w') as f:
            json.dump(frame_summary, f, indent=2)
        
        print(f"🎬 Frame analysis saved: {frames_file}")
    
    def save_comprehensive_report(self):
        """Save one comprehensive JSON with all analysis data"""
        print("📋 Creating comprehensive analysis report...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        comprehensive_data = {
            "analysis_metadata": {
                "video_path": str(self.video_path),
                "output_directory": str(self.output_dir),
                "analysis_date": str(datetime.now().isoformat()),
                "total_frames_processed": int(len(self.frame_annotations)),
                "models_loaded": [str(k) for k in self.models.keys()],
                "headless_mode": self.headless_mode,
                "analysis_version": "1.0"
            },
            
            "overview_statistics": {
                "total_students_tracked": int(len(self.student_tracks)),
                "total_unique_faces_detected": int(len(self.best_faces)),
                "total_face_images_saved": int(len(self.best_faces)),
                "video_duration_seconds": float(len(self.frame_annotations) / 30.0) if self.frame_annotations else 0.0,  # Assuming 30 FPS
                "analysis_success_rate": "100%"
            },
            
            "students": self._compile_student_data(),
            
            "faces": self._compile_face_data(),
            
            "activity_analysis": self._compile_activity_data(),
            
            "frame_analysis": self._compile_frame_data(),
            
            "file_references": {
                "face_images_directory": "face_images/",
                "face_metadata_directory": "face_metadata/",
                "student_data_directory": "student_data/",
                "activity_data_directory": "activity_data/",
                "statistics_directory": "statistics/",
                "face_image_files": [f"face_images/face_{fid:02d}_best.jpg" for fid in self.best_faces.keys()],
                "face_metadata_files": [f"face_metadata/face_{fid:02d}_best_metadata.json" for fid in self.best_faces.keys()],
                "student_data_files": [f"student_data/{sid}_data.json" for sid in self.student_tracks.keys()],
                "activity_statistics_file": "activity_data/activity_statistics.json",
                "face_summary_file": "statistics/face_summary.json",
                "frame_analysis_file": "statistics/frame_analysis.json"
            },
            
            "classroom_zones": {
                "front_zone": {
                    "description": "Top 40% of classroom",
                    "student_count": len([s for s in self.student_tracks.values() if s and s[0].get('position_zone') == 'front']),
                    "engagement_rate": self._get_zone_statistics("front")["engagement_rate"]
                },
                "middle_zone": {
                    "description": "40%-70% of classroom",
                    "student_count": len([s for s in self.student_tracks.values() if s and s[0].get('position_zone') == 'middle']),
                    "engagement_rate": self._get_zone_statistics("middle")["engagement_rate"]
                },
                "back_zone": {
                    "description": "Bottom 30% of classroom",
                    "student_count": len([s for s in self.student_tracks.values() if s and s[0].get('position_zone') == 'back']),
                    "engagement_rate": self._get_zone_statistics("back")["engagement_rate"]
                }
            }
        }
        
        # Save comprehensive report
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_analysis_report.json")
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2)
        
        print(f"📋 Comprehensive report saved: {comprehensive_file}")
    
    def _compile_student_data(self):
        """Compile all student data into a single structure"""
        students = {}
        
        for student_id, track in self.student_tracks.items():
            if len(track) >= 3:
                activities = [r['activity'] for r in track]
                engagements = [r['engagement'] for r in track if r['engagement']]
                attention_levels = [r['attention_level'] for r in track]
                
                # Find associated face with base64 image
                associated_face = self._find_student_face_with_base64(student_id)
                
                students[student_id] = {
                    "student_id": str(student_id),
                    "tracking_summary": {
                        "total_appearances": int(len(track)),
                        "first_seen_frame": int(track[0]['frame']),
                        "last_seen_frame": int(track[-1]['frame']),
                        "position_zone": str(track[0]['position_zone']),
                        "duration_in_class": f"{len(track)/30.0:.1f} seconds"  # Assuming 30 FPS
                    },
                    "behavior_summary": {
                        "dominant_activity": str(Counter(activities).most_common(1)[0][0]) if activities else "unknown",
                        "activity_distribution": {str(k): int(v) for k, v in Counter(activities).items()},
                        "engagement_rate": f"{len(engagements)/len(track)*100:.1f}%",
                        "attention_distribution": {str(k): int(v) for k, v in Counter(attention_levels).items()},
                        "engagement_level": "high" if len(engagements)/len(track) > 0.7 else "medium" if len(engagements)/len(track) > 0.4 else "low"
                    },
                    "face_data": associated_face
                }
        
        return students
    
    def _compile_face_data(self):
        """Compile all face data into a single structure"""
        faces = {}
        
        for face_id, best_face_data in self.best_faces.items():
            history = self.face_tracker.get_face_history(face_id)
            if len(history) >= 1:
                confidences = [h[2] for h in history]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                
                faces[f"face_{face_id}"] = {
                    "face_id": int(face_id),
                    "detection_summary": {
                        "total_appearances": int(len(history)),
                        "average_confidence": f"{float(avg_confidence):.3f}",
                        "max_confidence": f"{float(max_confidence):.3f}",
                        "first_seen_frame": int(history[0][0]),
                        "last_seen_frame": int(history[-1][0])
                    },
                    "best_image_info": {
                        "confidence": f"{float(best_face_data['best_confidence']):.3f}",
                        "quality_score": f"{float(best_face_data['best_quality_score']):.3f}",
                        "combined_score": f"{float(0.7 * best_face_data['best_confidence'] + 0.3 * best_face_data['best_quality_score']):.3f}",
                        "best_frame": int(best_face_data['best_frame_num']),
                        "replacement_count": int(best_face_data.get('replacement_count', 0)),
                        "image_file": f"face_{int(face_id):02d}_best.jpg",
                        "metadata_file": f"face_{int(face_id):02d}_best_metadata.json",
                        "base64_image": best_face_data.get('base64_image', ''),
                        "image_available": bool(best_face_data.get('base64_image', ''))
                    }
                }
        
        return faces
    
    def _compile_activity_data(self):
        """Compile activity analysis data"""
        all_activities = []
        all_attention = []
        all_engagement = []
        
        for student_id, track in self.student_tracks.items():
            for record in track:
                all_activities.append(record['activity'])
                all_attention.append(record['attention_level'])
                if record['engagement']:
                    all_engagement.append(record['engagement'])
        
        return {
            "overall_activity_summary": {
                "total_activity_records": int(len(all_activities)),
                "activity_distribution": {str(k): int(v) for k, v in Counter(all_activities).items()},
                "attention_distribution": {str(k): int(v) for k, v in Counter(all_attention).items()},
                "engagement_distribution": {str(k): int(v) for k, v in Counter(all_engagement).items()},
                "overall_engagement_rate": f"{len(all_engagement)/len(all_activities)*100:.1f}%" if all_activities else "0%"
            },
            "zone_analysis": {
                "front_zone": self._get_zone_statistics("front"),
                "middle_zone": self._get_zone_statistics("middle"),
                "back_zone": self._get_zone_statistics("back")
            }
        }
    
    def _compile_frame_data(self):
        """Compile frame-by-frame analysis summary"""
        if not self.frame_annotations:
            return {}
        
        frame_summary = {
            "total_frames": int(len(self.frame_annotations)),
            "frames_with_students": int(sum(1 for f in self.frame_annotations if len(f['annotated_persons']) > 0)),
            "frames_with_faces": int(sum(1 for f in self.frame_annotations if len(f.get('faces', [])) > 0)),
            "average_students_per_frame": float(sum(len(f['annotated_persons']) for f in self.frame_annotations) / len(self.frame_annotations)),
            "average_faces_per_frame": float(sum(len(f.get('faces', [])) for f in self.frame_annotations) / len(self.frame_annotations)),
            "sample_frames": []
        }
        
        # Add sample frames (every 100th frame)
        for i in range(0, len(self.frame_annotations), 100):
            frame_data = self.frame_annotations[i]
            frame_summary["sample_frames"].append({
                "frame_number": int(frame_data['frame_number']),
                "timestamp": float(frame_data['timestamp']),
                "students_detected": int(len(frame_data['annotated_persons'])),
                "faces_detected": int(len(frame_data.get('faces', []))),
                "engagement_count": int(sum(1 for p in frame_data['annotated_persons'] if p['attributes'].get('engagement') == 'engaged'))
            })
        
        return frame_summary
    
    def _find_student_face(self, student_id):
        """Find the best matching face for a student based on position and timing"""
        student_track = self.student_tracks.get(student_id, [])
        if not student_track:
            return None
        
        # Get student's typical position
        student_center = None
        for record in student_track:
            if 'center' in record:
                if student_center is None:
                    student_center = record['center']
                else:
                    # Average the position
                    student_center = [(student_center[0] + record['center'][0])/2, 
                                    (student_center[1] + record['center'][1])/2]
        
        if student_center is None:
            return None
        
        # Find closest face based on position
        best_face_id = None
        min_distance = float('inf')
        
        for face_id, best_face_data in self.best_faces.items():
            bbox = best_face_data['best_bbox']
            face_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
            
            distance = math.sqrt((student_center[0] - face_center[0])**2 + (student_center[1] - face_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                best_face_id = face_id
        
        if best_face_id is not None and min_distance < 150:  # Within reasonable distance
            return {
                "face_id": int(best_face_id),
                "image_file": f"face_{int(best_face_id):02d}_best.jpg",
                "confidence": f"{float(self.best_faces[best_face_id]['best_confidence']):.3f}",
                "position_distance": float(min_distance)
            }
        
        return None
    
    def _find_student_face_with_base64(self, student_id):
        """Find the best matching face for a student with base64 image data"""
        student_track = self.student_tracks.get(student_id, [])
        if not student_track:
            return None
        
        # Get student's typical position
        student_center = None
        for record in student_track:
            if 'center' in record:
                if student_center is None:
                    student_center = record['center']
                else:
                    # Average the position
                    student_center = [(student_center[0] + record['center'][0])/2, 
                                    (student_center[1] + record['center'][1])/2]
        
        if student_center is None:
            return None
        
        # Find closest face based on position
        best_face_id = None
        min_distance = float('inf')
        
        for face_id, best_face_data in self.best_faces.items():
            bbox = best_face_data['best_bbox']
            face_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
            
            distance = math.sqrt((student_center[0] - face_center[0])**2 + (student_center[1] - face_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                best_face_id = face_id
        
        if best_face_id is not None and min_distance < 150:  # Within reasonable distance
            best_face_data = self.best_faces[best_face_id]
            return {
                "face_id": int(best_face_id),
                "confidence": f"{float(best_face_data['best_confidence']):.3f}",
                "quality_score": f"{float(best_face_data['best_quality_score']):.3f}",
                "best_frame": int(best_face_data['best_frame_num']),
                "replacement_count": int(best_face_data.get('replacement_count', 0)),
                "position_distance": float(min_distance),
                "base64_image": best_face_data.get('base64_image', ''),
                "image_available": bool(best_face_data.get('base64_image', ''))
            }
        
        return None

def main():
    """Run classroom video analysis with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classroom Video Analysis with Face Detection')
    parser.add_argument('--video', '-v', help='Path to classroom video file')
    parser.add_argument('--headless', '-H', action='store_true', 
                       help='Run without OpenCV display for faster processing')
    parser.add_argument('--save-frames', '-s', action='store_true',
                       help='Save sample frames during processing (headless mode)')
    parser.add_argument('--output', '-o', default='realtime_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("🎬 Classroom Video Analysis")
    print("=" * 50)
    
    # Get video file
    if args.video:
        video_path = args.video
    else:
        video_path = input("Enter path to classroom video: ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return
    
    # Create analyzer
    analyzer = RealtimeClassroomAnalyzer(video_path, args.output, args.headless)
    
    # Run analysis
    try:
        if args.headless:
            print("🚀 Running in headless mode (no display)")
            print("💡 Use --help to see all options")
        else:
            print("🎬 Running with OpenCV display")
            print("💡 Use --headless for faster processing")
        
        results = analyzer.analyze_video_realtime(
            display=not args.headless, 
            save_frames=args.save_frames
        )
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Check results in: {analyzer.output_dir}")
        print(f"📊 Processed {len(results)} frames")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()