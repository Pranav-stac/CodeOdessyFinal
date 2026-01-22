"""
Face Tracking with Engagement Analysis
Tracks faces across video frames, assigns IDs, calculates engagement, and generates detailed reports
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
import time
from tqdm import tqdm


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


class FaceEngagementAnalyzer:
    """Analyzes face tracks with engagement metrics"""
    
    def __init__(self, video_path, output_dir="face_track_analysis"):
        """
        Initialize face tracking and engagement analyzer
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory for output files
        """
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.models = self.load_models()
        
        # Initialize face tracker
        self.face_tracker = FaceTracker(max_face_ids=100, max_distance=120)
        
        # Track engagement data per face ID
        self.face_engagement_data = defaultdict(lambda: {
            'face_id': None,
            'total_frames': 0,
            'total_time': 0.0,
            'engagement_scores': [],
            'engagement_states': [],  # 'engaged', 'partially_engaged', 'not_engaged'
            'activities': Counter(),
            'attention_levels': Counter(),
            'zones': Counter(),  # front, middle, back
            'bbox_history': [],
            'first_seen': None,
            'last_seen': None,
            'average_confidence': 0.0,
            'confidence_history': []
        })
        
        # Frame-by-frame tracking
        self.frame_data = []
        
        # Video properties
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
        
    def load_models(self):
        """Load YOLO models for detection, pose, and face"""
        models = {}
        
        # Get the directory where the script is running from
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
        except (NameError, AttributeError):
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Look for AI_Model_Weights
        weights_dir = os.path.join(base_path, "AI_Model_Weights", "AI_Model_Weights")
        if not os.path.exists(weights_dir):
            weights_dir = os.path.join(base_path, "AI_Model_Weights")
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
                    if hasattr(model, 'model'):
                        model.model.eval()
                        if hasattr(model.model, 'fuse'):
                            model.model.fuse()
                    models[model_type] = model
                    print(f"✅ Loaded {model_type}: {filename}")
                except Exception as e:
                    print(f"❌ Failed to load {model_type}: {e}")
            else:
                print(f"⚠️ Model not found: {model_path}")
        
        return models
    
    def calculate_engagement_score(self, activity, attention, zone, confidence):
        """
        Calculate engagement score based on multiple factors
        
        Args:
            activity (str): Detected activity
            attention (str): Attention level
            zone (str): Position zone (front/middle/back)
            confidence (float): Detection confidence
            
        Returns:
            tuple: (engagement_score, engagement_state)
        """
        engagement_score = 0.0
        
        # Activity-based scoring
        activity_scores = {
            'raising_hand': 1.0,
            'writing': 0.9,
            'listening': 0.6,
            'reading': 0.7,
            'talking': 0.5,
            'unknown': 0.2,
            'distracted': 0.1,
            'using_phone': 0.0,
            'sleeping': 0.0
        }
        engagement_score += activity_scores.get(activity, 0.3)
        
        # Attention-based scoring
        attention_scores = {
            'focused': 0.3,
            'partially_focused': 0.15,
            'distracted': 0.0,
            'not_visible': 0.0
        }
        engagement_score += attention_scores.get(attention, 0.1)
        
        # Zone-based scoring
        zone_scores = {
            'front': 0.2,
            'middle': 0.1,
            'back': 0.05
        }
        engagement_score += zone_scores.get(zone, 0.1)
        
        # Confidence bonus
        engagement_score += confidence * 0.1
        
        # Normalize to 0-1 range
        engagement_score = min(1.0, engagement_score)
        
        # Determine engagement state
        if engagement_score >= 0.8:
            engagement_state = 'engaged'
        elif engagement_score >= 0.5:
            engagement_state = 'partially_engaged'
        else:
            engagement_state = 'not_engaged'
        
        return engagement_score, engagement_state
    
    def match_face_to_person(self, face_bbox, persons, pose_data):
        """
        Match a face to the nearest person detection
        
        Args:
            face_bbox (tuple): Face bounding box (x1, y1, x2, y2)
            persons (list): List of person detections
            pose_data (list): List of pose analysis data
            
        Returns:
            dict: Matched person data with activity and attention
        """
        if not persons:
            return {
                'activity': 'unknown',
                'attention': 'not_visible',
                'zone': 'middle',
                'confidence': 0.0
            }
        
        face_center = ((face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2)
        
        # Find closest person
        min_distance = float('inf')
        matched_person = None
        matched_pose = None
        
        for i, person in enumerate(persons):
            person_center = person['center']
            distance = math.sqrt(
                (face_center[0] - person_center[0])**2 + 
                (face_center[1] - person_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                matched_person = person
                matched_pose = pose_data[i] if i < len(pose_data) else None
        
        # If face is too far from any person, return default
        if min_distance > 200:  # Threshold in pixels
            return {
                'activity': 'unknown',
                'attention': 'not_visible',
                'zone': 'middle',
                'confidence': 0.0
            }
        
        # Extract activity and attention from pose analysis
        pose_analysis = matched_pose['analysis'] if matched_pose else {'activity': 'unknown', 'confidence': 0.0}
        activity = pose_analysis.get('activity', 'unknown')
        confidence = matched_person.get('confidence', 0.0)
        
        # Determine attention level
        if activity in ['raising_hand', 'writing']:
            attention = 'focused'
        elif activity == 'listening':
            attention = 'partially_focused'
        else:
            attention = 'distracted'
        
        # Determine zone based on position
        face_y = face_center[1]
        if face_y < self.height * 0.4:
            zone = 'front'
        elif face_y < self.height * 0.7:
            zone = 'middle'
        else:
            zone = 'back'
        
        return {
            'activity': activity,
            'attention': attention,
            'zone': zone,
            'confidence': confidence
        }
    
    def analyze_video(self):
        """Main video analysis function"""
        print(f"🎬 Starting face tracking and engagement analysis: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {self.total_frames} frames, {self.fps:.1f} FPS, {self.width}x{self.height}")
        
        # Setup video writer for output
        output_video_path = os.path.join(self.output_dir, "output_with_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        
        # Progress bar
        pbar = tqdm(total=self.total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / self.fps if self.fps > 0 else 0
            
            # Detect faces
            tracked_faces = self.detect_faces(frame, frame_count)
            
            # Detect persons and poses
            persons = []
            pose_data = []
            if 'detection' in self.models:
                results = self.models['detection'](frame, verbose=False)
                persons = self.extract_persons(results)
                
                if 'pose' in self.models and persons:
                    pose_results = self.models['pose'](frame, verbose=False)
                    pose_data = self.extract_pose_data(pose_results)
            
            # Match faces to persons and calculate engagement
            for face in tracked_faces:
                face_id = face['id']
                face_bbox = face['bbox']
                
                # Match face to person
                person_data = self.match_face_to_person(face_bbox, persons, pose_data)
                
                # Calculate engagement
                engagement_score, engagement_state = self.calculate_engagement_score(
                    person_data['activity'],
                    person_data['attention'],
                    person_data['zone'],
                    face['confidence']
                )
                
                # Update face engagement data
                face_data = self.face_engagement_data[face_id]
                face_data['face_id'] = face_id
                face_data['total_frames'] += 1
                face_data['total_time'] += (1.0 / self.fps) if self.fps > 0 else 0
                face_data['engagement_scores'].append(engagement_score)
                face_data['engagement_states'].append(engagement_state)
                face_data['activities'][person_data['activity']] += 1
                face_data['attention_levels'][person_data['attention']] += 1
                face_data['zones'][person_data['zone']] += 1
                face_data['bbox_history'].append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'bbox': face_bbox,
                    'engagement_score': engagement_score,
                    'engagement_state': engagement_state,
                    'activity': person_data['activity']
                })
                face_data['confidence_history'].append(face['confidence'])
                
                if face_data['first_seen'] is None:
                    face_data['first_seen'] = timestamp
                face_data['last_seen'] = timestamp
                
                # Store face data for frame
                face['engagement_score'] = engagement_score
                face['engagement_state'] = engagement_state
                face['activity'] = person_data['activity']
                face['attention'] = person_data['attention']
                face['zone'] = person_data['zone']
            
            # Store frame data
            self.frame_data.append({
                'frame_number': frame_count,
                'timestamp': timestamp,
                'faces': tracked_faces
            })
            
            # Draw bounding boxes and labels
            annotated_frame = self.draw_tracking(frame.copy(), tracked_faces)
            out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"✅ Analysis complete! Processed {frame_count} frames")
        print(f"📹 Output video saved: {output_video_path}")
        
        # Generate reports
        self.generate_reports()
        
        return self.face_engagement_data
    
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
        
        return tracked_faces
    
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
        """Extract pose keypoints and analyze"""
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
    
    def analyze_pose(self, keypoints):
        """Analyze pose for activity detection"""
        if len(keypoints) < 17:
            return {'activity': 'unknown', 'confidence': 0.0}
        
        def get_keypoint(idx):
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                x, y, conf = keypoints[idx][:3]
                return [x, y] if conf > 0.3 else [0, 0]
            return [0, 0]
        
        left_wrist = get_keypoint(9)
        right_wrist = get_keypoint(10)
        left_shoulder = get_keypoint(5)
        right_shoulder = get_keypoint(6)
        
        activity = 'listening'
        confidence = 0.5
        
        # Check for raised hand
        left_hand_raised = (left_wrist[1] > 0 and left_shoulder[1] > 0 and 
                           left_wrist[1] < left_shoulder[1] - 20)
        right_hand_raised = (right_wrist[1] > 0 and right_shoulder[1] > 0 and 
                            right_wrist[1] < right_shoulder[1] - 20)
        
        if left_hand_raised or right_hand_raised:
            activity = 'raising_hand'
            confidence = 0.9
        # Check for writing
        elif (left_wrist[0] > 0 and left_shoulder[0] > 0 and 
              abs(left_wrist[0] - left_shoulder[0]) < 50 and
              left_wrist[1] > left_shoulder[1]) or \
             (right_wrist[0] > 0 and right_shoulder[0] > 0 and 
              abs(right_wrist[0] - right_shoulder[0]) < 50 and
              right_wrist[1] > right_shoulder[1]):
            activity = 'writing'
            confidence = 0.8
        else:
            activity = 'listening'
            confidence = 0.6
        
        return {'activity': activity, 'confidence': confidence}
    
    def draw_tracking(self, frame, tracked_faces):
        """Draw bounding boxes and labels on frame"""
        # Color palette for different face IDs
        colors = [
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
        
        for face in tracked_faces:
            face_id = face['id']
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get color for this face ID
            color = colors[face_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            engagement_state = face.get('engagement_state', 'unknown')
            activity = face.get('activity', 'unknown')
            engagement_score = face.get('engagement_score', 0.0)
            
            label = f"ID:{face_id} | {engagement_state} | {activity} | {engagement_score:.2f}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def generate_reports(self):
        """Generate detailed reports for each face ID"""
        print("📊 Generating detailed reports...")
        
        # Calculate final statistics for each face
        for face_id, face_data in self.face_engagement_data.items():
            if face_data['total_frames'] == 0:
                continue
            
            # Calculate average engagement
            if face_data['engagement_scores']:
                face_data['average_engagement'] = np.mean(face_data['engagement_scores'])
                face_data['max_engagement'] = np.max(face_data['engagement_scores'])
                face_data['min_engagement'] = np.min(face_data['engagement_scores'])
            else:
                face_data['average_engagement'] = 0.0
                face_data['max_engagement'] = 0.0
                face_data['min_engagement'] = 0.0
            
            # Calculate engagement percentages
            total_states = len(face_data['engagement_states'])
            if total_states > 0:
                face_data['engagement_percentages'] = {
                    'engaged': (face_data['engagement_states'].count('engaged') / total_states) * 100,
                    'partially_engaged': (face_data['engagement_states'].count('partially_engaged') / total_states) * 100,
                    'not_engaged': (face_data['engagement_states'].count('not_engaged') / total_states) * 100
                }
            else:
                face_data['engagement_percentages'] = {
                    'engaged': 0.0,
                    'partially_engaged': 0.0,
                    'not_engaged': 0.0
                }
            
            # Calculate average confidence
            if face_data['confidence_history']:
                face_data['average_confidence'] = np.mean(face_data['confidence_history'])
            else:
                face_data['average_confidence'] = 0.0
        
        # Generate individual reports for each face
        for face_id, face_data in self.face_engagement_data.items():
            if face_data['total_frames'] == 0:
                continue
            
            report = self.generate_individual_report(face_id, face_data)
            
            # Save individual report
            report_path = os.path.join(self.output_dir, f"face_{face_id:03d}_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Generate summary report
        summary_report = self.generate_summary_report()
        summary_path = os.path.join(self.output_dir, "summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Generate text report
        text_report = self.generate_text_report()
        text_report_path = os.path.join(self.output_dir, "detailed_report.txt")
        with open(text_report_path, 'w') as f:
            f.write(text_report)
        
        print(f"✅ Reports generated in: {self.output_dir}")
    
    def generate_individual_report(self, face_id, face_data):
        """Generate detailed report for a single face"""
        report = {
            'face_id': face_id,
            'tracking_summary': {
                'total_frames': face_data['total_frames'],
                'total_time_seconds': face_data['total_time'],
                'total_time_formatted': f"{int(face_data['total_time'] // 60)}m {int(face_data['total_time'] % 60)}s",
                'first_seen_seconds': face_data['first_seen'],
                'last_seen_seconds': face_data['last_seen'],
                'duration_seconds': face_data['last_seen'] - face_data['first_seen'] if face_data['first_seen'] else 0
            },
            'engagement_metrics': {
                'average_engagement_score': face_data.get('average_engagement', 0.0),
                'max_engagement_score': face_data.get('max_engagement', 0.0),
                'min_engagement_score': face_data.get('min_engagement', 0.0),
                'engagement_percentages': face_data.get('engagement_percentages', {}),
                'engagement_distribution': {
                    'engaged': face_data['engagement_states'].count('engaged'),
                    'partially_engaged': face_data['engagement_states'].count('partially_engaged'),
                    'not_engaged': face_data['engagement_states'].count('not_engaged')
                }
            },
            'activity_analysis': {
                'most_common_activity': face_data['activities'].most_common(1)[0][0] if face_data['activities'] else 'unknown',
                'activity_breakdown': dict(face_data['activities'])
            },
            'attention_analysis': {
                'most_common_attention': face_data['attention_levels'].most_common(1)[0][0] if face_data['attention_levels'] else 'unknown',
                'attention_breakdown': dict(face_data['attention_levels'])
            },
            'position_analysis': {
                'most_common_zone': face_data['zones'].most_common(1)[0][0] if face_data['zones'] else 'unknown',
                'zone_breakdown': dict(face_data['zones'])
            },
            'detection_quality': {
                'average_confidence': face_data['average_confidence'],
                'total_detections': len(face_data['confidence_history'])
            }
        }
        
        return report
    
    def generate_summary_report(self):
        """Generate summary report for all faces"""
        total_faces = len([f for f in self.face_engagement_data.values() if f['total_frames'] > 0])
        
        if total_faces == 0:
            return {'message': 'No faces detected in video'}
        
        all_engagement_scores = []
        for face_data in self.face_engagement_data.values():
            if face_data['total_frames'] > 0:
                all_engagement_scores.extend(face_data['engagement_scores'])
        
        summary = {
            'video_info': {
                'input_video': self.video_path,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'resolution': f"{self.width}x{self.height}",
                'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
            },
            'face_tracking_summary': {
                'total_unique_faces': total_faces,
                'total_tracked_frames': sum(f['total_frames'] for f in self.face_engagement_data.values()),
                'average_frames_per_face': sum(f['total_frames'] for f in self.face_engagement_data.values()) / total_faces if total_faces > 0 else 0
            },
            'engagement_summary': {
                'overall_average_engagement': np.mean(all_engagement_scores) if all_engagement_scores else 0.0,
                'overall_max_engagement': np.max(all_engagement_scores) if all_engagement_scores else 0.0,
                'overall_min_engagement': np.min(all_engagement_scores) if all_engagement_scores else 0.0
            },
            'face_ids': list(range(total_faces))
        }
        
        return summary
    
    def generate_text_report(self):
        """Generate human-readable text report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FACE TRACKING AND ENGAGEMENT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input Video: {self.video_path}")
        report_lines.append(f"Total Frames: {self.total_frames}")
        report_lines.append(f"FPS: {self.fps:.2f}")
        report_lines.append(f"Resolution: {self.width}x{self.height}")
        report_lines.append("")
        
        # Summary
        total_faces = len([f for f in self.face_engagement_data.values() if f['total_frames'] > 0])
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Unique Faces Tracked: {total_faces}")
        report_lines.append("")
        
        # Individual face reports
        report_lines.append("DETAILED FACE REPORTS")
        report_lines.append("=" * 80)
        
        for face_id in sorted(self.face_engagement_data.keys()):
            face_data = self.face_engagement_data[face_id]
            if face_data['total_frames'] == 0:
                continue
            
            report_lines.append(f"\nFace ID: {face_id:03d}")
            report_lines.append("-" * 80)
            
            # Tracking info
            report_lines.append(f"Tracking Duration: {face_data['total_time']:.2f} seconds")
            report_lines.append(f"Total Frames: {face_data['total_frames']}")
            report_lines.append(f"First Seen: {face_data['first_seen']:.2f}s")
            report_lines.append(f"Last Seen: {face_data['last_seen']:.2f}s")
            report_lines.append("")
            
            # Engagement metrics
            avg_engagement = face_data.get('average_engagement', 0.0)
            report_lines.append(f"Average Engagement Score: {avg_engagement:.3f}")
            report_lines.append(f"Max Engagement Score: {face_data.get('max_engagement', 0.0):.3f}")
            report_lines.append(f"Min Engagement Score: {face_data.get('min_engagement', 0.0):.3f}")
            report_lines.append("")
            
            # Engagement percentages
            eng_percentages = face_data.get('engagement_percentages', {})
            report_lines.append("Engagement Distribution:")
            report_lines.append(f"  Engaged: {eng_percentages.get('engaged', 0.0):.1f}%")
            report_lines.append(f"  Partially Engaged: {eng_percentages.get('partially_engaged', 0.0):.1f}%")
            report_lines.append(f"  Not Engaged: {eng_percentages.get('not_engaged', 0.0):.1f}%")
            report_lines.append("")
            
            # Activity breakdown
            report_lines.append("Activity Breakdown:")
            for activity, count in face_data['activities'].most_common():
                percentage = (count / face_data['total_frames']) * 100
                report_lines.append(f"  {activity}: {count} frames ({percentage:.1f}%)")
            report_lines.append("")
            
            # Attention breakdown
            report_lines.append("Attention Breakdown:")
            for attention, count in face_data['attention_levels'].most_common():
                percentage = (count / face_data['total_frames']) * 100
                report_lines.append(f"  {attention}: {count} frames ({percentage:.1f}%)")
            report_lines.append("")
            
            # Zone breakdown
            report_lines.append("Position Zone Breakdown:")
            for zone, count in face_data['zones'].most_common():
                percentage = (count / face_data['total_frames']) * 100
                report_lines.append(f"  {zone}: {count} frames ({percentage:.1f}%)")
            report_lines.append("")
            
            # Detection quality
            report_lines.append(f"Average Detection Confidence: {face_data['average_confidence']:.3f}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Tracking with Engagement Analysis')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='face_track_analysis',
                       help='Output directory for reports and video (default: face_track_analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    # Create analyzer
    analyzer = FaceEngagementAnalyzer(args.video_path, args.output_dir)
    
    # Run analysis
    try:
        analyzer.analyze_video()
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📹 Output video: {os.path.join(args.output_dir, 'output_with_tracking.mp4')}")
        print(f"📊 Reports: {os.path.join(args.output_dir, '*.json')} and {os.path.join(args.output_dir, 'detailed_report.txt')}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
