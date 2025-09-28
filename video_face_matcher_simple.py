#!/usr/bin/env python3
"""
Simple Video Face Matcher - Original faces only, no enhancement
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import face_recognition

class SimpleVideoFaceMatcher:
    """Simple face matcher using original faces only"""
    
    def __init__(self, database_path="face_database.json"):
        self.database_path = database_path
        self.face_database = {}
        self.load_database()
    
    def load_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    self.face_database = json.load(f)
                print(f"✅ Loaded face database with {len(self.face_database)} known faces")
            else:
                print("📝 Creating new face database")
                self.face_database = {}
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            self.face_database = {}
    
    def save_database(self):
        """Save face database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.face_database, f, indent=2)
            print(f"✅ Saved face database with {len(self.face_database)} faces")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def extract_face_encoding(self, face_image):
        """Extract face encoding from image (original only, no enhancement)"""
        try:
            # Ensure image is valid
            if face_image is None or face_image.size == 0:
                return None
            
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Ensure image is in correct format
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Use original face image without any enhancement
            try:
                # Try to get face encodings directly
                encodings = face_recognition.face_encodings(rgb_image)
                
                if encodings:
                    return encodings[0].tolist()
                    
            except Exception as e:
                pass
            
            # If direct method fails, try detecting face locations first
            try:
                # Detect face locations first
                face_locations = face_recognition.face_locations(rgb_image)
                
                if face_locations:
                    # Extract encodings using detected locations
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    
                    if encodings:
                        return encodings[0].tolist()
                        
            except Exception as e:
                pass
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error extracting face encoding: {e}")
            return None
    
    def add_face(self, person_id, face_image, video_id=None, frame_number=None):
        """Add a face to the database"""
        try:
            # Extract face encoding
            encoding = self.extract_face_encoding(face_image)
            
            if encoding is None:
                print(f"⚠️ Could not extract encoding for {person_id}")
                return False
            
            # Add to database
            if person_id not in self.face_database:
                self.face_database[person_id] = {
                    'encodings': [],
                    'images': [],
                    'videos': set(),
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'total_appearances': 0
                }
            
            # Add encoding
            self.face_database[person_id]['encodings'].append(encoding)
            
            # Add image info
            image_info = {
                'timestamp': datetime.now().isoformat(),
                'video_id': video_id,
                'frame_number': frame_number
            }
            self.face_database[person_id]['images'].append(image_info)
            
            # Add video info
            if video_id:
                self.face_database[person_id]['videos'].add(video_id)
            
            # Update metadata
            self.face_database[person_id]['last_seen'] = datetime.now().isoformat()
            self.face_database[person_id]['total_appearances'] += 1
            
            # Save database
            self.save_database()
            
            print(f"✅ Added face for {person_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding face: {e}")
            return False
    
    def find_matching_face(self, face_image, threshold=0.6):
        """Find matching face in database"""
        try:
            # Extract encoding from input image
            encoding = self.extract_face_encoding(face_image)
            
            if encoding is None:
                return None
            
            best_match = None
            best_distance = float('inf')
            
            # Compare with all faces in database
            for person_id, person_data in self.face_database.items():
                for stored_encoding in person_data['encodings']:
                    try:
                        # Calculate face distance
                        distance = face_recognition.face_distance([stored_encoding], [encoding])[0]
                        
                        if distance < best_distance and distance < threshold:
                            best_distance = distance
                            best_match = {
                                'person_id': person_id,
                                'distance': distance,
                                'confidence': 1 - distance
                            }
                    except Exception as e:
                        continue
            
            return best_match
            
        except Exception as e:
            print(f"❌ Error finding matching face: {e}")
            return None
    
    def process_video_faces(self, video_analysis_data):
        """Process faces from video analysis data"""
        try:
            video_id = video_analysis_data.get('video_id', 'unknown')
            faces_data = video_analysis_data.get('faces', [])
            
            print(f"🎬 Processing {len(faces_data)} faces from video {video_id}")
            
            for face_data in faces_data:
                if not isinstance(face_data, dict):
                    continue
                
                person_id = face_data.get('person_id', 'unknown')
                face_image = face_data.get('face_image')
                frame_number = face_data.get('frame_number', 0)
                
                if face_image is not None:
                    # Try to find matching face first
                    match = self.find_matching_face(face_image)
                    
                    if match:
                        # Update existing person
                        existing_id = match['person_id']
                        self.add_face(existing_id, face_image, video_id, frame_number)
                        print(f"🔄 Updated existing person {existing_id} (confidence: {match['confidence']:.3f})")
                    else:
                        # Add new person
                        self.add_face(person_id, face_image, video_id, frame_number)
                        print(f"➕ Added new person {person_id}")
            
            print(f"✅ Processed faces for video {video_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing video faces: {e}")
            return False
    
    def get_attendance_summary(self):
        """Get attendance summary"""
        try:
            summary = {
                'total_persons': len(self.face_database),
                'total_appearances': sum(person['total_appearances'] for person in self.face_database.values()),
                'persons': {}
            }
            
            for person_id, person_data in self.face_database.items():
                summary['persons'][person_id] = {
                    'total_appearances': person_data['total_appearances'],
                    'videos': list(person_data['videos']),
                    'first_seen': person_data['first_seen'],
                    'last_seen': person_data['last_seen'],
                    'encodings_count': len(person_data['encodings']),
                    'images_count': len(person_data['images'])
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {'total_persons': 0, 'total_appearances': 0, 'persons': {}}

# Test the simple face matcher
if __name__ == "__main__":
    print("🧪 Testing Simple Face Matcher")
    print("=" * 40)
    
    matcher = SimpleVideoFaceMatcher()
    
    # Test with dummy data
    dummy_analysis = {
        'video_id': 'test_video',
        'faces': [
            {
                'person_id': 'person_1',
                'face_image': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                'frame_number': 1
            }
        ]
    }
    
    success = matcher.process_video_faces(dummy_analysis)
    print(f"✅ Processing result: {success}")
    
    summary = matcher.get_attendance_summary()
    print(f"📊 Summary: {summary}")
    
    print("✅ Simple face matcher test completed!")

