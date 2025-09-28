#!/usr/bin/env python3
"""
Simple Enhanced Face Matcher - Original faces only, no enhancement
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

class SimpleEnhancedFaceMatcher:
    """Simple enhanced face matcher using original faces only"""
    
    def __init__(self, database_path="enhanced_face_database.json"):
        self.database_path = database_path
        self.face_database = {}
        self.load_face_database()
    
    def load_face_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    self.face_database = json.load(f)
                print(f"✅ Loaded enhanced face database with {len(self.face_database)} known faces")
            else:
                print("📝 Creating new enhanced face database")
                self.face_database = {}
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            self.face_database = {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.face_database, f, indent=2)
            print(f"✅ Saved enhanced face database with {len(self.face_database)} faces")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def extract_face_encoding(self, face_image):
        """Extract face encoding (original only, no enhancement)"""
        try:
            if face_image is None or face_image.size == 0:
                return None
            
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Check image size and quality
            height, width = rgb_image.shape[:2]
            if height < 20 or width < 20:
                print(f"⚠️ Image too small: {width}x{height}")
                return None
            
            # Use original face image without any enhancement
            try:
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    return encodings[0].tolist()
            except Exception as e:
                print(f"⚠️ Face encoding failed: {e}")
            
            # Try face detection approach
            try:
                face_locations = face_recognition.face_locations(rgb_image)
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if encodings:
                        return encodings[0].tolist()
            except Exception as e:
                print(f"⚠️ Face detection failed: {e}")
            
            # Try with different models
            try:
                face_locations = face_recognition.face_locations(rgb_image, model="cnn")
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if encodings:
                        return encodings[0].tolist()
            except Exception as e:
                print(f"⚠️ CNN face detection failed: {e}")
            
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
                    'total_appearances': 0,
                    'metadata': {}
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
            self.save_face_database()
            
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
    
    def process_video_faces(self, video_analysis_data, video_id=None):
        """Process faces from video analysis data"""
        try:
            # Validate input data
            if not isinstance(video_analysis_data, dict):
                print(f"⚠️ Invalid analysis data type: {type(video_analysis_data)}")
                return False
            
            # Handle both old and new calling conventions
            if video_id is None:
                video_id = video_analysis_data.get('video_id', 'unknown')
            
            faces_data = video_analysis_data.get('faces', [])
            
            # Handle both list and dictionary formats for faces data
            if isinstance(faces_data, dict):
                # Convert dictionary to list format
                faces_list = []
                for face_id, face_info in faces_data.items():
                    if isinstance(face_info, dict):
                        face_info['person_id'] = face_id  # Ensure person_id is set
                        faces_list.append(face_info)
                faces_data = faces_list
                print(f"🔄 Converted faces dictionary to list: {len(faces_data)} faces")
            elif not isinstance(faces_data, list):
                print(f"⚠️ Invalid faces data type: {type(faces_data)}")
                return False
            
            print(f"🎬 Processing {len(faces_data)} faces from video {video_id}")
            
            matched_count = 0
            new_count = 0
            
            for face_data in faces_data:
                if not isinstance(face_data, dict):
                    print(f"⚠️ Skipping invalid face data: {type(face_data)}")
                    continue
                
                # Handle different face data structures
                person_id = face_data.get('person_id', face_data.get('face_id', 'unknown'))
                face_image = face_data.get('face_image')
                frame_number = face_data.get('frame_number', 0)
                
                # If no direct face_image, try to get from best_image_info
                if face_image is None and 'best_image_info' in face_data:
                    best_image_info = face_data['best_image_info']
                    if 'base64_image' in best_image_info:
                        try:
                            import base64
                            from PIL import Image
                            import io
                            # Decode base64 image
                            base64_data = best_image_info['base64_image']
                            image_data = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_data))
                            face_image = np.array(pil_image)
                            print(f"🔄 Decoded base64 image for {person_id}")
                        except Exception as e:
                            print(f"⚠️ Could not decode base64 image for {person_id}: {e}")
                            face_image = None
                
                # Debug face image data (commented out for cleaner output)
                # print(f"🔍 Face data for {person_id}:")
                # print(f"   - person_id type: {type(person_id)}")
                # print(f"   - face_image type: {type(face_image)}")
                # print(f"   - frame_number type: {type(frame_number)}")
                
                if face_image is not None:
                    # Try to find matching face first
                    match = self.find_matching_face(face_image)
                    
                    if match:
                        # Update existing person
                        existing_id = match['person_id']
                        self.add_face(existing_id, face_image, video_id, frame_number)
                        print(f"🔄 Updated existing person {existing_id} (confidence: {match['confidence']:.3f})")
                        matched_count += 1
                    else:
                        # Add new person
                        self.add_face(person_id, face_image, video_id, frame_number)
                        print(f"➕ Added new person {person_id}")
                        new_count += 1
            
            print(f"✅ Processed faces for video {video_id}")
            
            # Return detailed results
            return {
                'matched': matched_count,
                'new': new_count,
                'total_processed': len(faces_data),
                'video_id': video_id,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error processing video faces: {e}")
            print(f"❌ Analysis data type: {type(video_analysis_data)}")
            print(f"❌ Analysis data content: {video_analysis_data}")
            return False
    
    def get_attendance_summary(self):
        """Get attendance summary"""
        try:
            summary = {
                'total_persons': len(self.face_database),
                'total_appearances': sum(person.get('total_appearances', 0) for person in self.face_database.values()),
                'persons': {}
            }
            
            for person_id, person_data in self.face_database.items():
                # Convert set to list for JSON serialization
                videos_list = list(person_data.get('videos', [])) if isinstance(person_data.get('videos'), set) else person_data.get('videos', [])
                
                summary['persons'][person_id] = {
                    'total_appearances': person_data.get('total_appearances', 0),
                    'videos': videos_list,
                    'first_seen': person_data.get('first_seen', ''),
                    'last_seen': person_data.get('last_seen', ''),
                    'encodings_count': len(person_data.get('encodings', [])),
                    'images_count': len(person_data.get('images', []))
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {'total_persons': 0, 'total_appearances': 0, 'persons': {}}

# Test the simple enhanced face matcher
if __name__ == "__main__":
    print("🧪 Testing Simple Enhanced Face Matcher")
    print("=" * 50)
    
    matcher = SimpleEnhancedFaceMatcher()
    
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
    
    print("✅ Simple enhanced face matcher test completed!")
