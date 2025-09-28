"""
Enhanced Face Matcher with Image Similarity Fallback
Uses both face encodings and image similarity for better matching
"""

import cv2
import numpy as np
import json
import os
import base64
from datetime import datetime
from collections import defaultdict
import face_recognition
from PIL import Image
import io
import math
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedFaceMatcher:
    def __init__(self, database_path="enhanced_face_database.json"):
        """
        Initialize the enhanced face matching system
        
        Args:
            database_path (str): Path to store face database
        """
        self.database_path = database_path
        self.face_database = self.load_face_database()
        self.attendance_records = defaultdict(list)
        
        # Face matching parameters
        self.face_match_threshold = 0.6  # For face encodings
        self.image_similarity_threshold = 0.7  # For image similarity
        self.min_face_size = 30
        self.max_faces_per_person = 5
        
    def load_face_database(self):
        """Load existing face database"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                print(f"✅ Loaded enhanced face database with {len(database)} known faces")
                return database
            except Exception as e:
                print(f"❌ Error loading face database: {e}")
                return {}
        else:
            print("📝 Creating new enhanced face database")
            return {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.face_database, f, indent=2)
            print(f"💾 Saved enhanced face database with {len(self.face_database)} faces")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def extract_face_encoding(self, face_image):
        """Extract face encoding with enhanced preprocessing"""
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
            
            height, width = rgb_image.shape[:2]
            
            # For small images, try multiple upscaling approaches
            if height < 100 or width < 100:
                # Method 1: Bicubic upscaling
                scale_factor = max(200/height, 200/width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                upscaled = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Method 2: Enhanced contrast
                enhanced = cv2.convertScaleAbs(upscaled, alpha=1.2, beta=10)
                
                # Method 3: Sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                # Try all versions
                test_images = [rgb_image, upscaled, enhanced, sharpened]
                
                for test_img in test_images:
                    try:
                        encodings = face_recognition.face_encodings(test_img)
                        if encodings:
                            return encodings[0].tolist()
                    except:
                        continue
                
                # Try face detection approach
                for test_img in test_images:
                    try:
                        face_locations = face_recognition.face_locations(test_img)
                        if face_locations:
                            encodings = face_recognition.face_encodings(test_img, face_locations)
                            if encodings:
                                return encodings[0].tolist()
                    except:
                        continue
                
                return None
            else:
                # Standard approach for larger images
                try:
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        return encodings[0].tolist()
                except:
                    pass
                
                try:
                    face_locations = face_recognition.face_locations(rgb_image)
                    if face_locations:
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        if encodings:
                            return encodings[0].tolist()
                except:
                    pass
                
                return None
                
        except Exception as e:
            return None
    
    def extract_image_features(self, face_image):
        """Extract image features for similarity matching"""
        try:
            if face_image is None or face_image.size == 0:
                return None
            
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Resize to standard size
            resized = cv2.resize(gray, (64, 64))
            
            # Extract features using histogram
            hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalize
            hist = hist / (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            return None
    
    def calculate_image_similarity(self, features1, features2):
        """Calculate similarity between image features"""
        try:
            if features1 is None or features2 is None:
                return 0.0
            
            # Use cosine similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def calculate_face_similarity(self, encoding1, encoding2):
        """Calculate similarity between face encodings"""
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0
            
            enc1 = np.array(encoding1)
            enc2 = np.array(encoding2)
            
            distance = np.linalg.norm(enc1 - enc2)
            similarity = max(0.0, 1.0 - distance)
            
            return similarity
            
        except Exception as e:
            return 0.0
    
    def find_best_match(self, face_encoding, image_features):
        """Find best match using both face encoding and image similarity"""
        best_match_id = None
        best_similarity = 0.0
        best_method = None
        
        for person_id, person_data in self.face_database.items():
            # Try face encoding matching first
            if face_encoding is not None:
                stored_encodings = person_data.get('encodings', [])
                for stored_encoding in stored_encodings:
                    similarity = self.calculate_face_similarity(face_encoding, stored_encoding)
                    if similarity > best_similarity and similarity >= self.face_match_threshold:
                        best_similarity = similarity
                        best_match_id = person_id
                        best_method = "face_encoding"
            
            # Try image similarity matching
            if image_features is not None:
                stored_features = person_data.get('image_features', [])
                for stored_feature in stored_features:
                    similarity = self.calculate_image_similarity(image_features, stored_feature)
                    if similarity > best_similarity and similarity >= self.image_similarity_threshold:
                        best_similarity = similarity
                        best_match_id = person_id
                        best_method = "image_similarity"
        
        return best_match_id, best_similarity, best_method
    
    def add_person(self, person_id, face_image, face_encoding=None, image_features=None, metadata=None):
        """Add a new person to the database"""
        if person_id not in self.face_database:
            self.face_database[person_id] = {
                'person_id': person_id,
                'encodings': [],
                'image_features': [],
                'images': [],
                'total_appearances': 0,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
        
        person_record = self.face_database[person_id]
        
        # Add face encoding
        if face_encoding is not None:
            encodings = person_record.get('encodings', [])
            if len(encodings) >= self.max_faces_per_person:
                encodings.pop(0)
            encodings.append(face_encoding)
            person_record['encodings'] = encodings
        
        # Add image features
        if image_features is not None:
            features = person_record.get('image_features', [])
            if len(features) >= self.max_faces_per_person:
                features.pop(0)
            features.append(image_features.tolist())
            person_record['image_features'] = features
        
        # Add image
        try:
            _, buffer = cv2.imencode('.jpg', face_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except:
            image_base64 = ""
        
        images = person_record.get('images', [])
        if len(images) >= self.max_faces_per_person:
            images.pop(0)
        images.append(image_base64)
        person_record['images'] = images
        
        # Update metadata
        person_record['total_appearances'] += 1
        person_record['last_seen'] = datetime.now().isoformat()
        
        if metadata:
            person_record['metadata'].update(metadata)
        
        print(f"✅ Added person {person_id} (appearances: {person_record['total_appearances']})")
    
    def process_video_faces(self, video_analysis_data, video_id=None):
        """Process faces from video analysis with enhanced matching"""
        try:
            print(f"🔍 Processing faces from video: {video_id or 'unknown'}")
            
            # Debug: Print analysis data structure
            print(f"📊 Analysis data keys: {list(video_analysis_data.keys())}")
            
            faces_data = video_analysis_data.get('faces', {})
            if not faces_data:
                print("⚠️ No face data found in video analysis")
                return {'matched': 0, 'new': 0, 'total_processed': 0}
            
            print(f"📊 Found {len(faces_data)} faces to process")
            
            matched_count = 0
            new_count = 0
            total_processed = 0
            
            for face_key, face_data in faces_data.items():
                try:
                    # Ensure face_data is a dictionary
                    if not isinstance(face_data, dict):
                        print(f"⚠️ Face data for {face_key} is not a dictionary: {type(face_data)}")
                        continue
                    
                    # Get best face image
                    best_image_info = face_data.get('best_image_info', {})
                    if not best_image_info:
                        continue
                    
                    # Decode image - try different possible keys
                    image_data = best_image_info.get('image_data') or best_image_info.get('base64_image')
                    if not image_data:
                        continue
                    
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Extract both face encoding and image features
                    face_encoding = self.extract_face_encoding(face_image)
                    image_features = self.extract_image_features(face_image)
                    
                    total_processed += 1
                    
                    # Find best match
                    matched_person_id, similarity, method = self.find_best_match(face_encoding, image_features)
                    
                    if matched_person_id:
                        # Update existing person
                        metadata = {
                            'video_id': video_id,
                            'face_id': face_data.get('face_id'),
                            'confidence': best_image_info.get('confidence'),
                            'quality_score': best_image_info.get('quality_score'),
                            'similarity': similarity,
                            'method': method
                        }
                        
                        self.add_person(matched_person_id, face_image, face_encoding, image_features, metadata)
                        matched_count += 1
                        print(f"✅ Matched {face_key} to person {matched_person_id} (similarity: {similarity:.3f}, method: {method})")
                    else:
                        # Add new person
                        new_person_id = f"person_{len(self.face_database) + 1}"
                        metadata = {
                            'video_id': video_id,
                            'face_id': face_data.get('face_id'),
                            'confidence': best_image_info.get('confidence'),
                            'quality_score': best_image_info.get('quality_score'),
                            'first_video': video_id
                        }
                        
                        self.add_person(new_person_id, face_image, face_encoding, image_features, metadata)
                        new_count += 1
                        print(f"➕ Added new person {new_person_id} for {face_key}")
                
                except Exception as e:
                    print(f"⚠️ Error processing {face_key}: {e}")
                    continue
            
            # Save database
            self.save_face_database()
            
            print(f"📊 Video processing complete: {matched_count} matched, {new_count} new")
            return {
                'matched': matched_count,
                'new': new_count,
                'total_processed': total_processed
            }
            
        except Exception as e:
            print(f"❌ Error processing video faces: {e}")
            return {'matched': 0, 'new': 0, 'total_processed': 0}
    
    def get_attendance_summary(self):
        """Get attendance summary"""
        summary = {}
        for person_id, person_data in self.face_database.items():
            # Get unique video IDs from metadata
            videos = set()
            metadata = person_data.get('metadata', {})
            for key, meta in metadata.items():
                if isinstance(meta, dict) and 'video_id' in meta:
                    videos.add(meta['video_id'])
                elif isinstance(meta, str) and meta:  # Handle case where metadata values are strings
                    videos.add(meta)
            
            summary[person_id] = {
                'total_appearances': person_data.get('total_appearances', 0),
                'first_seen': person_data.get('first_seen'),
                'last_seen': person_data.get('last_seen'),
                'videos': list(videos)
            }
        return summary

# Test the enhanced face matcher
if __name__ == "__main__":
    print("🧪 Testing Enhanced Face Matcher...")
    
    matcher = EnhancedFaceMatcher()
    
    # Test with a face image
    test_image_path = "realtime_analysis/face_images/face_1001_best.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        if img is not None:
            print(f"📸 Testing with: {test_image_path}")
            
            # Test face encoding
            encoding = matcher.extract_face_encoding(img)
            print(f"Face encoding: {'✅' if encoding else '❌'}")
            
            # Test image features
            features = matcher.extract_image_features(img)
            print(f"Image features: {'✅' if features is not None else '❌'}")
            
            if features is not None:
                print(f"Feature vector length: {len(features)}")
        else:
            print("❌ Could not load test image")
    else:
        print("❌ Test image not found")
