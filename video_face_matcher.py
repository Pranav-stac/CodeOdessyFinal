"""
Video-to-Video Face Matching System
Matches faces across multiple videos to track student attendance and maintain consistency
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

class VideoFaceMatcher:
    def __init__(self, database_path="face_database.json"):
        """
        Initialize the face matching system
        
        Args:
            database_path (str): Path to store face database
        """
        self.database_path = database_path
        self.face_database = self.load_face_database()
        self.face_encodings_cache = {}
        self.attendance_records = defaultdict(list)
        
        # Face matching parameters
        self.face_match_threshold = 0.6  # Lower = more strict matching
        self.min_face_size = 30  # Minimum face size in pixels (reduced for small images)
        self.max_faces_per_person = 5  # Maximum face encodings to store per person
        
    def load_face_database(self):
        """Load existing face database"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                print(f"✅ Loaded face database with {len(database)} known faces")
                return database
            except Exception as e:
                print(f"❌ Error loading face database: {e}")
                return {}
        else:
            print("📝 Creating new face database")
            return {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.face_database, f, indent=2)
            print(f"💾 Saved face database with {len(self.face_database)} faces")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def extract_face_encoding(self, face_image):
        """
        Extract face encoding from image with enhanced preprocessing
        
        Args:
            face_image (numpy.ndarray): Face image as numpy array
            
        Returns:
            list: Face encoding or None if extraction fails
        """
        try:
            # Ensure image is valid
            if face_image is None or face_image.size == 0:
                print("⚠️ Invalid face image provided")
                return None
            
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Ensure image is in correct format
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Enhanced preprocessing for small images
            height, width = rgb_image.shape[:2]
            
            # For very small images, try multiple upscaling approaches
            if height < 100 or width < 100:
                
                # Method 1: Bicubic upscaling
                scale_factor = max(200/height, 200/width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                upscaled = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Method 2: Try with enhanced contrast
                enhanced = cv2.convertScaleAbs(upscaled, alpha=1.2, beta=10)
                
                # Method 3: Apply sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                # Try all three versions
                test_images = [
                    ("original", rgb_image),
                    ("upscaled", upscaled),
                    ("enhanced", enhanced),
                    ("sharpened", sharpened)
                ]
                
                for name, test_img in test_images:
                    try:
                        encodings = face_recognition.face_encodings(test_img)
                        if encodings:
                            encoding = encodings[0].tolist()
                            print(f"✅ Face encoding extracted from {name} image")
                            return encoding
                    except Exception as e:
                        continue
                
                # If all methods fail, try with face detection first
                for name, test_img in test_images:
                    try:
                        # First detect face locations
                        face_locations = face_recognition.face_locations(test_img)
                        if face_locations:
                            # Extract encodings from detected faces
                            encodings = face_recognition.face_encodings(test_img, face_locations)
                            if encodings:
                                encoding = encodings[0].tolist()
                                print(f"✅ Face encoding extracted from detected face in {name} image")
                                return encoding
                    except Exception as e:
                        continue
                
                return None
            
            else:
                # For larger images, use standard approach
                # Resize if too large
                if height > 1000 or width > 1000:
                    scale = min(1000/height, 1000/width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    rgb_image = cv2.resize(rgb_image, (new_width, new_height))
                
                # Try standard face encoding
                try:
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        encoding = encodings[0].tolist()
                        print(f"✅ Face encoding extracted successfully")
                        return encoding
                except Exception as e:
                    pass
                
                # Try with face detection
                try:
                    face_locations = face_recognition.face_locations(rgb_image)
                    if face_locations:
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        if encodings:
                            encoding = encodings[0].tolist()
                            print(f"✅ Face encoding extracted from detected face")
                            return encoding
                except Exception as e:
                    pass
                
                return None
                
        except Exception as e:
            print(f"❌ Error extracting face encoding: {e}")
            return None
    
    def calculate_face_similarity(self, encoding1, encoding2):
        """
        Calculate similarity between two face encodings
        
        Args:
            encoding1 (list): First face encoding
            encoding2 (list): Second face encoding
            
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0
            
            # Convert lists back to numpy arrays
            enc1 = np.array(encoding1)
            enc2 = np.array(encoding2)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(enc1 - enc2)
            
            # Convert distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = max(0.0, 1.0 - distance)
            
            return similarity
            
        except Exception as e:
            print(f"❌ Error calculating face similarity: {e}")
            return 0.0
    
    def find_best_match(self, new_encoding, threshold=None):
        """
        Find the best matching face in the database
        
        Args:
            new_encoding (list): Face encoding to match
            threshold (float): Minimum similarity threshold
            
        Returns:
            tuple: (person_id, similarity_score) or (None, 0.0) if no match
        """
        if threshold is None:
            threshold = self.face_match_threshold
        
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, person_data in self.face_database.items():
            stored_encodings = person_data.get('encodings', [])
            
            for stored_encoding in stored_encodings:
                similarity = self.calculate_face_similarity(new_encoding, stored_encoding)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match_id = person_id
        
        return best_match_id, best_similarity
    
    def add_new_person(self, face_encoding, face_image, metadata=None):
        """
        Add a new person to the database
        
        Args:
            face_encoding (list): Face encoding
            face_image (numpy.ndarray): Face image
            metadata (dict): Additional metadata
            
        Returns:
            str: New person ID
        """
        # Generate new person ID
        person_id = f"person_{len(self.face_database) + 1:03d}"
        
        # Convert image to base64 for storage
        try:
            _, buffer = cv2.imencode('.jpg', face_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            image_base64 = ""
        
        # Create person record
        person_record = {
            'person_id': person_id,
            'encodings': [face_encoding],
            'images': [image_base64],
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'total_appearances': 1,
            'videos_seen': [],
            'metadata': metadata or {}
        }
        
        self.face_database[person_id] = person_record
        
        print(f"✅ Added new person: {person_id}")
        return person_id
    
    def update_existing_person(self, person_id, face_encoding, face_image, metadata=None):
        """
        Update existing person with new face data
        
        Args:
            person_id (str): Person ID to update
            face_encoding (list): New face encoding
            face_image (numpy.ndarray): New face image
            metadata (dict): Additional metadata
        """
        if person_id not in self.face_database:
            print(f"❌ Person {person_id} not found in database")
            return
        
        person_record = self.face_database[person_id]
        
        # Add new encoding (limit to max_faces_per_person)
        encodings = person_record.get('encodings', [])
        if len(encodings) >= self.max_faces_per_person:
            # Remove oldest encoding
            encodings.pop(0)
        
        encodings.append(face_encoding)
        person_record['encodings'] = encodings
        
        # Add new image
        try:
            _, buffer = cv2.imencode('.jpg', face_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            image_base64 = ""
        
        images = person_record.get('images', [])
        if len(images) >= self.max_faces_per_person:
            images.pop(0)
        
        images.append(image_base64)
        person_record['images'] = images
        
        # Update metadata
        person_record['last_seen'] = datetime.now().isoformat()
        person_record['total_appearances'] += 1
        
        if metadata:
            person_record['metadata'].update(metadata)
        
        print(f"✅ Updated person {person_id} (appearances: {person_record['total_appearances']})")
    
    def process_video_faces(self, video_analysis_data, video_id=None):
        """
        Process faces from a video analysis and match with existing database
        
        Args:
            video_analysis_data (dict): Video analysis results
            video_id (str): Unique identifier for the video
            
        Returns:
            dict: Matching results with attendance updates
        """
        if video_id is None:
            video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🔍 Processing faces from video: {video_id}")
        
        matching_results = {
            'video_id': video_id,
            'processed_faces': 0,
            'matched_faces': 0,
            'new_faces': 0,
            'matches': [],
            'new_persons': []
        }
        
        # Get faces from analysis data
        faces_data = video_analysis_data.get('faces', {})
        
        for face_key, face_data in faces_data.items():
            matching_results['processed_faces'] += 1
            
            # Get best image info
            best_image_info = face_data.get('best_image_info', {})
            
            if not best_image_info.get('base64_image'):
                print(f"⚠️ No image data for {face_key}")
                continue
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(best_image_info['base64_image'])
                image = Image.open(io.BytesIO(image_data))
                face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Extract face encoding
                face_encoding = self.extract_face_encoding(face_image)
                
                if face_encoding is None:
                    # Silently skip faces that can't be encoded (reduces spam)
                    continue
                
                # Find best match in database
                matched_person_id, similarity = self.find_best_match(face_encoding)
                
                if matched_person_id:
                    # Update existing person
                    metadata = {
                        'video_id': video_id,
                        'face_id': face_data.get('face_id'),
                        'confidence': best_image_info.get('confidence'),
                        'quality_score': best_image_info.get('quality_score'),
                        'similarity': similarity
                    }
                    
                    self.update_existing_person(matched_person_id, face_encoding, face_image, metadata)
                    
                    # Update attendance record
                    self.attendance_records[matched_person_id].append({
                        'video_id': video_id,
                        'timestamp': datetime.now().isoformat(),
                        'similarity': similarity,
                        'confidence': best_image_info.get('confidence')
                    })
                    
                    matching_results['matched_faces'] += 1
                    matching_results['matches'].append({
                        'face_id': face_data.get('face_id'),
                        'matched_person_id': matched_person_id,
                        'similarity': similarity,
                        'confidence': best_image_info.get('confidence')
                    })
                    
                    print(f"✅ Matched {face_key} to {matched_person_id} (similarity: {similarity:.3f})")
                
                else:
                    # Add new person
                    metadata = {
                        'video_id': video_id,
                        'face_id': face_data.get('face_id'),
                        'confidence': best_image_info.get('confidence'),
                        'quality_score': best_image_info.get('quality_score'),
                        'first_video': video_id
                    }
                    
                    new_person_id = self.add_new_person(face_encoding, face_image, metadata)
                    
                    # Create attendance record
                    self.attendance_records[new_person_id].append({
                        'video_id': video_id,
                        'timestamp': datetime.now().isoformat(),
                        'similarity': 1.0,  # Perfect match for new person
                        'confidence': best_image_info.get('confidence')
                    })
                    
                    matching_results['new_faces'] += 1
                    matching_results['new_persons'].append({
                        'person_id': new_person_id,
                        'face_id': face_data.get('face_id'),
                        'confidence': best_image_info.get('confidence')
                    })
                    
                    print(f"🆕 Added new person {new_person_id} from {face_key}")
                
            except Exception as e:
                print(f"❌ Error processing face {face_key}: {e}")
                continue
        
        # Save updated database
        self.save_face_database()
        
        # Save attendance records
        self.save_attendance_records()
        
        print(f"📊 Video processing complete: {matching_results['matched_faces']} matched, {matching_results['new_faces']} new")
        return matching_results
    
    def get_person_attendance(self, person_id):
        """
        Get attendance record for a specific person
        
        Args:
            person_id (str): Person ID
            
        Returns:
            list: Attendance records
        """
        return self.attendance_records.get(person_id, [])
    
    def get_all_attendance_summary(self):
        """
        Get attendance summary for all persons
        
        Returns:
            dict: Attendance summary
        """
        summary = {
            'total_persons': len(self.face_database),
            'total_videos_processed': len(set(record['video_id'] for records in self.attendance_records.values() for record in records)),
            'attendance_records': {}
        }
        
        for person_id, records in self.attendance_records.items():
            person_data = self.face_database.get(person_id, {})
            
            summary['attendance_records'][person_id] = {
                'person_id': person_id,
                'total_appearances': len(records),
                'videos_attended': len(set(record['video_id'] for record in records)),
                'first_seen': person_data.get('first_seen', 'Unknown'),
                'last_seen': person_data.get('last_seen', 'Unknown'),
                'attendance_rate': f"{len(records) / max(1, len(set(record['video_id'] for records in self.attendance_records.values() for record in records))) * 100:.1f}%",
                'recent_records': records[-5:]  # Last 5 records
            }
        
        return summary
    
    def save_attendance_records(self):
        """Save attendance records to file"""
        attendance_file = "attendance_records.json"
        try:
            with open(attendance_file, 'w') as f:
                json.dump(dict(self.attendance_records), f, indent=2)
            print(f"💾 Saved attendance records to {attendance_file}")
        except Exception as e:
            print(f"❌ Error saving attendance records: {e}")
    
    def load_attendance_records(self):
        """Load attendance records from file"""
        attendance_file = "attendance_records.json"
        if os.path.exists(attendance_file):
            try:
                with open(attendance_file, 'r') as f:
                    records = json.load(f)
                self.attendance_records = defaultdict(list, records)
                print(f"✅ Loaded attendance records from {attendance_file}")
            except Exception as e:
                print(f"❌ Error loading attendance records: {e}")
    
    def generate_attendance_report(self):
        """
        Generate comprehensive attendance report
        
        Returns:
            dict: Attendance report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_all_attendance_summary(),
            'detailed_records': dict(self.attendance_records),
            'face_database_stats': {
                'total_persons': len(self.face_database),
                'persons_with_multiple_videos': len([p for p in self.face_database.values() if len(p.get('videos_seen', [])) > 1])
            }
        }
        
        # Save report
        report_file = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📋 Attendance report saved to {report_file}")
        except Exception as e:
            print(f"❌ Error saving attendance report: {e}")
        
        return report
    
    def merge_similar_faces(self, similarity_threshold=0.8):
        """
        Merge similar faces that might be the same person
        
        Args:
            similarity_threshold (float): Threshold for merging faces
        """
        print(f"🔄 Merging similar faces with threshold {similarity_threshold}")
        
        person_ids = list(self.face_database.keys())
        merged_count = 0
        
        for i, person_id1 in enumerate(person_ids):
            if person_id1 not in self.face_database:
                continue
                
            person1_data = self.face_database[person_id1]
            encodings1 = person1_data.get('encodings', [])
            
            if not encodings1:
                continue
            
            for j, person_id2 in enumerate(person_ids[i+1:], i+1):
                if person_id2 not in self.face_database:
                    continue
                
                person2_data = self.face_database[person_id2]
                encodings2 = person2_data.get('encodings', [])
                
                if not encodings2:
                    continue
                
                # Check similarity between any encodings
                max_similarity = 0.0
                for enc1 in encodings1:
                    for enc2 in encodings2:
                        similarity = self.calculate_face_similarity(enc1, enc2)
                        max_similarity = max(max_similarity, similarity)
                
                if max_similarity >= similarity_threshold:
                    # Merge person2 into person1
                    print(f"🔗 Merging {person_id2} into {person_id1} (similarity: {max_similarity:.3f})")
                    
                    # Combine encodings and images
                    person1_data['encodings'].extend(encodings2)
                    person1_data['images'].extend(person2_data.get('images', []))
                    
                    # Update metadata
                    person1_data['total_appearances'] += person2_data.get('total_appearances', 1)
                    person1_data['last_seen'] = max(
                        person1_data.get('last_seen', ''),
                        person2_data.get('last_seen', '')
                    )
                    
                    # Merge attendance records
                    if person_id2 in self.attendance_records:
                        self.attendance_records[person_id1].extend(self.attendance_records[person_id2])
                        del self.attendance_records[person_id2]
                    
                    # Remove person2 from database
                    del self.face_database[person_id2]
                    merged_count += 1
                    break  # person_id2 is no longer valid
        
        if merged_count > 0:
            self.save_face_database()
            self.save_attendance_records()
            print(f"✅ Merged {merged_count} similar faces")
        else:
            print("ℹ️ No similar faces found for merging")

def main():
    """Test the face matching system"""
    matcher = VideoFaceMatcher()
    
    # Example usage
    print("🎭 Video Face Matcher Test")
    print("=" * 40)
    
    # Load existing database
    matcher.load_face_database()
    matcher.load_attendance_records()
    
    # Generate attendance report
    report = matcher.generate_attendance_report()
    
    print(f"📊 Database contains {len(matcher.face_database)} persons")
    print(f"📋 Attendance records for {len(matcher.attendance_records)} persons")

if __name__ == "__main__":
    main()
