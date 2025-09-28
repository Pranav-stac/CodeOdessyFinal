#!/usr/bin/env python3
"""
Vector-based face matcher using image embeddings and cosine similarity
"""

import json
import numpy as np
import cv2
import base64
from PIL import Image
import io
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class VectorFaceMatcher:
    def __init__(self, database_file="vector_face_database.pkl"):
        self.database_file = database_file
        self.face_database = {}
        self.embedding_model = None
        self.transform = None
        self.scaler = StandardScaler()
        self.similarity_threshold = 0.7  # Cosine similarity threshold
        
        # Initialize the embedding model
        self.initialize_embedding_model()
        
        # Load existing database
        self.load_database()
    
    def initialize_embedding_model(self):
        """Initialize a pre-trained model for image embeddings"""
        try:
            print("🤖 Loading ResNet18 for image embeddings...")
            self.embedding_model = models.resnet18(pretrained=True)
            self.embedding_model.eval()
            
            # Remove the final classification layer to get embeddings
            self.embedding_model = torch.nn.Sequential(*list(self.embedding_model.children())[:-1])
            
            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ ResNet18 embedding model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            self.embedding_model = None
            self.transform = None
    
    def extract_image_embedding(self, image):
        """Extract embedding vector from image"""
        if self.embedding_model is None or self.transform is None:
            return None
        
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Ensure uint8
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Resize image if too small
            height, width = rgb_image.shape[:2]
            if height < 50 or width < 50:
                # Upscale small images
                scale_factor = max(50 / height, 50 / width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to tensor
            tensor_image = self.transform(rgb_image).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model(tensor_image)
                embedding = embedding.squeeze().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"⚠️ Error extracting embedding: {e}")
            return None
    
    def add_face(self, person_id, face_image, video_id, frame_number):
        """Add a new face to the database"""
        try:
            # Extract embedding
            embedding = self.extract_image_embedding(face_image)
            if embedding is None:
                print(f"⚠️ Could not extract embedding for {person_id}")
                return False
            
            # Store in database
            if person_id not in self.face_database:
                self.face_database[person_id] = {
                    'embeddings': [],
                    'images': [],
                    'videos': set(),
                    'frame_numbers': [],
                    'metadata': []
                }
            
            self.face_database[person_id]['embeddings'].append(embedding)
            self.face_database[person_id]['images'].append(face_image)
            self.face_database[person_id]['videos'].add(video_id)
            self.face_database[person_id]['frame_numbers'].append(frame_number)
            self.face_database[person_id]['metadata'].append({
                'video_id': video_id,
                'frame_number': frame_number,
                'timestamp': frame_number / 25.0  # Assuming 25 FPS
            })
            
            print(f"✅ Added face embedding for {person_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding face: {e}")
            return False
    
    def find_matching_face(self, face_image, used_person_ids=None):
        """Find the most similar face in the database (one-to-one mapping)"""
        try:
            if used_person_ids is None:
                used_person_ids = set()
            
            # Extract embedding for the query image
            query_embedding = self.extract_image_embedding(face_image)
            if query_embedding is None:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # Search through all faces in database (excluding already used person IDs)
            for person_id, person_data in self.face_database.items():
                if person_id in used_person_ids:
                    continue  # Skip already matched persons
                
                embeddings = person_data['embeddings']
                
                if not embeddings:
                    continue
                
                # Calculate similarity with all embeddings for this person
                similarities = []
                for embedding in embeddings:
                    # Reshape for cosine similarity calculation
                    query_reshaped = query_embedding.reshape(1, -1)
                    embedding_reshaped = embedding.reshape(1, -1)
                    
                    similarity = cosine_similarity(query_reshaped, embedding_reshaped)[0][0]
                    similarities.append(similarity)
                
                # Use the maximum similarity for this person
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity and max_similarity > self.similarity_threshold:
                    best_similarity = max_similarity
                    best_match = {
                        'person_id': person_id,
                        'similarity': max_similarity,
                        'confidence': max_similarity
                    }
            
            return best_match
            
        except Exception as e:
            print(f"❌ Error finding matching face: {e}")
            return None
    
    def process_video_faces(self, video_analysis_data, video_id=None):
        """Process faces from video analysis data"""
        try:
            if video_id is None:
                video_id = "unknown"
            
            # Get faces data
            faces_data = video_analysis_data.get('faces', {})
            
            if isinstance(faces_data, dict):
                faces_list = list(faces_data.values())
                print(f"🔄 Converted faces dictionary to list: {len(faces_list)} faces")
            elif isinstance(faces_data, list):
                faces_list = faces_data
            else:
                print(f"⚠️ Invalid faces data type: {type(faces_data)}")
                return False
            
            print(f"🎬 Processing {len(faces_list)} faces from video {video_id}")
            
            matched_count = 0
            new_count = 0
            used_person_ids = set()  # Track used person IDs for one-to-one mapping
            
            for face_data in faces_list:
                if not isinstance(face_data, dict):
                    print(f"⚠️ Skipping invalid face data: {type(face_data)}")
                    continue
                
                # Get face information
                person_id = face_data.get('person_id', face_data.get('face_id', 'unknown'))
                face_image = face_data.get('face_image')
                frame_number = face_data.get('frame_number', 0)
                
                # If no direct face_image, try to get from best_image_info
                if face_image is None and 'best_image_info' in face_data:
                    best_image_info = face_data['best_image_info']
                    if 'base64_image' in best_image_info:
                        try:
                            # Decode base64 image
                            base64_data = best_image_info['base64_image']
                            image_data = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_data))
                            face_image = np.array(pil_image)
                            print(f"🔄 Decoded base64 image for {person_id}")
                        except Exception as e:
                            print(f"⚠️ Could not decode base64 image for {person_id}: {e}")
                            face_image = None
                
                if face_image is not None:
                    # Try to find matching face (excluding already used person IDs)
                    match = self.find_matching_face(face_image, used_person_ids)
                    
                    if match:
                        # Update existing person
                        existing_id = match['person_id']
                        self.add_face(existing_id, face_image, video_id, frame_number)
                        used_person_ids.add(existing_id)  # Mark as used
                        print(f"🔄 Updated existing person {existing_id} (similarity: {match['similarity']:.3f})")
                        matched_count += 1
                    else:
                        # Add new person
                        self.add_face(person_id, face_image, video_id, frame_number)
                        used_person_ids.add(person_id)  # Mark as used
                        print(f"➕ Added new person {person_id}")
                        new_count += 1
                else:
                    print(f"⚠️ No face image available for {person_id}")
            
            print(f"✅ Processed faces for video {video_id}")
            
            # Save database
            self.save_database()
            
            # Return detailed results
            return {
                'matched': matched_count,
                'new': new_count,
                'total_processed': len(faces_list),
                'video_id': video_id,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error processing video faces: {e}")
            return False
    
    def get_attendance_summary(self):
        """Get attendance summary"""
        try:
            summary = {
                'total_persons': len(self.face_database),
                'total_videos': set(),
                'total_appearances': 0,
                'persons': []
            }
            
            for person_id, person_data in self.face_database.items():
                videos = list(person_data['videos'])
                appearances = len(person_data['embeddings'])
                
                summary['total_videos'].update(videos)
                summary['total_appearances'] += appearances
                
                person_info = {
                    'person_id': person_id,
                    'total_appearances': appearances,
                    'videos_attended': videos,
                    'attendance_rate': len(videos) / max(len(summary['total_videos']), 1),
                    'first_seen': min(person_data['frame_numbers']) if person_data['frame_numbers'] else 0,
                    'last_seen': max(person_data['frame_numbers']) if person_data['frame_numbers'] else 0
                }
                
                summary['persons'].append(person_info)
            
            summary['total_videos'] = list(summary['total_videos'])
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {}
    
    def map_attendance_by_person_id(self, video_analysis_data, video_id=None):
        """Map attendance and engagement data by person ID"""
        try:
            if video_id is None:
                video_id = "unknown"
            
            # Get faces data
            faces_data = video_analysis_data.get('faces', {})
            
            if isinstance(faces_data, dict):
                faces_list = list(faces_data.values())
            elif isinstance(faces_data, list):
                faces_list = faces_data
            else:
                print(f"⚠️ Invalid faces data type: {type(faces_data)}")
                return {}
            
            # Create person ID mapping
            person_mapping = {}
            used_person_ids = set()
            
            for face_data in faces_list:
                if not isinstance(face_data, dict):
                    continue
                
                face_id = face_data.get('face_id', face_data.get('person_id', 'unknown'))
                face_image = face_data.get('face_image')
                
                # Decode base64 image if needed
                if face_image is None and 'best_image_info' in face_data:
                    best_image_info = face_data['best_image_info']
                    if 'base64_image' in best_image_info:
                        try:
                            base64_data = best_image_info['base64_image']
                            image_data = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_data))
                            face_image = np.array(pil_image)
                        except Exception as e:
                            face_image = None
                
                if face_image is not None:
                    # Find matching person ID
                    match = self.find_matching_face(face_image, used_person_ids)
                    
                    if match:
                        person_id = match['person_id']
                        used_person_ids.add(person_id)
                    else:
                        person_id = face_id
                        used_person_ids.add(person_id)
                    
                    person_mapping[face_id] = person_id
            
            # Map engagement data by person ID
            engagement_by_person = {}
            students_data = video_analysis_data.get('students', {})
            
            for student_id, student_data in students_data.items():
                # Find corresponding person ID
                person_id = person_mapping.get(student_id, student_id)
                
                if person_id not in engagement_by_person:
                    engagement_by_person[person_id] = {
                        'person_id': person_id,
                        'total_frames': 0,
                        'engagement_score': 0.0,
                        'activity_breakdown': {},
                        'zone_analysis': {},
                        'face_detections': 0,
                        'videos': set()
                    }
                
                # Aggregate engagement data
                engagement_by_person[person_id]['total_frames'] += student_data.get('total_frames', 0)
                engagement_by_person[person_id]['engagement_score'] += student_data.get('engagement_score', 0.0)
                engagement_by_person[person_id]['face_detections'] += 1
                engagement_by_person[person_id]['videos'].add(video_id)
                
                # Aggregate activity breakdown
                activity_breakdown = student_data.get('activity_breakdown', {})
                for activity, count in activity_breakdown.items():
                    if activity not in engagement_by_person[person_id]['activity_breakdown']:
                        engagement_by_person[person_id]['activity_breakdown'][activity] = 0
                    engagement_by_person[person_id]['activity_breakdown'][activity] += count
                
                # Aggregate zone analysis
                zone_analysis = student_data.get('zone_analysis', {})
                for zone, data in zone_analysis.items():
                    if zone not in engagement_by_person[person_id]['zone_analysis']:
                        engagement_by_person[person_id]['zone_analysis'][zone] = {'time_spent': 0, 'engagement': 0.0}
                    engagement_by_person[person_id]['zone_analysis'][zone]['time_spent'] += data.get('time_spent', 0)
                    engagement_by_person[person_id]['zone_analysis'][zone]['engagement'] += data.get('engagement', 0.0)
            
            # Convert sets to lists for JSON serialization
            for person_data in engagement_by_person.values():
                person_data['videos'] = list(person_data['videos'])
            
            return {
                'person_mapping': person_mapping,
                'engagement_by_person': engagement_by_person,
                'total_unique_persons': len(engagement_by_person),
                'video_id': video_id
            }
            
        except Exception as e:
            print(f"❌ Error mapping attendance by person ID: {e}")
            return {}
    
    def get_person_engagement_summary(self, video_analysis_data, video_id=None):
        """Get comprehensive engagement summary by person ID"""
        try:
            mapping_result = self.map_attendance_by_person_id(video_analysis_data, video_id)
            
            if not mapping_result:
                return {}
            
            engagement_by_person = mapping_result['engagement_by_person']
            
            # Calculate summary statistics
            total_persons = len(engagement_by_person)
            total_engagement = sum(person['engagement_score'] for person in engagement_by_person.values())
            avg_engagement = total_engagement / max(total_persons, 1)
            
            # Find most/least engaged persons
            sorted_persons = sorted(engagement_by_person.items(), 
                                  key=lambda x: x[1]['engagement_score'], reverse=True)
            
            most_engaged = sorted_persons[0] if sorted_persons else None
            least_engaged = sorted_persons[-1] if sorted_persons else None
            
            summary = {
                'video_id': video_id,
                'total_persons': total_persons,
                'total_engagement_score': total_engagement,
                'average_engagement': avg_engagement,
                'most_engaged_person': {
                    'person_id': most_engaged[0],
                    'engagement_score': most_engaged[1]['engagement_score']
                } if most_engaged else None,
                'least_engaged_person': {
                    'person_id': least_engaged[0],
                    'engagement_score': least_engaged[1]['engagement_score']
                } if least_engaged else None,
                'person_details': engagement_by_person,
                'person_mapping': mapping_result['person_mapping']
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting person engagement summary: {e}")
            return {}
    
    def save_database(self):
        """Save face database to file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"💾 Saved face database to {self.database_file}")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def load_database(self):
        """Load face database from file"""
        try:
            if Path(self.database_file).exists():
                with open(self.database_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"✅ Loaded face database with {len(self.face_database)} persons")
            else:
                print("📝 Creating new face database")
                self.face_database = {}
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.face_database = {}

if __name__ == "__main__":
    # Test the vector face matcher
    matcher = VectorFaceMatcher()
    
    # Load test data
    with open("realtime_analysis/comprehensive_analysis_report.json", 'r') as f:
        analysis_data = json.load(f)
    
    # Process faces
    results = matcher.process_video_faces(analysis_data, "test_video")
    print(f"Results: {results}")
    
    # Get summary
    summary = matcher.get_attendance_summary()
    print(f"Summary: {summary}")
