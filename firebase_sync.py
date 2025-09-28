"""
Firebase Realtime Database Integration for Classroom Analyzer
Syncs all collected data including engagement, attendance, face data, and lecture types
"""

import json
import os
from datetime import datetime, date
from pathlib import Path
import base64
from typing import Dict, List, Any, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
    print("✅ Firebase SDK available")
except ImportError as e:
    FIREBASE_AVAILABLE = False
    print(f"⚠️ Firebase SDK not available: {e}")
    print("📝 Install with: pip install firebase-admin")

class FirebaseSync:
    def __init__(self, firebase_config: Dict[str, str]):
        """
        Initialize Firebase sync with configuration
        
        Args:
            firebase_config: Firebase configuration dictionary
        """
        self.firebase_config = firebase_config
        self.app = None
        self.db_ref = None
        self.initialized = False
        
        if FIREBASE_AVAILABLE:
            self.initialize_firebase()
        else:
            print("❌ Firebase not available. Please install firebase-admin")
    
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if service account file exists
            service_account_file = "firebase_service_account.json"
            if os.path.exists(service_account_file):
                # Use service account file
                cred = credentials.Certificate(service_account_file)
                print("✅ Using Firebase service account file")
            else:
                # Create credentials from config (fallback)
                cred = credentials.Certificate({
                    "type": "service_account",
                    "project_id": self.firebase_config["projectId"],
                    "private_key_id": "9774817afd9b8418fcc3d1e992bb7f7efa1d2a7e",
                    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDKH+M9rEVpYKaZ\naQPY0SUmUTWQxX1ehlBF5k7uRAxRqPydLOL6O+JUBOVltRsr0AQkZ9BiHSjhgkQI\njB+cXCcXyTpG1HJN+f5w9rbmgVgQu6sWX3RF86vZ4JH5r4k/bO2kYMOvCV7OH1KG\nlc5e3eNOK08KquROVUHsJwxQQfRrMeP29ljaETEzDDG8dbC+wm9WTD+KvZVA+cEs\nO7H9pKTmRKim1XWFRxuF0865okVflSSUa2iOr4VeZR94QjsyMQ35BWI64HVUrIWs\nj7leLMWhTZbPKefGWJnrrBjiLu88SvQvA9MT+2cFIPHSdVt/3L5bLT+5T6uSviHe\nH9EjUrFXAgMBAAECggEAJVjmBrfFrZZ3wiH23YGroRxhTupG44fi3j+TdxAozI/y\neWJZYE5fYBH1hykDJ8HMeiOai/6CVsTYqFGn557ZdFoww/SuWD0oqBvWKNpeeN0M\nXFpv5BsFtoD5yA6IyCjq8jQVBEsxctmzyH12v+20+fUjqj9wZw722tlOO6gsLwAZ\nYGejsxKqn+8E7qMGokPuNE81Bg0KUISf4X2USqqNn/A5ZJ8DKki534EVB7mxqlce\nPOAEk2EybH3mQAD2lKSKEeNWBRWKPxwrKaOE2uJ2pgrSLhN8VebjUQdxSlaAbcxa\n/0ji9UcL+znAdb7tBG9O3iDtlz6Np7oDhdpmldKsiQKBgQDn/5nS13kydWibiBw+\nHMyFAoROjJ/AwxZhQVi/rHKjXIyhDwckv/q9LogVGHp5s6Tj7SE935GhfLqlxeCC\nv4xPFSz3p/qGbF0GNUD+U9eY7hn+H3LRb8ytvdd0stCtKuJJ/G4fngqvGZ+hrlFA\nbtJ4Avdn1ldoDPthcPmB1W2iPwKBgQDfCRZbL4qGQmXZVSZQXlIJD3UAHPg5XLA8\n9ZsFVp8CrgJ1RdX+dvmvW1hbFUECAysggnUq2veA2F4sqcMyfUDlGlY5lSNGmEOc\neYtqExa0NcP1r2/V10wwrP7SAEAmAiD+FyWesoSh7ZqKTG0xdQxyI2fmMe0dQ1LJ\nZIlansh66QKBgCv80KOjaz1f+YeT1RcJytVlVsS18QxRcQrbowIkpk/HGnrnKImV\nROtdyTuGuqIcp6T3rxfWLfyac6E+1YS04NuVvkLuvJeEMFce/cW7C+PZMWB3ggOn\n2P0QQ6vCw8IxoVo53H9uLcRpbVTwgkNfP9S1a0dq4oO+AmPFUemGGBVtAoGAOfaa\noMgSMCJBZeIDOw/IMiKaPZVJzV6RTsZlq1V/raqbaO7lVmSFL1WF+OXlJyi7pI9C\n9AhEGnAgyWq1GAIFQ4U8s2pW6JnZuGi+GqsrU1pFuywE3IY/fsb20ozOHxKekGpj\nlILAhXTCnP6PwjKxMViSP/jprpk4gq5mI+7wG0ECgYBYeyaounBX+X/n5QB2LNbo\nQhpq7p35Zc+TNezQIbJdeNl/umuZmUCziv2mB4Ujh3aYtQJuD5Dre8VS2l0BXXJ5\nWZJKScJWWerShKH+qitcj6OuJpACfoz44SQ42/DGQLriiQZp4off3oxgpnqtbmlh\nxucFGHyMEMiq6Kw8rEHWEQ==\n-----END PRIVATE KEY-----\n",
                    "client_email": "firebase-adminsdk-fbsvc@code4-509a0.iam.gserviceaccount.com",
                    "client_id": "106393700306260764255",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40code4-509a0.iam.gserviceaccount.com",
                    "universe_domain": "googleapis.com"
                })
                print("✅ Using Firebase config credentials")
            
            # Initialize Firebase app (check if already initialized)
            try:
                self.app = firebase_admin.initialize_app(cred, {
                    'databaseURL': self.firebase_config['databaseURL']
                })
            except ValueError as e:
                if "already exists" in str(e):
                    # App already initialized, get existing app
                    self.app = firebase_admin.get_app()
                else:
                    raise e
            
            # Get database reference
            self.db_ref = db.reference()
            self.initialized = True
            print("✅ Firebase initialized successfully")
            
        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            print("📝 Using local storage only")
            self.initialized = False
    
    def sync_daily_data(self, date_str: str = None) -> bool:
        """
        Sync all data for a specific date to Firebase
        
        Args:
            date_str: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            bool: True if sync successful, False otherwise
        """
        if not self.initialized:
            print("⚠️ Firebase not initialized. Data will be saved locally only.")
            return self.save_local_backup(date_str)
        
        if date_str is None:
            date_str = date.today().strftime("%Y-%m-%d")
        
        try:
            print(f"🔄 Syncing data for {date_str} to Firebase...")
            
            # Collect all data for the date
            daily_data = self.collect_daily_data(date_str)
            
            if not daily_data:
                print(f"⚠️ No data found for {date_str}")
                return False
            
            # Sync to Firebase
            self.db_ref.child('classroom_analyzer').child('daily_data').child(date_str).set(daily_data)
            
            print(f"✅ Successfully synced data for {date_str}")
            return True
            
        except Exception as e:
            print(f"❌ Firebase sync failed: {e}")
            print("📝 Saving local backup...")
            return self.save_local_backup(date_str)
    
    def collect_daily_data(self, date_str: str) -> Dict[str, Any]:
        """
        Collect all data for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Dict containing all collected data
        """
        daily_data = {
            'date': date_str,
            'sync_timestamp': datetime.now().isoformat(),
            'engagement_data': {},
            'attendance_data': {},
            'face_data': {},
            'lecture_classifications': {},
            'video_metadata': {},
            'analysis_reports': {},
            'statistics': {}
        }
        
        # 1. Collect comprehensive analysis report as single JSON string
        daily_data['comprehensive_analysis_report'] = self.collect_comprehensive_analysis_report(date_str)
        
        # 2. Collect face database as single JSON string
        daily_data['face_database'] = self.collect_face_database_json(date_str)
        
        # 3. Collect attendance data as single JSON string
        daily_data['attendance_data'] = self.collect_attendance_data_json(date_str)
        
        # 4. Collect video metadata as single JSON string
        daily_data['video_metadata'] = self.collect_video_metadata_json(date_str)
        
        # 5. Collect lecture classifications as single JSON string
        daily_data['lecture_classifications'] = self.collect_lecture_classifications_json(date_str)
        
        # 6. Collect all analysis reports as single JSON string
        daily_data['analysis_reports'] = self.collect_analysis_reports_json(date_str)
        
        # 7. Collect raw data files as single JSON string
        daily_data['raw_data_files'] = self.collect_raw_data_files_json(date_str)
        
        # 8. Calculate statistics as single JSON string
        daily_data['statistics'] = self.calculate_daily_statistics_json(daily_data)
        
        return daily_data
    
    def collect_comprehensive_analysis_report(self, date_str: str) -> str:
        """Collect comprehensive analysis report as JSON string"""
        analysis_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis"
        ]
        
        for analysis_dir in analysis_dirs:
            if os.path.exists(analysis_dir):
                comprehensive_file = os.path.join(analysis_dir, "comprehensive_analysis_report.json")
                if os.path.exists(comprehensive_file):
                    try:
                        with open(comprehensive_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"⚠️ Error reading comprehensive analysis report: {e}")
        
        return json.dumps({}, ensure_ascii=False, indent=2)
    
    def collect_face_database_json(self, date_str: str) -> str:
        """Collect face database as JSON string"""
        try:
            from enhanced_face_matcher import EnhancedFaceMatcher
            face_matcher = EnhancedFaceMatcher()
            face_matcher.load_face_database()
            
            if hasattr(face_matcher, 'face_database'):
                return json.dumps(face_matcher.face_database, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Error collecting face database: {e}")
        
        return json.dumps({}, ensure_ascii=False, indent=2)
    
    def collect_attendance_data_json(self, date_str: str) -> str:
        """Collect attendance data as JSON string"""
        try:
            from enhanced_face_matcher import EnhancedFaceMatcher
            face_matcher = EnhancedFaceMatcher()
            face_matcher.load_face_database()
            
            if hasattr(face_matcher, 'get_attendance_summary'):
                attendance_summary = face_matcher.get_attendance_summary()
                return json.dumps(attendance_summary, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Error collecting attendance data: {e}")
        
        return json.dumps({}, ensure_ascii=False, indent=2)
    
    def collect_video_metadata_json(self, date_str: str) -> str:
        """Collect video metadata as JSON string"""
        video_metadata = {
            'processed_videos': [],
            'video_stats': {},
            'all_video_files': []
        }
        
        processed_dirs = [
            f"video_processing/{date_str}/processed",
            f"analysis_history/{date_str}",
            f"video_processing/{date_str}/input"
        ]
        
        for processed_dir in processed_dirs:
            if os.path.exists(processed_dir):
                for video_file in os.listdir(processed_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        video_path = os.path.join(processed_dir, video_file)
                        try:
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                duration = frame_count / fps if fps > 0 else 0
                                
                                video_info = {
                                    'filename': video_file,
                                    'file_path': video_path,
                                    'fps': fps,
                                    'frame_count': frame_count,
                                    'width': width,
                                    'height': height,
                                    'duration': duration,
                                    'file_size': os.path.getsize(video_path),
                                    'last_modified': os.path.getmtime(video_path),
                                    'directory': processed_dir
                                }
                                
                                video_metadata['processed_videos'].append(video_info)
                                video_metadata['all_video_files'].append(video_info)
                                cap.release()
                        except Exception as e:
                            print(f"⚠️ Error reading video metadata for {video_file}: {e}")
        
        # Calculate video stats
        if video_metadata['processed_videos']:
            videos = video_metadata['processed_videos']
            video_metadata['video_stats'] = {
                'total_videos': len(videos),
                'total_duration': sum(v['duration'] for v in videos),
                'total_frames': sum(v['frame_count'] for v in videos),
                'average_fps': sum(v['fps'] for v in videos) / len(videos),
                'total_size': sum(v['file_size'] for v in videos),
                'average_duration': sum(v['duration'] for v in videos) / len(videos),
                'resolution_distribution': {
                    'widths': list(set(v['width'] for v in videos)),
                    'heights': list(set(v['height'] for v in videos))
                }
            }
        
        return json.dumps(video_metadata, ensure_ascii=False, indent=2)
    
    def collect_lecture_classifications_json(self, date_str: str) -> str:
        """Collect lecture classifications as JSON string"""
        classifications = {
            'classifications': {},
            'classification_method': 'vision_llm',
            'confidence_scores': {},
            'total_classifications': 0
        }
        
        try:
            # Look for lecture classification data in analysis results
            analysis_dirs = [
                f"analysis_results",
                f"analysis_history/{date_str}",
                f"realtime_analysis"
            ]
            
            for analysis_dir in analysis_dirs:
                if os.path.exists(analysis_dir):
                    # Look for comprehensive analysis reports
                    for root, dirs, files in os.walk(analysis_dir):
                        for file in files:
                            if file == "comprehensive_analysis_report.json":
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    
                                    # Extract lecture classification data
                                    if 'lecture_classification' in data:
                                        lecture_data = data['lecture_classification']
                                        video_name = os.path.basename(root)
                                        
                                        classifications['classifications'][video_name] = {
                                            'lecture_type': lecture_data.get('lecture_type', 'unknown'),
                                            'confidence': lecture_data.get('confidence', 0.0),
                                            'method': lecture_data.get('method', 'unknown'),
                                            'timestamp': lecture_data.get('timestamp', ''),
                                            'video_path': lecture_data.get('video_path', '')
                                        }
                                        
                                        classifications['confidence_scores'][video_name] = lecture_data.get('confidence', 0.0)
                                        classifications['total_classifications'] += 1
                                        
                                        # Update classification method
                                        if lecture_data.get('method') == 'vision_llm':
                                            classifications['classification_method'] = 'vision_llm'
                                        elif lecture_data.get('method') == 'rule_based':
                                            classifications['classification_method'] = 'rule_based'
                                            
                                except Exception as e:
                                    print(f"⚠️ Error reading {file_path}: {e}")
                                    continue
            
            # If no classifications found, try to get from current analysis
            if classifications['total_classifications'] == 0:
                try:
                    from lightweight_vision_classifier import LightweightVisionClassifier
                    vision_classifier = LightweightVisionClassifier()
                    classifications['classification_method'] = 'vision_llm'
                    print("✅ Vision LLM classifier available for lecture classification")
                except Exception as e:
                    print(f"⚠️ Vision classifier not available: {e}")
                    classifications['classification_method'] = 'not_available'
                
        except Exception as e:
            print(f"⚠️ Error collecting lecture classifications: {e}")
        
        return json.dumps(classifications, ensure_ascii=False, indent=2)
    
    def collect_analysis_reports_json(self, date_str: str) -> str:
        """Collect all analysis reports as JSON string"""
        reports = {
            'comprehensive_reports': [],
            'attendance_reports': [],
            'engagement_reports': [],
            'all_reports': []
        }
        
        report_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis"
        ]
        
        for report_dir in report_dirs:
            if os.path.exists(report_dir):
                for file in os.listdir(report_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(report_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            report_entry = {
                                'filename': file,
                                'file_path': file_path,
                                'file_size': os.path.getsize(file_path),
                                'last_modified': os.path.getmtime(file_path),
                                'data': data
                            }
                            
                            if 'comprehensive_analysis_report' in file:
                                reports['comprehensive_reports'].append(report_entry)
                            elif 'attendance_report' in file:
                                reports['attendance_reports'].append(report_entry)
                            elif 'engagement_report' in file:
                                reports['engagement_reports'].append(report_entry)
                            
                            reports['all_reports'].append(report_entry)
                            
                        except Exception as e:
                            print(f"⚠️ Error reading report {file}: {e}")
        
        return json.dumps(reports, ensure_ascii=False, indent=2)
    
    def collect_raw_data_files_json(self, date_str: str) -> str:
        """Collect all raw data files as JSON string"""
        raw_data = {
            'json_files': [],
            'image_files': [],
            'video_files': [],
            'log_files': [],
            'all_files': []
        }
        
        data_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis",
            f"video_processing/{date_str}",
            f"firebase_backups"
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_info = {
                                'filename': file,
                                'file_path': file_path,
                                'file_size': os.path.getsize(file_path),
                                'last_modified': os.path.getmtime(file_path),
                                'directory': root,
                                'relative_path': os.path.relpath(file_path, data_dir)
                            }
                            
                            if file.endswith('.json'):
                                raw_data['json_files'].append(file_info)
                            elif file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                                raw_data['image_files'].append(file_info)
                            elif file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                                raw_data['video_files'].append(file_info)
                            elif file.endswith(('.log', '.txt')):
                                raw_data['log_files'].append(file_info)
                            
                            raw_data['all_files'].append(file_info)
                            
                        except Exception as e:
                            print(f"⚠️ Error reading file {file}: {e}")
        
        return json.dumps(raw_data, ensure_ascii=False, indent=2)
    
    def calculate_daily_statistics_json(self, daily_data: Dict[str, Any]) -> str:
        """Calculate daily statistics as JSON string"""
        stats = {
            'summary': {
                'date': daily_data['date'],
                'sync_timestamp': daily_data['sync_timestamp']
            },
            'engagement_metrics': {},
            'attendance_metrics': {},
            'technical_metrics': {}
        }
        
        # Parse comprehensive analysis report for statistics
        try:
            comprehensive_data = json.loads(daily_data.get('comprehensive_analysis_report', '{}'))
            if 'overview_statistics' in comprehensive_data:
                stats['engagement_metrics'] = comprehensive_data['overview_statistics']
        except:
            pass
        
        # Parse attendance data for statistics
        try:
            attendance_data = json.loads(daily_data.get('attendance_data', '{}'))
            if 'summary' in attendance_data:
                stats['attendance_metrics'] = attendance_data['summary']
        except:
            pass
        
        # Parse video metadata for statistics
        try:
            video_data = json.loads(daily_data.get('video_metadata', '{}'))
            if 'video_stats' in video_data:
                stats['technical_metrics'] = video_data['video_stats']
        except:
            pass
        
        return json.dumps(stats, ensure_ascii=False, indent=2)
    
    def collect_engagement_data(self, date_str: str) -> Dict[str, Any]:
        """Collect engagement data for the date with complete data"""
        engagement_data = {
            'total_students': 0,
            'engagement_scores': {},
            'activity_breakdown': {},
            'zone_analysis': {},
            'temporal_analysis': {},
            'complete_data': {}
        }
        
        # Look for analysis results
        analysis_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis"
        ]
        
        for analysis_dir in analysis_dirs:
            if os.path.exists(analysis_dir):
                comprehensive_file = os.path.join(analysis_dir, "comprehensive_analysis_report.json")
                if os.path.exists(comprehensive_file):
                    try:
                        with open(comprehensive_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Store complete data
                        engagement_data['complete_data'] = data
                        
                        # Extract specific engagement data
                        if 'overview_statistics' in data:
                            stats = data['overview_statistics']
                            engagement_data['total_students'] = stats.get('total_students', 0)
                            engagement_data['engagement_scores'] = stats.get('engagement_scores', {})
                            engagement_data['activity_breakdown'] = stats.get('activity_breakdown', {})
                            engagement_data['zone_analysis'] = stats.get('zone_analysis', {})
                            engagement_data['temporal_analysis'] = stats.get('temporal_analysis', {})
                        
                        # Extract additional data from comprehensive report
                        if 'students' in data:
                            engagement_data['students_data'] = data['students']
                        if 'activity_analysis' in data:
                            engagement_data['activity_analysis'] = data['activity_analysis']
                        if 'frame_analysis' in data:
                            engagement_data['frame_analysis'] = data['frame_analysis']
                        if 'classroom_zones' in data:
                            engagement_data['classroom_zones'] = data['classroom_zones']
                        
                        break
                    except Exception as e:
                        print(f"⚠️ Error reading engagement data: {e}")
        
        return engagement_data
    
    def collect_attendance_data(self, date_str: str) -> Dict[str, Any]:
        """Collect attendance data for the date"""
        attendance_data = {
            'total_persons': 0,
            'total_appearances': 0,
            'attendance_records': {},
            'face_matching_stats': {}
        }
        
        # Load from enhanced face matcher
        try:
            from enhanced_face_matcher import EnhancedFaceMatcher
            face_matcher = EnhancedFaceMatcher()
            face_matcher.load_face_database()
            
            attendance_summary = face_matcher.get_attendance_summary()
            attendance_data['total_persons'] = len(attendance_summary)
            attendance_data['total_appearances'] = sum([p.get('total_appearances', 0) for p in attendance_summary.values()])
            attendance_data['attendance_records'] = attendance_summary
            
            # Add face matching statistics
            if hasattr(face_matcher, 'face_database'):
                face_db = face_matcher.face_database
                attendance_data['face_matching_stats'] = {
                    'total_encodings': sum(len(p.get('encodings', [])) for p in face_db.values()),
                    'total_features': sum(len(p.get('image_features', [])) for p in face_db.values()),
                    'total_images': sum(len(p.get('images', [])) for p in face_db.values())
                }
                
        except Exception as e:
            print(f"⚠️ Error collecting attendance data: {e}")
        
        return attendance_data
    
    def collect_face_data(self, date_str: str) -> Dict[str, Any]:
        """Collect face recognition data for the date with complete data"""
        face_data = {
            'face_database': {},
            'matching_results': {},
            'quality_metrics': {},
            'complete_face_database': {}
        }
        
        try:
            from enhanced_face_matcher import EnhancedFaceMatcher
            face_matcher = EnhancedFaceMatcher()
            face_matcher.load_face_database()
            
            if hasattr(face_matcher, 'face_database'):
                # Store complete face database
                face_data['complete_face_database'] = face_matcher.face_database
                
                # Convert face database to JSON-serializable format
                face_db = face_matcher.face_database
                for person_id, person_data in face_db.items():
                    face_data['face_database'][person_id] = {
                        'person_id': person_data.get('person_id'),
                        'total_appearances': person_data.get('total_appearances', 0),
                        'first_seen': person_data.get('first_seen'),
                        'last_seen': person_data.get('last_seen'),
                        'encodings_count': len(person_data.get('encodings', [])),
                        'features_count': len(person_data.get('image_features', [])),
                        'images_count': len(person_data.get('images', [])),
                        'videos': person_data.get('videos', []),
                        'metadata': person_data.get('metadata', {}),
                        'encodings': person_data.get('encodings', []),
                        'image_features': person_data.get('image_features', []),
                        'images': person_data.get('images', [])
                    }
                
        except Exception as e:
            print(f"⚠️ Error collecting face data: {e}")
        
        return face_data
    
    def collect_lecture_classifications(self, date_str: str) -> Dict[str, Any]:
        """Collect lecture classification data for the date"""
        classifications = {
            'classifications': {},
            'classification_method': 'rule_based',
            'confidence_scores': {}
        }
        
        # Look for classification results
        analysis_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis"
        ]
        
        for analysis_dir in analysis_dirs:
            if os.path.exists(analysis_dir):
                comprehensive_file = os.path.join(analysis_dir, "comprehensive_analysis_report.json")
                if os.path.exists(comprehensive_file):
                    try:
                        with open(comprehensive_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'lecture_classification' in data:
                            classifications['classifications'] = data['lecture_classification']
                            classifications['classification_method'] = data.get('classification_method', 'rule_based')
                            classifications['confidence_scores'] = data.get('confidence_scores', {})
                        
                        break
                    except Exception as e:
                        print(f"⚠️ Error reading classification data: {e}")
        
        return classifications
    
    def collect_video_metadata(self, date_str: str) -> Dict[str, Any]:
        """Collect video metadata for the date with complete data"""
        video_metadata = {
            'processed_videos': [],
            'video_stats': {},
            'all_video_files': []
        }
        
        # Look for processed videos
        processed_dirs = [
            f"video_processing/{date_str}/processed",
            f"analysis_history/{date_str}",
            f"video_processing/{date_str}/input"
        ]
        
        for processed_dir in processed_dirs:
            if os.path.exists(processed_dir):
                for video_file in os.listdir(processed_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        video_path = os.path.join(processed_dir, video_file)
                        try:
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                duration = frame_count / fps if fps > 0 else 0
                                
                                video_info = {
                                    'filename': video_file,
                                    'file_path': video_path,
                                    'fps': fps,
                                    'frame_count': frame_count,
                                    'width': width,
                                    'height': height,
                                    'duration': duration,
                                    'file_size': os.path.getsize(video_path),
                                    'last_modified': os.path.getmtime(video_path),
                                    'directory': processed_dir
                                }
                                
                                video_metadata['processed_videos'].append(video_info)
                                video_metadata['all_video_files'].append(video_info)
                                cap.release()
                        except Exception as e:
                            print(f"⚠️ Error reading video metadata for {video_file}: {e}")
        
        # Calculate video stats
        if video_metadata['processed_videos']:
            videos = video_metadata['processed_videos']
            video_metadata['video_stats'] = {
                'total_videos': len(videos),
                'total_duration': sum(v['duration'] for v in videos),
                'total_frames': sum(v['frame_count'] for v in videos),
                'average_fps': sum(v['fps'] for v in videos) / len(videos),
                'total_size': sum(v['file_size'] for v in videos),
                'average_duration': sum(v['duration'] for v in videos) / len(videos),
                'resolution_distribution': {
                    'widths': list(set(v['width'] for v in videos)),
                    'heights': list(set(v['height'] for v in videos))
                }
            }
        
        return video_metadata
    
    def collect_analysis_reports(self, date_str: str) -> Dict[str, Any]:
        """Collect analysis reports for the date with complete data"""
        reports = {
            'comprehensive_reports': [],
            'attendance_reports': [],
            'engagement_reports': [],
            'all_reports': []
        }
        
        # Look for various report types
        report_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis"
        ]
        
        for report_dir in report_dirs:
            if os.path.exists(report_dir):
                for file in os.listdir(report_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(report_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Create report entry with complete data
                            report_entry = {
                                'filename': file,
                                'file_path': file_path,
                                'file_size': os.path.getsize(file_path),
                                'last_modified': os.path.getmtime(file_path),
                                'data': data  # Complete data content
                            }
                            
                            # Categorize reports
                            if 'comprehensive_analysis_report' in file:
                                reports['comprehensive_reports'].append(report_entry)
                            elif 'attendance_report' in file:
                                reports['attendance_reports'].append(report_entry)
                            elif 'engagement_report' in file:
                                reports['engagement_reports'].append(report_entry)
                            
                            # Add to all reports
                            reports['all_reports'].append(report_entry)
                            
                        except Exception as e:
                            print(f"⚠️ Error reading report {file}: {e}")
        
        return reports
    
    def collect_raw_data_files(self, date_str: str) -> Dict[str, Any]:
        """Collect all raw data files for the date"""
        raw_data = {
            'json_files': [],
            'image_files': [],
            'video_files': [],
            'log_files': [],
            'all_files': []
        }
        
        # Look for data in various directories
        data_dirs = [
            f"analysis_results",
            f"analysis_history/{date_str}",
            f"realtime_analysis",
            f"video_processing/{date_str}",
            f"firebase_backups"
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_info = {
                                'filename': file,
                                'file_path': file_path,
                                'file_size': os.path.getsize(file_path),
                                'last_modified': os.path.getmtime(file_path),
                                'directory': root,
                                'relative_path': os.path.relpath(file_path, data_dir)
                            }
                            
                            # Categorize files
                            if file.endswith('.json'):
                                raw_data['json_files'].append(file_info)
                            elif file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                                raw_data['image_files'].append(file_info)
                            elif file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                                raw_data['video_files'].append(file_info)
                            elif file.endswith(('.log', '.txt')):
                                raw_data['log_files'].append(file_info)
                            
                            raw_data['all_files'].append(file_info)
                            
                        except Exception as e:
                            print(f"⚠️ Error reading file {file}: {e}")
        
        return raw_data
    
    def calculate_daily_statistics(self, daily_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the day"""
        stats = {
            'summary': {
                'date': daily_data['date'],
                'total_students': daily_data['engagement_data'].get('total_students', 0),
                'total_videos': len(daily_data['video_metadata'].get('processed_videos', [])),
                'total_appearances': daily_data['attendance_data'].get('total_appearances', 0),
                'face_matches': len(daily_data['face_data'].get('face_database', {}))
            },
            'engagement_metrics': {
                'average_engagement': 0,
                'most_engaged_zone': 'unknown',
                'peak_activity_time': 'unknown'
            },
            'attendance_metrics': {
                'attendance_rate': 0,
                'most_active_student': 'unknown',
                'average_appearances': 0
            },
            'technical_metrics': {
                'face_recognition_accuracy': 0,
                'classification_confidence': 0,
                'data_quality_score': 0
            }
        }
        
        # Calculate engagement metrics
        engagement_data = daily_data['engagement_data']
        if engagement_data.get('engagement_scores'):
            scores = list(engagement_data['engagement_scores'].values())
            if scores:
                stats['engagement_metrics']['average_engagement'] = sum(scores) / len(scores)
        
        # Calculate attendance metrics
        attendance_data = daily_data['attendance_data']
        if attendance_data.get('attendance_records'):
            records = attendance_data['attendance_records']
            if records:
                appearances = [r.get('total_appearances', 0) for r in records.values()]
                stats['attendance_metrics']['average_appearances'] = sum(appearances) / len(appearances)
                
                # Find most active student
                most_active = max(records.items(), key=lambda x: x[1].get('total_appearances', 0))
                stats['attendance_metrics']['most_active_student'] = most_active[0]
        
        return stats
    
    def save_local_backup(self, date_str: str) -> bool:
        """Save data locally as backup when Firebase is not available"""
        try:
            daily_data = self.collect_daily_data(date_str)
            
            # Create backup directory
            backup_dir = Path("firebase_backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Save to local file
            backup_file = backup_dir / f"daily_data_{date_str}.json"
            with open(backup_file, 'w') as f:
                json.dump(daily_data, f, indent=2)
            
            print(f"💾 Local backup saved: {backup_file}")
            return True
            
        except Exception as e:
            print(f"❌ Local backup failed: {e}")
            return False
    
    def sync_all_historical_data(self) -> bool:
        """Sync all historical data to Firebase"""
        if not self.initialized:
            print("⚠️ Firebase not initialized. Cannot sync historical data.")
            return False
        
        try:
            print("🔄 Syncing all historical data to Firebase...")
            
            # Find all dates with data
            dates_with_data = set()
            
            # Check analysis_history
            if os.path.exists("analysis_history"):
                for date_dir in os.listdir("analysis_history"):
                    if os.path.isdir(os.path.join("analysis_history", date_dir)):
                        dates_with_data.add(date_dir)
            
            # Check video_processing
            if os.path.exists("video_processing"):
                for date_dir in os.listdir("video_processing"):
                    if os.path.isdir(os.path.join("video_processing", date_dir)):
                        dates_with_data.add(date_dir)
            
            # Sync each date
            success_count = 0
            for date_str in sorted(dates_with_data):
                if self.sync_daily_data(date_str):
                    success_count += 1
            
            print(f"✅ Synced {success_count}/{len(dates_with_data)} dates successfully")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Historical data sync failed: {e}")
            return False

# Firebase configuration
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyCyqy5gpfZuCAhsd5xLbBxtw-Vbhudcsqs",
    "authDomain": "code4-509a0.firebaseapp.com",
    "databaseURL": "https://code4-509a0-default-rtdb.firebaseio.com",
    "projectId": "code4-509a0",
    "storageBucket": "code4-509a0.firebasestorage.app",
    "messagingSenderId": "358047249983",
    "appId": "1:358047249983:web:98f56e1b4d531ec4f3ea8a",
    "measurementId": "G-56XMYHQ9N4"
}

# Test the Firebase sync
if __name__ == "__main__":
    print("🧪 Testing Firebase Sync...")
    
    # Initialize Firebase sync
    firebase_sync = FirebaseSync(FIREBASE_CONFIG)
    
    # Test sync for today
    if firebase_sync.sync_daily_data():
        print("✅ Daily sync test successful")
    else:
        print("❌ Daily sync test failed")
    
    # Test sync all historical data
    if firebase_sync.sync_all_historical_data():
        print("✅ Historical sync test successful")
    else:
        print("❌ Historical sync test failed")
