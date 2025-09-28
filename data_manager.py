"""
Data Manager for Classroom Analyzer
Handles data persistence, analysis history, and cross-video tracking
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil
from pathlib import Path
import hashlib

class DataManager:
    def __init__(self, database_path="classroom_analyzer.db", data_dir="analysis_history"):
        """
        Initialize the data manager
        
        Args:
            database_path (str): Path to SQLite database
            data_dir (str): Directory for storing analysis data
        """
        self.database_path = database_path
        self.data_dir = data_dir
        self.ensure_data_directory()
        self.init_database()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"📁 Data directory: {self.data_dir}")
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Videos table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_path TEXT UNIQUE NOT NULL,
                        video_name TEXT NOT NULL,
                        video_hash TEXT NOT NULL,
                        duration REAL,
                        fps REAL,
                        resolution TEXT,
                        file_size INTEGER,
                        analysis_date TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Analysis sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id INTEGER NOT NULL,
                        session_name TEXT,
                        output_directory TEXT NOT NULL,
                        headless_mode BOOLEAN DEFAULT FALSE,
                        total_frames INTEGER,
                        analysis_duration REAL,
                        models_used TEXT,
                        status TEXT DEFAULT 'completed',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (video_id) REFERENCES videos (id)
                    )
                ''')
                
                # Students table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT UNIQUE NOT NULL,
                        first_seen_session INTEGER,
                        last_seen_session INTEGER,
                        total_appearances INTEGER DEFAULT 0,
                        position_zone TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (first_seen_session) REFERENCES analysis_sessions (id),
                        FOREIGN KEY (last_seen_session) REFERENCES analysis_sessions (id)
                    )
                ''')
                
                # Faces table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT NOT NULL,
                        session_id INTEGER NOT NULL,
                        person_id TEXT,
                        best_image_path TEXT,
                        best_confidence REAL,
                        quality_score REAL,
                        base64_image TEXT,
                        appearances INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                    )
                ''')
                
                # Attendance records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        session_id INTEGER NOT NULL,
                        attendance_type TEXT DEFAULT 'present',
                        timestamp TEXT,
                        confidence REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id),
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                    )
                ''')
                
                # Lecture classifications table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lecture_classifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id INTEGER NOT NULL,
                        session_id INTEGER,
                        lecture_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        classification_method TEXT,
                        reasoning TEXT,
                        all_scores TEXT,
                        classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (video_id) REFERENCES videos (id),
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                    )
                ''')
                
                # Face matching records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_matching_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        face_id TEXT NOT NULL,
                        matched_person_id TEXT,
                        similarity_score REAL,
                        match_type TEXT DEFAULT 'new',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                    )
                ''')
                
                conn.commit()
                print("✅ Database initialized successfully")
                
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"❌ Error calculating file hash: {e}")
            return ""
    
    def register_video(self, video_path: str, metadata: Dict = None) -> int:
        """
        Register a video in the database
        
        Args:
            video_path (str): Path to video file
            metadata (dict): Video metadata
            
        Returns:
            int: Video ID in database
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            video_hash = self.calculate_file_hash(video_path)
            video_name = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            
            metadata = metadata or {}
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Check if video already exists
                cursor.execute('SELECT id FROM videos WHERE video_hash = ?', (video_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    print(f"📹 Video already registered: {video_name} (ID: {existing[0]})")
                    return existing[0]
                
                # Insert new video
                cursor.execute('''
                    INSERT INTO videos (video_path, video_name, video_hash, duration, fps, 
                                      resolution, file_size, analysis_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_path, video_name, video_hash,
                    metadata.get('duration'),
                    metadata.get('fps'),
                    metadata.get('resolution'),
                    file_size,
                    datetime.now().isoformat()
                ))
                
                video_id = cursor.lastrowid
                conn.commit()
                
                print(f"✅ Video registered: {video_name} (ID: {video_id})")
                return video_id
                
        except Exception as e:
            print(f"❌ Error registering video: {e}")
            return None
    
    def create_analysis_session(self, video_id: int, output_dir: str, 
                               session_name: str = None, **kwargs) -> int:
        """
        Create a new analysis session
        
        Args:
            video_id (int): Video ID
            output_dir (str): Output directory for analysis
            session_name (str): Optional session name
            **kwargs: Additional session parameters
            
        Returns:
            int: Session ID
        """
        try:
            if session_name is None:
                session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_sessions (video_id, session_name, output_directory, 
                                                 headless_mode, models_used, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    video_id, session_name, output_dir,
                    kwargs.get('headless_mode', False),
                    json.dumps(kwargs.get('models_used', [])),
                    kwargs.get('status', 'started')
                ))
                
                session_id = cursor.lastrowid
                conn.commit()
                
                print(f"✅ Analysis session created: {session_name} (ID: {session_id})")
                return session_id
                
        except Exception as e:
            print(f"❌ Error creating analysis session: {e}")
            return None
    
    def update_analysis_session(self, session_id: int, **kwargs):
        """Update analysis session with results"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Build update query dynamically
                update_fields = []
                values = []
                
                for key, value in kwargs.items():
                    if key in ['total_frames', 'analysis_duration', 'status']:
                        update_fields.append(f"{key} = ?")
                        values.append(value)
                
                if update_fields:
                    values.append(session_id)
                    query = f"UPDATE analysis_sessions SET {', '.join(update_fields)} WHERE id = ?"
                    cursor.execute(query, values)
                    conn.commit()
                    
                    print(f"✅ Analysis session updated: {session_id}")
                    
        except Exception as e:
            print(f"❌ Error updating analysis session: {e}")
    
    def save_analysis_results(self, session_id: int, analysis_data: Dict):
        """
        Save comprehensive analysis results
        
        Args:
            session_id (int): Analysis session ID
            analysis_data (dict): Analysis results data
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Save students
                students_data = analysis_data.get('students', {})
                for student_id, student_info in students_data.items():
                    self.save_student_data(cursor, student_id, student_info, session_id)
                
                # Save faces
                faces_data = analysis_data.get('faces', {})
                for face_id, face_info in faces_data.items():
                    self.save_face_data(cursor, face_id, face_info, session_id)
                
                conn.commit()
                print(f"✅ Analysis results saved for session {session_id}")
                
        except Exception as e:
            print(f"❌ Error saving analysis results: {e}")
    
    def save_student_data(self, cursor, student_id: str, student_info: Dict, session_id: int):
        """Save student data to database"""
        try:
            tracking_info = student_info.get('tracking_summary', {})
            
            # Check if student exists
            cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing student
                cursor.execute('''
                    UPDATE students SET last_seen_session = ?, total_appearances = total_appearances + 1,
                                      position_zone = ?
                    WHERE student_id = ?
                ''', (session_id, tracking_info.get('position_zone'), student_id))
            else:
                # Insert new student
                cursor.execute('''
                    INSERT INTO students (student_id, first_seen_session, last_seen_session,
                                        total_appearances, position_zone)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    student_id, session_id, session_id,
                    1, tracking_info.get('position_zone')
                ))
            
            # Create attendance record
            cursor.execute('''
                INSERT INTO attendance_records (student_id, session_id, attendance_type, timestamp)
                VALUES (?, ?, 'present', ?)
            ''', (student_id, session_id, datetime.now().isoformat()))
            
        except Exception as e:
            print(f"❌ Error saving student data for {student_id}: {e}")
    
    def save_face_data(self, cursor, face_id: str, face_info: Dict, session_id: int):
        """Save face data to database"""
        try:
            best_image_info = face_info.get('best_image_info', {})
            
            cursor.execute('''
                INSERT INTO faces (face_id, session_id, best_confidence, quality_score,
                                 base64_image, appearances)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                face_id, session_id,
                best_image_info.get('confidence', 0),
                best_image_info.get('quality_score', 0),
                best_image_info.get('base64_image', ''),
                face_info.get('detection_summary', {}).get('total_appearances', 1)
            ))
            
        except Exception as e:
            print(f"❌ Error saving face data for {face_id}: {e}")
    
    def save_lecture_classification(self, video_id: int, session_id: int, classification: Dict):
        """Save lecture classification results"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO lecture_classifications (video_id, session_id, lecture_type,
                                                       confidence, classification_method,
                                                       reasoning, all_scores)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_id, session_id, classification['type'],
                    classification['confidence'], classification['method'],
                    classification.get('reasoning', ''),
                    json.dumps(classification.get('all_scores', {}))
                ))
                
                conn.commit()
                print(f"✅ Lecture classification saved: {classification['type']}")
                
        except Exception as e:
            print(f"❌ Error saving lecture classification: {e}")
    
    def save_face_matching_results(self, session_id: int, matching_results: Dict):
        """Save face matching results"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Save matches
                for match in matching_results.get('matches', []):
                    cursor.execute('''
                        INSERT INTO face_matching_records (session_id, face_id, matched_person_id,
                                                         similarity_score, match_type)
                        VALUES (?, ?, ?, ?, 'matched')
                    ''', (session_id, match['face_id'], match['matched_person_id'], 
                         match['similarity']))
                
                # Save new faces
                for person in matching_results.get('new_persons', []):
                    cursor.execute('''
                        INSERT INTO face_matching_records (session_id, face_id, matched_person_id,
                                                         similarity_score, match_type)
                        VALUES (?, ?, ?, ?, 'new')
                    ''', (session_id, person['face_id'], person['person_id'], 1.0))
                
                conn.commit()
                print(f"✅ Face matching results saved for session {session_id}")
                
        except Exception as e:
            print(f"❌ Error saving face matching results: {e}")
    
    def get_video_history(self, video_id: int = None) -> List[Dict]:
        """Get video analysis history"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                if video_id:
                    cursor.execute('''
                        SELECT v.*, COUNT(s.id) as session_count
                        FROM videos v
                        LEFT JOIN analysis_sessions s ON v.id = s.video_id
                        WHERE v.id = ?
                        GROUP BY v.id
                    ''', (video_id,))
                else:
                    cursor.execute('''
                        SELECT v.*, COUNT(s.id) as session_count
                        FROM videos v
                        LEFT JOIN analysis_sessions s ON v.id = s.video_id
                        GROUP BY v.id
                        ORDER BY v.created_at DESC
                    ''')
                
                videos = []
                for row in cursor.fetchall():
                    videos.append({
                        'id': row[0],
                        'video_path': row[1],
                        'video_name': row[2],
                        'video_hash': row[3],
                        'duration': row[4],
                        'fps': row[5],
                        'resolution': row[6],
                        'file_size': row[7],
                        'analysis_date': row[8],
                        'created_at': row[9],
                        'session_count': row[10]
                    })
                
                return videos
                
        except Exception as e:
            print(f"❌ Error getting video history: {e}")
            return []
    
    def get_analysis_sessions(self, video_id: int = None) -> List[Dict]:
        """Get analysis sessions"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                if video_id:
                    cursor.execute('''
                        SELECT s.*, v.video_name
                        FROM analysis_sessions s
                        JOIN videos v ON s.video_id = v.id
                        WHERE s.video_id = ?
                        ORDER BY s.created_at DESC
                    ''', (video_id,))
                else:
                    cursor.execute('''
                        SELECT s.*, v.video_name
                        FROM analysis_sessions s
                        JOIN videos v ON s.video_id = v.id
                        ORDER BY s.created_at DESC
                    ''')
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'id': row[0],
                        'video_id': row[1],
                        'session_name': row[2],
                        'output_directory': row[3],
                        'headless_mode': row[4],
                        'total_frames': row[5],
                        'analysis_duration': row[6],
                        'models_used': json.loads(row[7]) if row[7] else [],
                        'status': row[8],
                        'created_at': row[9],
                        'video_name': row[10]
                    })
                
                return sessions
                
        except Exception as e:
            print(f"❌ Error getting analysis sessions: {e}")
            return []
    
    def get_attendance_summary(self) -> Dict:
        """Get comprehensive attendance summary"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute('''
                    SELECT 
                        COUNT(DISTINCT s.student_id) as total_students,
                        COUNT(DISTINCT ar.session_id) as total_sessions,
                        COUNT(ar.id) as total_attendance_records
                    FROM students s
                    LEFT JOIN attendance_records ar ON s.student_id = ar.student_id
                ''')
                
                stats = cursor.fetchone()
                
                # Get student attendance details
                cursor.execute('''
                    SELECT 
                        s.student_id,
                        s.total_appearances,
                        COUNT(ar.id) as sessions_attended,
                        s.position_zone,
                        s.first_seen_session,
                        s.last_seen_session
                    FROM students s
                    LEFT JOIN attendance_records ar ON s.student_id = ar.student_id
                    GROUP BY s.student_id
                    ORDER BY s.total_appearances DESC
                ''')
                
                student_details = []
                for row in cursor.fetchall():
                    student_details.append({
                        'student_id': row[0],
                        'total_appearances': row[1],
                        'sessions_attended': row[2],
                        'position_zone': row[3],
                        'first_seen_session': row[4],
                        'last_seen_session': row[5]
                    })
                
                return {
                    'total_students': stats[0],
                    'total_sessions': stats[1],
                    'total_attendance_records': stats[2],
                    'student_details': student_details,
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {}
    
    def get_lecture_classifications(self, video_id: int = None) -> List[Dict]:
        """Get lecture classification history"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                if video_id:
                    cursor.execute('''
                        SELECT lc.*, v.video_name, s.session_name
                        FROM lecture_classifications lc
                        JOIN videos v ON lc.video_id = v.id
                        LEFT JOIN analysis_sessions s ON lc.session_id = s.id
                        WHERE lc.video_id = ?
                        ORDER BY lc.classified_at DESC
                    ''', (video_id,))
                else:
                    cursor.execute('''
                        SELECT lc.*, v.video_name, s.session_name
                        FROM lecture_classifications lc
                        JOIN videos v ON lc.video_id = v.id
                        LEFT JOIN analysis_sessions s ON lc.session_id = s.id
                        ORDER BY lc.classified_at DESC
                    ''')
                
                classifications = []
                for row in cursor.fetchall():
                    classifications.append({
                        'id': row[0],
                        'video_id': row[1],
                        'session_id': row[2],
                        'lecture_type': row[3],
                        'confidence': row[4],
                        'classification_method': row[5],
                        'reasoning': row[6],
                        'all_scores': json.loads(row[7]) if row[7] else {},
                        'classified_at': row[8],
                        'video_name': row[9],
                        'session_name': row[10]
                    })
                
                return classifications
                
        except Exception as e:
            print(f"❌ Error getting lecture classifications: {e}")
            return []
    
    def backup_data(self, backup_path: str = None):
        """Create backup of database and analysis data"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"classroom_analyzer_backup_{timestamp}"
            
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup database
            shutil.copy2(self.database_path, backup_path)
            
            # Backup analysis data directory
            if os.path.exists(self.data_dir):
                shutil.copytree(self.data_dir, os.path.join(backup_path, "analysis_data"))
            
            print(f"✅ Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return None
    
    def restore_data(self, backup_path: str):
        """Restore data from backup"""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Restore database
            db_backup = os.path.join(backup_path, os.path.basename(self.database_path))
            if os.path.exists(db_backup):
                shutil.copy2(db_backup, self.database_path)
            
            # Restore analysis data
            analysis_backup = os.path.join(backup_path, "analysis_data")
            if os.path.exists(analysis_backup):
                if os.path.exists(self.data_dir):
                    shutil.rmtree(self.data_dir)
                shutil.copytree(analysis_backup, self.data_dir)
            
            print(f"✅ Data restored from: {backup_path}")
            
        except Exception as e:
            print(f"❌ Error restoring data: {e}")
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old analysis data"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Find old sessions
                cursor.execute('''
                    SELECT id, output_directory FROM analysis_sessions 
                    WHERE created_at < datetime(?, 'unixepoch')
                ''', (cutoff_date,))
                
                old_sessions = cursor.fetchall()
                
                for session_id, output_dir in old_sessions:
                    # Remove analysis files
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                        print(f"🗑️ Removed old analysis: {output_dir}")
                
                # Clean up database records
                cursor.execute('''
                    DELETE FROM analysis_sessions 
                    WHERE created_at < datetime(?, 'unixepoch')
                ''', (cutoff_date,))
                
                conn.commit()
                print(f"✅ Cleaned up data older than {days_old} days")
                
        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Video statistics
                cursor.execute('SELECT COUNT(*) FROM videos')
                total_videos = cursor.fetchone()[0]
                
                # Session statistics
                cursor.execute('SELECT COUNT(*) FROM analysis_sessions')
                total_sessions = cursor.fetchone()[0]
                
                # Student statistics
                cursor.execute('SELECT COUNT(*) FROM students')
                total_students = cursor.fetchone()[0]
                
                # Face statistics
                cursor.execute('SELECT COUNT(*) FROM faces')
                total_faces = cursor.fetchone()[0]
                
                # Classification statistics
                cursor.execute('''
                    SELECT lecture_type, COUNT(*) 
                    FROM lecture_classifications 
                    GROUP BY lecture_type
                ''')
                classifications = dict(cursor.fetchall())
                
                return {
                    'total_videos': total_videos,
                    'total_sessions': total_sessions,
                    'total_students': total_students,
                    'total_faces': total_faces,
                    'lecture_classifications': classifications,
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

def main():
    """Test the data manager"""
    dm = DataManager()
    
    print("📊 Data Manager Test")
    print("=" * 40)
    
    # Get statistics
    stats = dm.get_statistics()
    print(f"📈 Statistics: {stats}")
    
    # Get video history
    videos = dm.get_video_history()
    print(f"📹 Videos in database: {len(videos)}")
    
    # Get analysis sessions
    sessions = dm.get_analysis_sessions()
    print(f"🔍 Analysis sessions: {len(sessions)}")

if __name__ == "__main__":
    main()

