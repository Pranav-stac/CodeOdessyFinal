"""
Automated Video Processing System
Processes videos from organized folder structure with date-based organization
"""

import os
import cv2
import json
import shutil
from datetime import datetime, date
from pathlib import Path
import threading
import time
from collections import defaultdict

class AutomatedVideoProcessor:
    def __init__(self, base_folder="video_processing", analysis_folder="analysis_history"):
        """
        Initialize automated video processor
        
        Args:
            base_folder (str): Base folder for video organization
            analysis_folder (str): Folder for storing analysis results
        """
        self.base_folder = Path(base_folder)
        self.analysis_folder = Path(analysis_folder)
        self.processing_log = []
        self.current_processing = None
        self.is_processing = False
        
        # Create folder structure
        self.setup_folder_structure()
        
    def setup_folder_structure(self):
        """Create the required folder structure"""
        try:
            # Create base folders
            self.base_folder.mkdir(exist_ok=True)
            self.analysis_folder.mkdir(exist_ok=True)
            
            # Create today's date folder
            today_folder = self.base_folder / date.today().strftime("%Y-%m-%d")
            today_folder.mkdir(exist_ok=True)
            
            # Create subfolders
            (today_folder / "input").mkdir(exist_ok=True)
            (today_folder / "processed").mkdir(exist_ok=True)
            (today_folder / "reports").mkdir(exist_ok=True)
            
            print(f"✅ Folder structure created: {self.base_folder}")
            print(f"📁 Today's folder: {today_folder}")
            
        except Exception as e:
            print(f"❌ Error creating folder structure: {e}")
    
    def get_today_folder(self):
        """Get today's processing folder"""
        return self.base_folder / date.today().strftime("%Y-%m-%d")
    
    def get_available_dates(self):
        """Get list of available date folders"""
        dates = []
        if self.base_folder.exists():
            for item in self.base_folder.iterdir():
                if item.is_dir() and len(item.name) == 10:  # YYYY-MM-DD format
                    try:
                        datetime.strptime(item.name, "%Y-%m-%d")
                        dates.append(item.name)
                    except ValueError:
                        continue
        return sorted(dates, reverse=True)  # Most recent first
    
    def get_videos_for_date(self, date_str):
        """Get list of videos for a specific date"""
        date_folder = self.base_folder / date_str
        input_folder = date_folder / "input"
        
        if not input_folder.exists():
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        videos = []
        
        for file in input_folder.iterdir():
            if file.suffix.lower() in video_extensions:
                videos.append({
                    'name': file.name,
                    'path': str(file),
                    'size': file.stat().st_size,
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                })
        
        return sorted(videos, key=lambda x: x['modified'], reverse=True)
    
    def get_processed_videos(self, date_str):
        """Get list of already processed videos for a date"""
        date_folder = self.base_folder / date_str
        reports_folder = date_folder / "reports"
        
        if not reports_folder.exists():
            return []
        
        processed = []
        for report_file in reports_folder.glob("*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    processed.append({
                        'video_name': report.get('video_name', 'Unknown'),
                        'report_file': str(report_file),
                        'processed_at': report.get('processed_at', 'Unknown'),
                        'status': report.get('status', 'Unknown')
                    })
            except Exception as e:
                print(f"⚠️ Error reading report {report_file}: {e}")
        
        return processed
    
    def process_video(self, video_path, output_dir, video_name, show_footage=False, realtime_display=True):
        """Process a single video"""
        try:
            from realtime_classroom_analyzer import RealtimeClassroomAnalyzer
            
            # Initialize analyzer
            analyzer = RealtimeClassroomAnalyzer(
                video_path=video_path,
                output_dir=output_dir
            )
            
            # Process video
            print(f"🎬 Processing video: {video_name}")
            success = analyzer.analyze_video_realtime(display=show_footage, save_frames=False)
            
            if success:
                # Perform lecture classification
                lecture_classification = self.classify_lecture_type(video_path)
                
                # Generate comprehensive report
                report = self.generate_processing_report(analyzer, video_name, video_path, lecture_classification)
                
                # Save report
                report_file = Path(output_dir) / f"{Path(video_name).stem}_report.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                print(f"✅ Video processed successfully: {video_name}")
                return True, str(report_file)
            else:
                print(f"❌ Video processing failed: {video_name}")
                return False, None
                
        except Exception as e:
            print(f"❌ Error processing video {video_name}: {e}")
            return False, None
    
    def classify_lecture_type(self, video_path):
        """Classify lecture type using vision LLM"""
        try:
            from lightweight_vision_classifier import LightweightVisionClassifier
            
            print(f"🎓 Classifying lecture type for: {Path(video_path).name}")
            classifier = LightweightVisionClassifier()
            result = classifier.classify_video_frame(video_path, frame_time=0.5)
            
            print(f"✅ Classification: {result['lecture_type']} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            print(f"⚠️ Lecture classification failed: {e}")
            return {
                'lecture_type': 'unknown',
                'confidence': 0.0,
                'method': 'failed',
                'error': str(e)
            }
    
    def generate_processing_report(self, analyzer, video_name, video_path, lecture_classification=None):
        """Generate comprehensive processing report"""
        try:
            # Load analysis data
            comprehensive_file = Path(analyzer.output_dir) / "comprehensive_analysis_report.json"
            
            if comprehensive_file.exists():
                with open(comprehensive_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                analysis_data = {}
            
            # Add lecture classification to analysis data
            if lecture_classification:
                analysis_data['lecture_classification'] = lecture_classification
                
                # Update comprehensive report with lecture classification
                with open(comprehensive_file, 'w') as f:
                    json.dump(analysis_data, f, indent=2)
            
            # Create report
            report = {
                'video_name': video_name,
                'video_path': str(video_path),
                'processed_at': datetime.now().isoformat(),
                'status': 'completed',
                'output_directory': str(analyzer.output_dir),
                'analysis_summary': analysis_data.get('overview_statistics', {}),
                'face_count': len(analysis_data.get('faces', {})),
                'student_count': len(analysis_data.get('students', {})),
                'processing_time': getattr(analyzer, 'processing_time', 0),
                'file_references': analysis_data.get('file_references', {}),
                'lecture_classification': lecture_classification
            }
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return {
                'video_name': video_name,
                'video_path': str(video_path),
                'processed_at': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'lecture_classification': lecture_classification
            }
    
    def process_all_videos(self, date_str=None, progress_callback=None, show_footage=False, realtime_display=True):
        """Process all videos for a specific date or today"""
        if date_str is None:
            date_str = date.today().strftime("%Y-%m-%d")
        
        date_folder = self.base_folder / date_str
        input_folder = date_folder / "input"
        reports_folder = date_folder / "reports"
        processed_folder = date_folder / "processed"
        
        if not input_folder.exists():
            print(f"❌ No input folder found for date: {date_str}")
            return False
        
        # Get videos to process
        videos = self.get_videos_for_date(date_str)
        processed_videos = self.get_processed_videos(date_str)
        processed_names = {p['video_name'] for p in processed_videos}
        
        # Filter out already processed videos
        videos_to_process = [v for v in videos if v['name'] not in processed_names]
        
        if not videos_to_process:
            print(f"✅ All videos already processed for {date_str}")
            return True
        
        print(f"🎬 Found {len(videos_to_process)} videos to process for {date_str}")
        
        self.is_processing = True
        self.current_processing = {
            'date': date_str,
            'total_videos': len(videos_to_process),
            'processed': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
        
        try:
            for i, video in enumerate(videos_to_process):
                if not self.is_processing:  # Check if processing was stopped
                    break
                
                video_name = video['name']
                video_path = video['path']
                
                # Calculate progress (0-100% for this date's videos)
                progress = (i / len(videos_to_process)) * 100
                
                if progress_callback:
                    progress_callback(progress, f"📹 Processing {i+1}/{len(videos_to_process)}: {video_name}")
                print(f"\n📹 Processing {i+1}/{len(videos_to_process)}: {video_name}")
                
                # Create output directory
                output_dir = self.analysis_folder / date_str / Path(video_name).stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process video
                success, report_file = self.process_video(video_path, output_dir, video_name, show_footage, realtime_display)
                
                if success:
                    # Move video to processed folder
                    processed_video_path = processed_folder / video_name
                    shutil.move(video_path, processed_video_path)
                    
                    # Move report to reports folder
                    if report_file:
                        report_dest = reports_folder / Path(report_file).name
                        shutil.move(report_file, report_dest)
                    
                    self.current_processing['processed'] += 1
                    print(f"✅ Completed: {video_name}")
                else:
                    self.current_processing['failed'] += 1
                    print(f"❌ Failed: {video_name}")
                
                # Update progress
                if progress_callback:
                    progress = ((i + 1) / len(videos_to_process)) * 100
                    progress_callback(progress, f"Processed {i+1}/{len(videos_to_process)} videos")
                
                # Add to log
                self.processing_log.append({
                    'video_name': video_name,
                    'date': date_str,
                    'status': 'completed' if success else 'failed',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Final summary
            total_time = datetime.now() - self.current_processing['start_time']
            print(f"\n🎉 Processing complete for {date_str}")
            print(f"📊 Processed: {self.current_processing['processed']}")
            print(f"❌ Failed: {self.current_processing['failed']}")
            print(f"⏱️ Total time: {total_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during batch processing: {e}")
            return False
        finally:
            self.is_processing = False
            self.current_processing = None
    
    def process_all_available_dates(self, progress_callback=None, show_footage=False, realtime_display=True):
        """Process all videos from all available date folders"""
        try:
            available_dates = self.get_available_dates()
            
            if not available_dates:
                if progress_callback:
                    progress_callback(0, "📁 No date folders found")
                print("📁 No date folders found")
                return False
            
            if progress_callback:
                progress_callback(5, f"📅 Found {len(available_dates)} date folder(s): {available_dates}")
            print(f"📅 Found {len(available_dates)} date folder(s): {available_dates}")
            
            total_processed = 0
            total_failed = 0
            total_videos = 0
            
            # Count total videos first
            for date_str in available_dates:
                videos = self.get_videos_for_date(date_str)
                processed_videos = self.get_processed_videos(date_str)
                processed_names = {p['video_name'] for p in processed_videos}
                unprocessed_videos = [v for v in videos if v['name'] not in processed_names]
                total_videos += len(unprocessed_videos)
            
            if total_videos == 0:
                if progress_callback:
                    progress_callback(100, "✅ All videos already processed")
                print("✅ All videos already processed")
                return True
            
            current_video = 0
            
            for date_str in available_dates:
                if progress_callback:
                    progress_callback(10, f"📅 Processing videos for {date_str}...")
                print(f"\n📅 Processing videos for {date_str}...")
                
                # Check if there are unprocessed videos for this date
                videos = self.get_videos_for_date(date_str)
                processed_videos = self.get_processed_videos(date_str)
                processed_names = {p['video_name'] for p in processed_videos}
                
                unprocessed_videos = [v for v in videos if v['name'] not in processed_names]
                
                if not unprocessed_videos:
                    if progress_callback:
                        progress_callback(20, f"✅ All videos already processed for {date_str}")
                    print(f"✅ All videos already processed for {date_str}")
                    continue
                
                if progress_callback:
                    progress_callback(15, f"🎬 Found {len(unprocessed_videos)} unprocessed video(s) for {date_str}")
                print(f"🎬 Found {len(unprocessed_videos)} unprocessed video(s) for {date_str}")
                
                # Process videos for this date
                success = self.process_all_videos(date_str, progress_callback, show_footage, realtime_display)
                
                if success:
                    processed_count = len(unprocessed_videos)
                    total_processed += processed_count
                    current_video += processed_count
                    progress = 20 + (current_video / total_videos) * 70  # 20-90% range
                    if progress_callback:
                        progress_callback(progress, f"✅ Successfully processed {processed_count} videos for {date_str}")
                    print(f"✅ Successfully processed {processed_count} videos for {date_str}")
                else:
                    failed_count = len(unprocessed_videos)
                    total_failed += failed_count
                    current_video += failed_count
                    progress = 20 + (current_video / total_videos) * 70  # 20-90% range
                    if progress_callback:
                        progress_callback(progress, f"❌ Failed to process {failed_count} videos for {date_str}")
                    print(f"❌ Failed to process {failed_count} videos for {date_str}")
            
            # Final summary
            if progress_callback:
                progress_callback(95, f"📊 Overall processing complete! Processed: {total_processed}, Failed: {total_failed}")
            print(f"\n📊 Overall processing complete!")
            print(f"✅ Successfully processed: {total_processed} videos")
            print(f"❌ Failed: {total_failed} videos")
            
            if progress_callback:
                progress_callback(100, f"✅ Processing complete! {total_processed} videos processed successfully")
            
            return total_processed > 0
            
        except Exception as e:
            print(f"❌ Error processing all dates: {e}")
            return False
    
    def stop_processing(self):
        """Stop current processing"""
        self.is_processing = False
        print("⏹️ Processing stopped by user")
    
    def get_processing_status(self):
        """Get current processing status"""
        if self.current_processing:
            return {
                'is_processing': self.is_processing,
                'date': self.current_processing['date'],
                'total_videos': self.current_processing['total_videos'],
                'processed': self.current_processing['processed'],
                'failed': self.current_processing['failed'],
                'progress': (self.current_processing['processed'] / self.current_processing['total_videos']) * 100 if self.current_processing['total_videos'] > 0 else 0
            }
        else:
            return {'is_processing': False}
    
    def auto_detect_and_process(self, show_footage=False, realtime_display=True):
        """Automatically detect and process videos in today's input folder"""
        today = date.today().strftime("%Y-%m-%d")
        input_folder = self.base_folder / today / "input"
        
        print(f"🔍 Auto-detecting videos in {input_folder}")
        
        if not input_folder.exists():
            print(f"❌ Input directory not found: {input_folder}")
            return False
        
        # Check for videos
        videos = self.get_videos_for_date(today)
        
        if not videos:
            print(f"📁 No videos found in {input_folder}")
            return False
        
        print(f"🚀 Auto-processing {len(videos)} video(s) found in input folder")
        return self.process_all_videos(today, None, show_footage, realtime_display)
    
    def get_historical_reports(self, date_str=None):
        """Get all historical reports"""
        if date_str:
            dates = [date_str]
        else:
            dates = self.get_available_dates()
        
        all_reports = []
        
        for date_folder in dates:
            reports_folder = self.base_folder / date_folder / "reports"
            if reports_folder.exists():
                for report_file in reports_folder.glob("*.json"):
                    try:
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                            report['date'] = date_folder
                            report['report_path'] = str(report_file)
                            all_reports.append(report)
                    except Exception as e:
                        print(f"⚠️ Error reading report {report_file}: {e}")
        
        return sorted(all_reports, key=lambda x: x.get('processed_at', ''), reverse=True)
    
    def get_report_summary(self, date_str=None):
        """Get summary of all reports"""
        reports = self.get_historical_reports(date_str)
        
        summary = {
            'total_videos': len(reports),
            'completed': len([r for r in reports if r.get('status') == 'completed']),
            'failed': len([r for r in reports if r.get('status') == 'error']),
            'total_faces': sum(r.get('face_count', 0) for r in reports),
            'total_students': sum(r.get('student_count', 0) for r in reports),
            'dates': list(set(r.get('date', '') for r in reports)),
            'reports': reports
        }
        
        return summary

# Test the automated processor
if __name__ == "__main__":
    print("🧪 Testing Automated Video Processor...")
    
    processor = AutomatedVideoProcessor()
    
    # Test folder structure
    print(f"📁 Base folder: {processor.base_folder}")
    print(f"📅 Available dates: {processor.get_available_dates()}")
    
    # Test today's videos
    today = date.today().strftime("%Y-%m-%d")
    videos = processor.get_videos_for_date(today)
    print(f"🎬 Videos for today: {len(videos)}")
    
    for video in videos:
        print(f"  - {video['name']} ({video['size']} bytes)")
    
    # Test historical reports
    reports = processor.get_historical_reports()
    print(f"📊 Historical reports: {len(reports)}")
    
    summary = processor.get_report_summary()
    print(f"📈 Summary: {summary['total_videos']} videos, {summary['completed']} completed, {summary['failed']} failed")
