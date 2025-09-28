"""
Test script for attendance report functionality
"""

import json
from enhanced_face_matcher import EnhancedFaceMatcher
from datetime import datetime

def test_attendance_report():
    """Test the attendance report generation"""
    print("🧪 Testing Attendance Report Generation")
    print("=" * 50)
    
    # Initialize face matcher
    face_matcher = EnhancedFaceMatcher()
    
    # Load existing face database
    print("📊 Loading face database...")
    face_matcher.load_face_database()
    
    # Get attendance summary
    print("📋 Generating attendance summary...")
    attendance_summary = face_matcher.get_attendance_summary()
    
    print(f"✅ Found {len(attendance_summary)} students in database")
    
    # Create comprehensive attendance report
    attendance_report = {
        'summary': {
            'total_persons': len(attendance_summary),
            'total_videos_processed': len(set([video for person_data in attendance_summary.values() for video in person_data.get('videos', [])])),
            'total_appearances': sum([person_data.get('total_appearances', 0) for person_data in attendance_summary.values()])
        },
        'persons': attendance_summary,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n📊 ATTENDANCE REPORT SUMMARY:")
    print("=" * 30)
    print(f"👥 Total Students: {attendance_report['summary']['total_persons']}")
    print(f"🎬 Videos Processed: {attendance_report['summary']['total_videos_processed']}")
    print(f"📊 Total Appearances: {attendance_report['summary']['total_appearances']}")
    print(f"📅 Generated: {attendance_report['generated_at']}")
    
    print("\n👥 STUDENT DETAILS:")
    print("=" * 30)
    
    for person_id, person_data in attendance_summary.items():
        total_appearances = person_data.get('total_appearances', 0)
        videos = person_data.get('videos', [])
        first_seen = person_data.get('first_seen', 'Unknown')
        last_seen = person_data.get('last_seen', 'Unknown')
        unique_videos = len(set(videos)) if videos else 0
        
        print(f"\n👤 Student: {person_id}")
        print(f"   📊 Total Appearances: {total_appearances}")
        print(f"   🎬 Videos Attended: {unique_videos}")
        print(f"   📅 First Seen: {first_seen}")
        print(f"   📅 Last Seen: {last_seen}")
        print(f"   🎯 Videos: {', '.join(set(videos)) if videos else 'None'}")
        
        # Face matching details
        if person_id in face_matcher.face_database:
            face_data = face_matcher.face_database[person_id]
            encodings_count = len(face_data.get('encodings', []))
            features_count = len(face_data.get('image_features', []))
            images_count = len(face_data.get('images', []))
            
            print(f"   🔍 Face Encodings: {encodings_count}")
            print(f"   🖼️ Image Features: {features_count}")
            print(f"   📸 Stored Images: {images_count}")
    
    print("\n✅ Attendance report test completed successfully!")
    
    # Save report to file
    report_file = f"attendance_report_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(attendance_report, f, indent=2)
    
    print(f"💾 Report saved to: {report_file}")

if __name__ == "__main__":
    test_attendance_report()
