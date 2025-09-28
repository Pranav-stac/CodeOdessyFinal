"""
Test script for Firebase sync functionality
"""

import os
import json
from datetime import datetime, date
from firebase_sync import FirebaseSync, FIREBASE_CONFIG

def test_firebase_sync():
    """Test Firebase sync functionality"""
    print("🧪 Testing Firebase Sync Integration")
    print("=" * 50)
    
    # Initialize Firebase sync
    print("🔄 Initializing Firebase sync...")
    firebase_sync = FirebaseSync(FIREBASE_CONFIG)
    
    if not firebase_sync.initialized:
        print("❌ Firebase not initialized. Testing local backup...")
        
        # Test local backup
        today = date.today().strftime("%Y-%m-%d")
        success = firebase_sync.save_local_backup(today)
        
        if success:
            print("✅ Local backup test successful")
            print(f"💾 Check firebase_backups/daily_data_{today}.json")
        else:
            print("❌ Local backup test failed")
        
        return
    
    print("✅ Firebase initialized successfully")
    
    # Test daily sync
    print("\n🔄 Testing daily data sync...")
    today = date.today().strftime("%Y-%m-%d")
    success = firebase_sync.sync_daily_data(today)
    
    if success:
        print("✅ Daily sync test successful")
    else:
        print("❌ Daily sync test failed")
    
    # Test data collection
    print("\n📊 Testing data collection...")
    daily_data = firebase_sync.collect_daily_data(today)
    
    print(f"📅 Date: {daily_data['date']}")
    
    # Parse JSON strings for display
    try:
        engagement_data = json.loads(daily_data.get('comprehensive_analysis_report', '{}'))
        students = engagement_data.get('overview_statistics', {}).get('total_students_tracked', 0)
        print(f"👥 Students: {students}")
    except:
        print(f"👥 Students: 0")
    
    try:
        video_data = json.loads(daily_data.get('video_metadata', '{}'))
        videos = len(video_data.get('processed_videos', []))
        print(f"🎬 Videos: {videos}")
    except:
        print(f"🎬 Videos: 0")
    
    try:
        attendance_data = json.loads(daily_data.get('attendance_data', '{}'))
        appearances = attendance_data.get('summary', {}).get('total_appearances', 0)
        print(f"📊 Appearances: {appearances}")
    except:
        print(f"📊 Appearances: 0")
    
    try:
        face_data = json.loads(daily_data.get('face_database', '{}'))
        face_matches = len(face_data)
        print(f"🔍 Face Matches: {face_matches}")
    except:
        print(f"🔍 Face Matches: 0")
    
    # Test historical sync
    print("\n📚 Testing historical data sync...")
    success = firebase_sync.sync_all_historical_data()
    
    if success:
        print("✅ Historical sync test successful")
    else:
        print("❌ Historical sync test failed")
    
    print("\n✅ Firebase sync test completed!")

def test_data_structure():
    """Test the data structure being sent to Firebase"""
    print("\n🧪 Testing Data Structure")
    print("=" * 30)
    
    firebase_sync = FirebaseSync(FIREBASE_CONFIG)
    today = date.today().strftime("%Y-%m-%d")
    
    # Collect sample data
    daily_data = firebase_sync.collect_daily_data(today)
    
    # Print data structure
    print("📊 Data Structure:")
    print(f"├── date: {daily_data['date']}")
    print(f"├── sync_timestamp: {daily_data['sync_timestamp']}")
    print(f"├── comprehensive_analysis_report: JSON string (complete data)")
    print(f"├── face_database: JSON string (complete face data)")
    print(f"├── attendance_data: JSON string (complete attendance data)")
    print(f"├── video_metadata: JSON string (complete video data)")
    print(f"├── lecture_classifications: JSON string (complete classification data)")
    print(f"├── analysis_reports: JSON string (complete reports data)")
    print(f"├── raw_data_files: JSON string (complete files data)")
    print(f"└── statistics: JSON string (complete statistics)")
    
    # Show sample data sizes
    print(f"\n📊 Data Sizes:")
    for key, value in daily_data.items():
        if isinstance(value, str) and key != 'date' and key != 'sync_timestamp':
            size_kb = len(value) / 1024
            print(f"├── {key}: {size_kb:.1f} KB")
    
    # Save sample data to file
    sample_file = f"firebase_sample_data_{today}.json"
    with open(sample_file, 'w') as f:
        json.dump(daily_data, f, indent=2)
    
    print(f"\n💾 Sample data saved to: {sample_file}")

if __name__ == "__main__":
    test_firebase_sync()
    test_data_structure()
