"""
Setup script for automated video processing
Creates folder structure and provides instructions
"""

import os
from pathlib import Path
from datetime import date

def setup_automated_processing():
    """Setup the automated video processing system"""
    print("🚀 Setting up Automated Video Processing System")
    print("=" * 50)
    
    # Create base folders
    base_folder = Path("video_processing")
    analysis_folder = Path("analysis_history")
    
    # Create today's folder structure
    today = date.today().strftime("%Y-%m-%d")
    today_folder = base_folder / today
    
    folders_to_create = [
        base_folder,
        analysis_folder,
        today_folder,
        today_folder / "input",
        today_folder / "processed", 
        today_folder / "reports"
    ]
    
    print("📁 Creating folder structure...")
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {folder}")
    
    # Create README file
    readme_content = f"""
# Automated Video Processing System

## Folder Structure
```
video_processing/
├── {today}/
│   ├── input/          # Place your videos here
│   ├── processed/      # Processed videos are moved here
│   └── reports/        # Analysis reports are saved here
└── [other dates]/
    ├── input/
    ├── processed/
    └── reports/

analysis_history/
└── [analysis results for each video]
```

## How to Use

### 1. Add Videos
- Place your video files in: `video_processing/{today}/input/`
- Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv

### 2. Process Videos
- Open the Classroom Analyzer GUI
- Click "🚀 Auto Process Videos" button
- The system will automatically process all videos in the input folder

### 3. View Results
- Click "📚 View History" to see all processed videos
- Click "📁 Open Video Folder" to access the folder structure
- Each video gets its own analysis report

### 4. Organization
- Videos are organized by date (YYYY-MM-DD format)
- Processed videos are moved to the "processed" folder
- Analysis reports are saved in the "reports" folder
- All analysis data is stored in "analysis_history"

## Features
- ✅ One-click processing of multiple videos
- ✅ Automatic folder organization by date
- ✅ Historical report viewing
- ✅ Progress tracking
- ✅ Error handling and logging
- ✅ Face matching across videos
- ✅ Lecture type classification
- ✅ Attendance tracking

## Notes
- The system will skip videos that have already been processed
- You can stop processing at any time
- All results are automatically saved and organized
- Previous analysis reports are always accessible
"""
    
    readme_file = base_folder / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n📝 Created README: {readme_file}")
    
    # Create sample video placeholder
    sample_file = today_folder / "input" / "README.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(f"""
# Add your videos here

Place your video files in this folder and use the "🚀 Auto Process Videos" button in the GUI.

Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv

Example:
- lecture_1.mp4
- classroom_session.avi
- student_presentation.mov

The system will automatically process all videos and organize the results.
""")
    
    print(f"📄 Created sample instructions: {sample_file}")
    
    print(f"\n🎉 Setup complete!")
    print(f"📁 Video folder: {today_folder / 'input'}")
    print(f"📊 Analysis folder: {analysis_folder}")
    print(f"\n💡 Next steps:")
    print(f"  1. Add videos to: {today_folder / 'input'}")
    print(f"  2. Run: python classroom_analyzer_gui.py")
    print(f"  3. Click '🚀 Auto Process Videos'")
    print(f"  4. View results with '📚 View History'")

if __name__ == "__main__":
    setup_automated_processing()
