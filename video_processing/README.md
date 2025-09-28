
# Automated Video Processing System

## Folder Structure
```
video_processing/
├── 2025-09-28/
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
- Place your video files in: `video_processing/2025-09-28/input/`
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
