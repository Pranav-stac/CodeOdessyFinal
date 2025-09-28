# Classroom Analyzer - Build Instructions

## Prerequisites

1. **Python 3.8+** installed
2. **All dependencies** installed (see requirements_build.txt)
3. **All model files** present in AI_Model_Weights folder
4. **All project files** present

## Quick Build (Recommended)

### Option 1: Using Batch File (Windows)
```bash
# Double-click build.bat or run in command prompt
build.bat
```

### Option 2: Using Python Script
```bash
# Install requirements
pip install -r requirements_build.txt

# Run build script
python build_exe.py
```

## Manual Build

### Step 1: Test Components
```bash
python test_build_components.py
```

### Step 2: Install Requirements
```bash
pip install -r requirements_build.txt
```

### Step 3: Build with PyInstaller
```bash
pyinstaller --clean --noconfirm ClassroomAnalyzer.spec
```

## Build Output

- **Executable**: `dist/ClassroomAnalyzer.exe`
- **Size**: ~500MB-1GB (includes all dependencies and models)
- **Requirements**: Windows 10/11, 4GB+ RAM

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**
   ```bash
   pip install -r requirements_build.txt
   ```

2. **Model Files Missing**
   - Ensure AI_Model_Weights folder contains:
     - yolov8s.pt
     - yolov8n-pose.pt
     - yolov12s-face.pt

3. **Firebase Issues**
   - Ensure firebase_service_account.json is present
   - Check Firebase credentials

4. **Large File Size**
   - Normal for ML applications
   - Consider using --exclude-module for unused packages

### Build Optimization:

```bash
# For smaller file size (may break some features)
pyinstaller --onefile --windowed --exclude-module matplotlib --exclude-module tensorflow classroom_analyzer_gui.py
```

## Features Included in Build:

✅ Real-time video analysis  
✅ Face detection and matching  
✅ Pose analysis  
✅ Engagement tracking  
✅ Attendance management  
✅ Lecture classification  
✅ Automated video processing  
✅ Firebase synchronization  
✅ Historical reports  
✅ GUI interface  

## File Structure Required:

```
ClassroomAnalyzer/
├── classroom_analyzer_gui.py
├── realtime_classroom_analyzer.py
├── analysis_viewer.py
├── video_face_matcher.py
├── vector_face_matcher.py
├── lecture_classifier.py
├── lightweight_vision_classifier.py
├── data_manager.py
├── automated_video_processor.py
├── firebase_sync.py
├── classroom_labels.json
├── classroom_icon.ico
├── classroom_icon.png
├── firebase_service_account.json
├── AI_Model_Weights/
│   ├── yolov8s.pt
│   ├── yolov8n-pose.pt
│   └── yolov12s-face.pt
└── requirements_build.txt
```