# 🚀 Quick Start Guide - Classroom Video Analyzer

## ⚡ 5-Minute Setup

### 1. **Run Complete Setup**
```bash
python setup_complete.py
```
This will check everything and prepare your environment.

### 2. **Test the Application**
```bash
python classroom_analyzer_gui.py
```
Make sure the GUI opens and works correctly.

### 3. **Build Executable (Choose One)**

#### Option A: GUI Builder (Recommended)
```bash
python -m auto_py_to_exe
```
- Set Script: `classroom_analyzer_gui.py`
- Onefile: ✅ True
- Window Based: ✅ True
- Add Data: `AI_Model_Weights` folder + `classroom_labels.json`

#### Option B: Command Line
```bash
python build_executable.py
```

#### Option C: Use Build Scripts
- **Windows**: Double-click `build_windows.bat`
- **Mac/Linux**: Run `./build_unix.sh`

## 📁 Required Files Structure
```
Your Project/
├── classroom_analyzer_gui.py          ✅ Main GUI
├── realtime_classroom_analyzer.py     ✅ Core engine
├── classroom_labels.json              ✅ Label schema
├── AI_Model_Weights/                  ✅ AI models
│   └── AI_Model_Weights/
│       ├── yolov8s.pt                ✅ Person detection
│       ├── yolov8n-pose.pt           ✅ Pose estimation
│       └── yolov12s-face.pt          ✅ Face detection
├── requirements.txt                   ✅ Dependencies
├── setup_complete.py                  ✅ Setup script
└── build_executable.py               ✅ Build script
```

## 🎯 What You Get

### **Standalone Executable**
- ✅ No Python installation needed for users
- ✅ All dependencies bundled
- ✅ Professional GUI interface
- ✅ Easy to distribute

### **Features**
- 🎬 **Video Analysis**: Process classroom videos
- 👥 **Student Tracking**: Detect and track students
- 😊 **Face Recognition**: Identify and track faces
- 📊 **Activity Detection**: Writing, raising hand, listening, distracted
- 📈 **Engagement Analysis**: Real-time engagement levels
- 📁 **Comprehensive Reports**: Detailed analysis results

## 🔧 Troubleshooting

### **"Module not found" errors**
```bash
pip install -r requirements.txt
```

### **"AI models not found"**
- Ensure `AI_Model_Weights/AI_Model_Weights/` folder exists
- Download models from: https://github.com/ultralytics/ultralytics

### **GUI doesn't open**
- Check Python version (3.8+ required)
- Install tkinter: `pip install tk`

### **Build fails**
- Use `--onefile` instead of `--onedir`
- Add missing modules to `--hidden-import`
- Check file paths are correct

## 📦 Distribution

### **Create Distribution Package**
1. Build executable
2. Create folder: `ClassroomAnalyzer_v1.0/`
3. Copy files:
   - `ClassroomAnalyzer.exe`
   - `AI_Model_Weights/` folder
   - `classroom_labels.json`
   - `README.txt`
4. Zip the folder
5. Share with users

### **User Instructions**
1. Extract the zip file
2. Run `ClassroomAnalyzer.exe`
3. Select video file
4. Click "Start Analysis"
5. View results

## 🎉 Success!

Your Classroom Video Analyzer is now ready for distribution! 

**File sizes:**
- Executable: ~800MB - 1.2GB
- Distribution package: ~1GB - 1.5GB

**Supported platforms:**
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu/CentOS)

**No technical knowledge required for end users!** 🚀

