# 🎓 Classroom Video Analyzer - Build Instructions

## 📋 Overview
This guide will help you convert the Classroom Video Analyzer into a standalone executable that anyone can download and use without installing Python or dependencies.

## 🚀 Quick Start (Recommended)

### Method 1: Auto-Py-to-Exe (Easiest)
1. **Install Python** (3.8+ recommended)
2. **Run the setup script:**
   ```bash
   python build_with_auto_py_to_exe.py
   ```
3. **Choose option 1** for GUI-based building
4. **Launch the build interface:**
   ```bash
   python -m auto_py_to_exe
   ```
5. **Configure in the GUI:**
   - Script Location: `classroom_analyzer_gui.py`
   - Onefile: ✅ True
   - Window Based: ✅ True
   - Add Data Files:
     - `AI_Model_Weights` folder
     - `classroom_labels.json` file
   - Hidden Imports: `cv2, ultralytics, torch, numpy, PIL, tkinter`
6. **Click "CONVERT .PY TO .EXE"**

### Method 2: Command Line (Advanced)
```bash
# Install dependencies
pip install -r requirements.txt

# Build executable
python build_executable.py
```

## 📁 File Structure Required
```
Your Project/
├── classroom_analyzer_gui.py          # Main GUI application
├── realtime_classroom_analyzer.py     # Core analysis engine
├── classroom_labels.json              # Label schema
├── AI_Model_Weights/                  # AI model weights
│   └── AI_Model_Weights/
│       ├── yolov8s.pt
│       ├── yolov8n-pose.pt
│       └── yolov12s-face.pt
├── requirements.txt                   # Dependencies
├── build_executable.py               # Build script
├── build_with_auto_py_to_exe.py      # GUI build script
└── BUILD_INSTRUCTIONS.md             # This file
```

## 🔧 Prerequisites

### System Requirements
- **Python 3.8+** (3.9+ recommended)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ free disk space**
- **Windows 10/11, macOS 10.14+, or Linux**

### Python Dependencies
```bash
pip install -r requirements.txt
```

### AI Model Weights
Ensure these files are in `AI_Model_Weights/AI_Model_Weights/`:
- `yolov8s.pt` - Person detection
- `yolov8n-pose.pt` - Pose estimation
- `yolov12s-face.pt` - Face detection

## 🛠️ Build Process Details

### Step 1: Prepare Environment
```bash
# Clone or download the project
# Ensure all files are in the correct locations
# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Test the Application
```bash
# Test the GUI application
python classroom_analyzer_gui.py
```

### Step 3: Build Executable

#### Option A: Using Auto-Py-to-Exe
1. Install auto-py-to-exe:
   ```bash
   pip install auto-py-to-exe
   ```

2. Launch the GUI:
   ```bash
   python -m auto_py_to_exe
   ```

3. Configure the build:
   - **Script Location**: `classroom_analyzer_gui.py`
   - **Onefile**: ✅ True
   - **Window Based**: ✅ True
   - **Additional Files**:
     - `AI_Model_Weights` → `AI_Model_Weights`
     - `classroom_labels.json` → `.`
   - **Hidden Imports**: `cv2, ultralytics, torch, numpy, PIL, tkinter`

#### Option B: Using PyInstaller Directly
```bash
pyinstaller --onefile --windowed --name=ClassroomAnalyzer \
  --add-data="AI_Model_Weights;AI_Model_Weights" \
  --add-data="classroom_labels.json;." \
  --hidden-import=cv2 \
  --hidden-import=ultralytics \
  --hidden-import=torch \
  --hidden-import=numpy \
  --hidden-import=PIL \
  --hidden-import=tkinter \
  classroom_analyzer_gui.py
```

### Step 4: Create Distribution Package
After building, create a distribution package:

1. **Create installer folder:**
   ```
   ClassroomAnalyzer_v1.0/
   ├── ClassroomAnalyzer.exe (or ClassroomAnalyzer on Mac/Linux)
   ├── AI_Model_Weights/
   │   └── AI_Model_Weights/
   │       ├── yolov8s.pt
   │       ├── yolov8n-pose.pt
   │       └── yolov12s-face.pt
   ├── classroom_labels.json
   └── README.txt
   ```

2. **Zip the folder** for distribution

## 📦 Distribution Options

### Option 1: Simple Zip Distribution
- Zip the entire `ClassroomAnalyzer_v1.0` folder
- Users extract and run the executable
- **Pros**: Simple, works everywhere
- **Cons**: Users need to extract files

### Option 2: Installer Package (Windows)
Use tools like:
- **Inno Setup** - Free, professional installer
- **NSIS** - Free, flexible installer
- **Advanced Installer** - Commercial, feature-rich

### Option 3: App Store Distribution
- **Microsoft Store** (Windows)
- **Mac App Store** (macOS)
- **Snap Store** (Linux)

## 🔍 Troubleshooting

### Common Issues

#### 1. "Module not found" errors
**Solution**: Add missing modules to hidden imports:
```bash
--hidden-import=module_name
```

#### 2. Large executable size
**Solutions**:
- Use `--exclude-module` for unused modules
- Enable UPX compression
- Use `--onedir` instead of `--onefile`

#### 3. AI models not found
**Solution**: Ensure model weights are in the correct directory structure

#### 4. GUI not appearing
**Solution**: Check that `--windowed` flag is used (not `--console`)

### Build Optimization

#### Reduce File Size
```bash
# Exclude unnecessary modules
--exclude-module=matplotlib
--exclude-module=pandas
--exclude-module=jupyter

# Use UPX compression
--upx-dir=/path/to/upx
```

#### Improve Performance
```bash
# Use onefile for single executable
--onefile

# Disable console window
--windowed

# Strip debug information
--strip
```

## 🧪 Testing the Executable

### Test Checklist
- [ ] Executable runs without errors
- [ ] GUI opens correctly
- [ ] Video file selection works
- [ ] Analysis starts and completes
- [ ] Results are saved correctly
- [ ] All AI models load properly
- [ ] No missing dependencies

### Test on Different Systems
- [ ] Windows 10/11
- [ ] macOS 10.14+
- [ ] Linux (Ubuntu/CentOS)
- [ ] Different Python versions
- [ ] Different hardware configurations

## 📊 Build Statistics

### Typical Build Sizes
- **Onefile**: 800MB - 1.2GB
- **Onedir**: 600MB - 900MB
- **Compressed**: 200MB - 400MB

### Build Time
- **First build**: 5-15 minutes
- **Subsequent builds**: 2-5 minutes
- **Clean build**: 10-20 minutes

## 🚀 Advanced Features

### Code Signing (Windows)
```bash
# Sign the executable
signtool sign /f certificate.pfx /p password ClassroomAnalyzer.exe
```

### Version Information
Add version info to the executable:
```python
# Create version_info.txt
version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Your Company'),
         StringStruct(u'FileDescription', u'Classroom Video Analyzer'),
         StringStruct(u'FileVersion', u'1.0.0.0'),
         StringStruct(u'InternalName', u'ClassroomAnalyzer'),
         StringStruct(u'LegalCopyright', u'Copyright (C) 2024'),
         StringStruct(u'OriginalFilename', u'ClassroomAnalyzer.exe'),
         StringStruct(u'ProductName', u'Classroom Video Analyzer'),
         StringStruct(u'ProductVersion', u'1.0.0.0')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
```

## 📞 Support

### Getting Help
1. Check the application logs
2. Verify all dependencies are installed
3. Ensure AI model weights are present
4. Check system requirements

### Common Solutions
- **Reinstall Python** if there are dependency issues
- **Use virtual environment** to avoid conflicts
- **Check file permissions** on the executable
- **Run as administrator** if needed (Windows)

## 🎉 Success!

Once built successfully, you'll have:
- ✅ Standalone executable
- ✅ No Python installation required for users
- ✅ All dependencies bundled
- ✅ Professional GUI interface
- ✅ Easy distribution

Your Classroom Video Analyzer is now ready for distribution! 🚀

