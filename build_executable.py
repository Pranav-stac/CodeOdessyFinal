"""
Build script for creating standalone executable
Run this script to build the Classroom Analyzer into a distributable executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'opencv-python',
        'ultralytics', 
        'numpy',
        'torch',
        'pyinstaller'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def create_spec_file():
    """Create PyInstaller spec file for better control"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Data files to include
datas = [
    ('AI_Model_Weights', 'AI_Model_Weights'),
    ('classroom_labels.json', '.'),
]

# Hidden imports (modules that PyInstaller might miss)
hiddenimports = [
    'cv2',
    'ultralytics',
    'torch',
    'torchvision',
    'numpy',
    'PIL',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'collections',
    'datetime',
    'threading',
    'json',
    'base64',
    'math',
    'random',
    'time',
    'os',
    'sys',
    'pathlib',
    'webbrowser',
    'subprocess'
]

a = Analysis(
    ['classroom_analyzer_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ClassroomAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Add icon file if available
)
'''
    
    with open('ClassroomAnalyzer.spec', 'w') as f:
        f.write(spec_content)
    
    print("📝 Created PyInstaller spec file")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # Build command
    cmd = [
        'pyinstaller',
        '--onefile',  # Single executable file
        '--windowed',  # No console window
        '--name=ClassroomAnalyzer',
        '--add-data=AI_Model_Weights;AI_Model_Weights',  # Windows
        '--add-data=classroom_labels.json;.',
        '--add-data=classroom_icon.ico;.',
        '--add-data=classroom_icon.png;.',
        '--hidden-import=cv2',
        '--hidden-import=ultralytics',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=numpy',
        '--hidden-import=PIL',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.scrolledtext',
        '--icon=classroom_icon.ico',  # Add custom icon
        'classroom_analyzer_gui.py'  # Use correct GUI file
    ]
    
    # Adjust for different operating systems
    if sys.platform == "linux" or sys.platform == "darwin":
        cmd[4] = '--add-data=AI_Model_Weights:AI_Model_Weights'
        cmd[5] = '--add-data=classroom_labels.json:.'
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_installer_package():
    """Create a simple installer package"""
    print("📦 Creating installer package...")
    
    # Create installer directory
    installer_dir = Path("ClassroomAnalyzer_Installer")
    installer_dir.mkdir(exist_ok=True)
    
    # Copy executable
    exe_path = Path("dist/ClassroomAnalyzer.exe" if sys.platform == "win32" else "dist/ClassroomAnalyzer")
    if exe_path.exists():
        shutil.copy2(exe_path, installer_dir / exe_path.name)
    
    # Copy model weights
    if os.path.exists("AI_Model_Weights"):
        shutil.copytree("AI_Model_Weights", installer_dir / "AI_Model_Weights", dirs_exist_ok=True)
    
    # Copy labels file
    if os.path.exists("classroom_labels.json"):
        shutil.copy2("classroom_labels.json", installer_dir)
    
    # Create README
    readme_content = """# Classroom Video Analyzer

## Installation
1. Extract all files to a folder
2. Run ClassroomAnalyzer.exe (Windows) or ClassroomAnalyzer (Mac/Linux)

## Requirements
- Windows 10/11, macOS 10.14+, or Linux
- 4GB+ RAM
- 2GB+ free disk space
- Modern CPU (GPU recommended for faster processing)

## Usage
1. Launch the application
2. Click "Browse" to select a video file
3. Choose output directory (optional)
4. Select analysis options
5. Click "Start Analysis"
6. View results in the output directory

## Supported Video Formats
- MP4, AVI, MOV, MKV, WMV, FLV, WebM

## Troubleshooting
- Ensure AI model weights are in the AI_Model_Weights folder
- Check that you have sufficient disk space
- For GPU acceleration, ensure CUDA drivers are installed

## Support
For technical support, check the application logs or contact the developer.
"""
    
    with open(installer_dir / "README.txt", 'w') as f:
        f.write(readme_content)
    
    # Create batch file for easy launch (Windows)
    if sys.platform == "win32":
        batch_content = """@echo off
echo Starting Classroom Video Analyzer...
ClassroomAnalyzer.exe
pause
"""
        with open(installer_dir / "Run_ClassroomAnalyzer.bat", 'w') as f:
            f.write(batch_content)
    
    print(f"✅ Installer package created: {installer_dir}")
    return installer_dir

def main():
    """Main build process"""
    print("🎓 Classroom Video Analyzer - Build Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    if not build_executable():
        return
    
    # Create installer package
    installer_dir = create_installer_package()
    
    print("\n🎉 Build complete!")
    print(f"📁 Executable location: dist/")
    print(f"📦 Installer package: {installer_dir}")
    print("\nTo distribute:")
    print(f"1. Zip the {installer_dir} folder")
    print("2. Share with users")
    print("3. Users extract and run the executable")

if __name__ == "__main__":
    main()
