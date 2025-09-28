#!/usr/bin/env python3
"""
PyInstaller build script for Classroom Analyzer
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_spec_file():
    """Create a custom .spec file for PyInstaller"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['classroom_analyzer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('AI_Model_Weights', 'AI_Model_Weights'),
        ('classroom_labels.json', '.'),
        ('classroom_icon.ico', '.'),
        ('classroom_icon.png', '.'),
        ('firebase_service_account.json', '.'),
    ],
    hiddenimports=[
        'cv2',
        'ultralytics',
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'sklearn',
        'sklearn.metrics.pairwise',
        'sklearn.preprocessing',
        'face_recognition',
        'firebase_admin',
        'firebase_admin.credentials',
        'firebase_admin.db',
        'json',
        'base64',
        'pickle',
        'sqlite3',
        'threading',
        'datetime',
        'pathlib',
        'shutil',
        'io',
        'warnings',
        'traceback',
        'realtime_classroom_analyzer',
        'analysis_viewer',
        'video_face_matcher',
        'vector_face_matcher',
        'lecture_classifier',
        'lightweight_vision_classifier',
        'data_manager',
        'automated_video_processor',
        'firebase_sync',
    ],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='classroom_icon.ico'
)
'''
    
    with open('ClassroomAnalyzer.spec', 'w') as f:
        f.write(spec_content)
    
    print("✅ Created ClassroomAnalyzer.spec file")

def check_dependencies():
    """Check if all required dependencies are installed"""
    # Map package names to import names
    package_mapping = {
        'opencv-python': 'cv2',
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'Pillow': 'PIL',
        'scikit-learn': 'sklearn',
        'face-recognition': 'face_recognition',
        'firebase-admin': 'firebase_admin',
        'pyinstaller': 'PyInstaller'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def create_requirements_file():
    """Create requirements.txt for the project"""
    requirements = """opencv-python>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
scikit-learn>=1.3.0
face-recognition>=1.3.0
firebase-admin>=6.2.0
pyinstaller>=5.13.0
matplotlib>=3.7.0
accelerate>=0.20.0
tensorflow>=2.13.0
pyasn1>=0.4.8
protobuf>=4.21.0
cachetools>=5.3.0
attrs>=23.1.0
pydantic>=2.0.0
"""
    
    with open('requirements_build.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements_build.txt")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🚀 Building executable with PyInstaller...")
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # Build command
    cmd = [
        'pyinstaller',
        '--clean',
        '--noconfirm',
        'ClassroomAnalyzer.spec'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build successful!")
        print(f"📁 Executable location: dist/ClassroomAnalyzer.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main build process"""
    print("🏗️  Classroom Analyzer - PyInstaller Build Script")
    print("=" * 50)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Create requirements file
    print("\n📝 Creating requirements file...")
    create_requirements_file()
    
    # Create spec file
    print("\n📝 Creating PyInstaller spec file...")
    create_spec_file()
    
    # Build executable
    print("\n🔨 Building executable...")
    if build_executable():
        print("\n🎉 Build completed successfully!")
        print("📁 Find your executable in: dist/ClassroomAnalyzer.exe")
        return True
    else:
        print("\n❌ Build failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
