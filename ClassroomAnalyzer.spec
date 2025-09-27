# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['classroom_analyzer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('AI_Model_Weights', 'AI_Model_Weights'), ('classroom_labels.json', '.'), ('classroom_icon.ico', '.'), ('classroom_icon.png', '.')],
    hiddenimports=['cv2', 'ultralytics', 'torch', 'torchvision', 'numpy', 'PIL', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.scrolledtext', 'sys', 'random'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
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
    icon=['classroom_icon.ico'],
)
