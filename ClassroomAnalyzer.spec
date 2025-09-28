# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['classroom_analyzer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('AI_Model_Weights', 'AI_Model_Weights'), ('classroom_labels.json', '.'), ('classroom_icon.ico', '.'), ('classroom_icon.png', '.'), ('firebase_service_account.json', '.')],
    hiddenimports=['cv2', 'ultralytics', 'torch', 'torchvision', 'numpy', 'PIL', 'sklearn', 'face_recognition', 'face_recognition_models', 'firebase_admin', 'matplotlib', 'accelerate', 'tensorflow', 'pyasn1', 'protobuf', 'cachetools', 'attrs', 'pydantic', 'jmespath', 'peft', 'dlib'],
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
