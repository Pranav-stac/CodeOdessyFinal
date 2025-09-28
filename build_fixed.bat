@echo off
echo Building Classroom Analyzer with FIXED face_recognition support...
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building executable with face_recognition data files...
pyinstaller --onefile --windowed ^
--name=ClassroomAnalyzer ^
--add-data="AI_Model_Weights;AI_Model_Weights" ^
--add-data="classroom_labels.json;." ^
--add-data="classroom_icon.ico;." ^
--add-data="classroom_icon.png;." ^
--add-data="firebase_service_account.json;." ^
--add-data="C:\Users\HP\AppData\Local\Programs\Python\Python311\Lib\site-packages\face_recognition_models;face_recognition_models" ^
--hidden-import=cv2 ^
--hidden-import=ultralytics ^
--hidden-import=torch ^
--hidden-import=torchvision ^
--hidden-import=numpy ^
--hidden-import=PIL ^
--hidden-import=sklearn ^
--hidden-import=face_recognition ^
--hidden-import=face_recognition_models ^
--hidden-import=firebase_admin ^
--hidden-import=matplotlib ^
--hidden-import=accelerate ^
--hidden-import=tensorflow ^
--hidden-import=pyasn1 ^
--hidden-import=protobuf ^
--hidden-import=cachetools ^
--hidden-import=attrs ^
--hidden-import=pydantic ^
--hidden-import=jmespath ^
--hidden-import=peft ^
--hidden-import=dlib ^
--icon=classroom_icon.ico ^
classroom_analyzer_gui.py

if exist dist\ClassroomAnalyzer.exe (
    echo.
    echo ✅ BUILD SUCCESSFUL!
    echo 📁 Executable: dist\ClassroomAnalyzer.exe
    echo 🎉 All features included with face_recognition support:
    echo    - YOLO Models (Detection, Pose, Face)
    echo    - Vector Face Matching
    echo    - Vision LLM Classification
    echo    - Firebase Sync
    echo    - Automated Processing
    echo    - Attendance Tracking
    echo    - Engagement Analysis
    echo    - Face Recognition Models
) else (
    echo.
    echo ❌ BUILD FAILED!
)

pause

