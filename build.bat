@echo off
echo Building Classroom Analyzer Executable...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements
echo Installing requirements...
pip install -r requirements_build.txt

REM Run the build script
echo.
echo Running build script...
python build_exe.py

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable location: dist\ClassroomAnalyzer.exe
echo.
pause

