"""
Final Fixed Classroom Analyzer GUI Application
All overlapping issues resolved, custom icon working
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
import webbrowser
import base64
from PIL import Image, ImageTk
import io

# Import sys with fallback for executable environment
try:
    import sys
except ImportError:
    # Fallback for executable environment
    sys = None
    print("⚠️ sys module not available in executable environment")

# Import your existing analyzer
from realtime_classroom_analyzer import RealtimeClassroomAnalyzer
from analysis_viewer import AnalysisViewer
from video_face_matcher import VideoFaceMatcher
from vector_face_matcher import VectorFaceMatcher as EnhancedFaceMatcher
from lecture_classifier import LectureClassifier
from lightweight_vision_classifier import LightweightVisionClassifier as VisionLectureClassifier
from data_manager import DataManager
from automated_video_processor import AutomatedVideoProcessor
from firebase_sync import FirebaseSync, FIREBASE_CONFIG

class ClassroomAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Classroom Video Analyzer")
        self.root.geometry("1400x1200")  # Larger window for all features
        self.root.configure(bg='#f0f0f0')
        self.root.minsize(1300, 900)  # Set minimum size to prevent cropping
        self.root.resizable(True, True)  # Allow resizing
        
        # Set custom icon
        self.set_custom_icon()
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="analysis_results")
        self.headless_mode = tk.BooleanVar(value=False)
        self.save_frames = tk.BooleanVar(value=False)
        self.show_footage = tk.BooleanVar(value=True)
        self.real_time_display = tk.BooleanVar(value=True)
        self.is_analyzing = False
        self.analyzer = None
        self.model_status = {"detection": False, "pose": False, "face": False}
        
        # New components
        self.face_matcher = VideoFaceMatcher()
        self.enhanced_face_matcher = EnhancedFaceMatcher()
        self.lecture_classifier = LectureClassifier()
        self.vision_classifier = VisionLectureClassifier()
        self.data_manager = DataManager()
        self.analysis_viewer = None
        self.auto_processor = AutomatedVideoProcessor()
        self.current_session_id = None
        
        # Initialize Firebase sync
        self.firebase_sync = FirebaseSync(FIREBASE_CONFIG)
        
        # Initialize auto-processing preferences
        self.auto_show_footage = tk.BooleanVar(value=False)
        self.auto_realtime_display = tk.BooleanVar(value=True)
        
        # Auto-detect and process videos on startup (disabled - only run when button clicked)
        # self.auto_detect_videos()
        
        # Check classification method
        if hasattr(self.lecture_classifier, 'model') and self.lecture_classifier.model:
            self.classification_method = "LLM-based"
        else:
            self.classification_method = "Rule-based"
        
        # Create GUI
        self.create_widgets()
        self.setup_styles()
        
        # Check model status after GUI is created
        self.root.after(1000, self.check_model_status)
        
    def set_custom_icon(self):
        """Set custom icon for the application"""
        try:
            if os.path.exists('classroom_icon.ico'):
                self.root.iconbitmap('classroom_icon.ico')
                print("✅ Custom icon loaded successfully")
            else:
                print("⚠️ Custom icon not found, creating simple icon...")
                self.create_simple_icon()
        except Exception as e:
            print(f"❌ Could not set custom icon: {e}")
            try:
                self.create_simple_icon()
            except:
                print("❌ Failed to create fallback icon")
    
    def create_simple_icon(self):
        """Create a simple icon programmatically"""
        try:
            from PIL import Image, ImageDraw
            size = 64
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw classroom icon
            draw.ellipse([4, 4, size-4, size-4], fill=(52, 152, 219), outline=(41, 128, 185), width=3)
            draw.rectangle([8, 12, size-8, 24], fill=(45, 45, 45), outline=(30, 30, 30))
            
            # Chalk lines
            for i in range(3):
                y = 16 + i * 2
                draw.line([12, y, size-12, y], fill=(255, 255, 255), width=1)
            
            # Desk
            draw.rectangle([10, 32, size-10, 36], fill=(139, 69, 19), outline=(101, 67, 33))
            
            # Students
            for i in range(3):
                x = 16 + i * 12
                draw.ellipse([x, 28, x+4, 32], fill=(255, 220, 177))
                draw.rectangle([x+1, 32, x+3, 40], fill=(52, 152, 219))
                draw.line([x-2, 34, x+1, 34], fill=(255, 220, 177), width=2)
                draw.line([x+3, 34, x+6, 34], fill=(255, 220, 177), width=2)
            
            # Teacher
            teacher_x = 40
            draw.ellipse([teacher_x, 20, teacher_x+6, 26], fill=(255, 220, 177))
            draw.rectangle([teacher_x+1, 26, teacher_x+5, 36], fill=(231, 76, 60))
            draw.line([teacher_x-2, 28, teacher_x+1, 28], fill=(255, 220, 177), width=2)
            draw.line([teacher_x+5, 28, teacher_x+8, 28], fill=(255, 220, 177), width=2)
            
            img.save('temp_icon.ico', format='ICO', sizes=[(64, 64), (32, 32), (16, 16)])
            self.root.iconbitmap('temp_icon.ico')
            print("✅ Simple icon created and applied")
            
        except Exception as e:
            print(f"❌ Could not create icon: {e}")
        
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#2c3e50')
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        
        # Configure buttons
        style.configure('Action.TButton', font=('Arial', 11, 'bold'))
        style.configure('Danger.TButton', font=('Arial', 11, 'bold'))
        style.configure('Success.TButton', font=('Arial', 11, 'bold'))
    
    def check_model_status(self):
        """Check if AI models are loaded and available"""
        try:
            # Update status to checking
            if hasattr(self, 'model_status_frame'):
                for widget in self.model_status_frame.winfo_children():
                    if isinstance(widget, ttk.Label) and "Checking" in widget.cget("text"):
                        widget.config(text="🔄 Checking models...")
            
            # Check if model files exist first
            model_files = {
                'detection': 'yolov8s.pt',
                'pose': 'yolov8n-pose.pt', 
                'face': 'yolov12s-face.pt'
            }
            
            # Get the directory where the executable is running from
            if sys is not None:
                try:
                    if getattr(sys, 'frozen', False):
                        # Running as PyInstaller executable
                        base_path = sys._MEIPASS
                    else:
                        # Running as script
                        base_path = os.path.dirname(os.path.abspath(__file__))
                except (NameError, AttributeError):
                    # Fallback if sys is not available
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    print("⚠️ sys module not available, using script directory")
            else:
                # sys is None, use script directory
                base_path = os.path.dirname(os.path.abspath(__file__))
                print("⚠️ sys module not available, using script directory")
            
            # Look for AI_Model_Weights in the executable directory
            weights_dir = os.path.join(base_path, "AI_Model_Weights", "AI_Model_Weights")
            
            # If not found in executable, try current directory
            if not os.path.exists(weights_dir):
                weights_dir = os.path.join(base_path, "AI_Model_Weights")
            
            # If still not found, try relative to script location
            if not os.path.exists(weights_dir):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                weights_dir = os.path.join(script_dir, "AI_Model_Weights", "AI_Model_Weights")
            
            # Check which model files exist
            self.model_status = {}
            for model_type, filename in model_files.items():
                model_path = os.path.join(weights_dir, filename)
                self.model_status[model_type] = os.path.exists(model_path)
                if self.model_status[model_type]:
                    print(f"✅ Found {model_type}: {model_path}")
                else:
                    print(f"❌ Missing {model_type}: {model_path}")
            
            # Update UI
            self.update_model_status_display()
            
            print(f"🔍 Model Status: {self.model_status}")
            
            # Log status to GUI
            loaded_count = sum(self.model_status.values())
            total_count = len(self.model_status)
            self.log_message(f"🤖 Model check complete: {loaded_count}/{total_count} models found")
            
        except Exception as e:
            print(f"❌ Error checking model status: {e}")
            self.model_status = {"detection": False, "pose": False, "face": False}
            self.update_model_status_display()
            self.log_message(f"❌ Model check failed: {str(e)}")
    
    def update_model_status_display(self):
        """Update the model status display in the GUI"""
        try:
            if hasattr(self, 'model_status_frame'):
                # Clear existing status
                for widget in self.model_status_frame.winfo_children():
                    widget.destroy()
                
                # Create new status display
                ttk.Label(self.model_status_frame, text="AI Model Status:", 
                         font=('Arial', 10, 'bold')).pack(anchor='w')
                
                for model_name, status in self.model_status.items():
                    status_text = "✅ Loaded" if status else "❌ Not Available"
                    color = "green" if status else "red"
                    
                    frame = ttk.Frame(self.model_status_frame)
                    frame.pack(fill='x', pady=2)
                    
                    ttk.Label(frame, text=f"{model_name.title()}:", 
                             font=('Arial', 9)).pack(side='left')
                    ttk.Label(frame, text=status_text, 
                             foreground=color, font=('Arial', 9, 'bold')).pack(side='left', padx=(10, 0))
                
                # Overall status
                all_loaded = all(self.model_status.values())
                overall_status = "✅ All Models Ready" if all_loaded else "⚠️ Some Models Missing"
                overall_color = "green" if all_loaded else "orange"
                
                ttk.Label(self.model_status_frame, text=f"Overall: {overall_status}", 
                         foreground=overall_color, font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 0))
                
        except Exception as e:
            print(f"❌ Error updating model status display: {e}")
        
    def create_widgets(self):
        """Create all GUI widgets with proper layout"""
        # Create scrollable main container
        canvas = tk.Canvas(self.root, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main container with padding
        main_frame = ttk.Frame(scrollable_frame, padding="25")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title with icon
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Add icon to title
        try:
            if os.path.exists('classroom_icon.png'):
                icon_img = Image.open('classroom_icon.png')
                icon_img = icon_img.resize((50, 50), Image.Resampling.LANCZOS)
                self.icon_photo = ImageTk.PhotoImage(icon_img)
                icon_label = ttk.Label(title_frame, image=self.icon_photo)
                icon_label.pack(side=tk.LEFT, padx=(0, 15))
        except:
            pass
        
        title_label = ttk.Label(title_frame, text="🎓 Classroom Video Analyzer", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Create sections with proper spacing
        self.create_video_section(main_frame)
        self.create_options_section(main_frame)
        self.create_analysis_section(main_frame)
        self.create_progress_section(main_frame)
        self.create_results_section(main_frame)
        self.create_status_bar(main_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_video_section(self, parent):
        """Create video selection section"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text="📹 Video Selection", padding="15")
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Video file selection
        file_frame = ttk.Frame(section_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Video File:").pack(side=tk.LEFT, padx=(0, 10))
        video_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=60)
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_video_file).pack(side=tk.RIGHT)
        
        # Video info
        self.video_info_label = ttk.Label(section_frame, text="No video selected", style='Info.TLabel')
        self.video_info_label.pack(anchor=tk.W)
        
        # Model Status Section
        self.create_model_status_section(section_frame)
        
    def create_model_status_section(self, parent):
        """Create model status display section"""
        # Model status frame
        self.model_status_frame = ttk.LabelFrame(parent, text="🤖 AI Model Status", padding="10")
        self.model_status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Header with refresh button
        header_frame = ttk.Frame(self.model_status_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="AI Model Status:", 
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(header_frame, text="🔄 Refresh", 
                  command=self.check_model_status).pack(side=tk.RIGHT)
        
        # Initial status
        ttk.Label(self.model_status_frame, text="Checking models...", 
                 font=('Arial', 10)).pack(anchor='w')
        
    def create_options_section(self, parent):
        """Create analysis options section"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text="⚙️ Analysis Options", padding="15")
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Output directory
        output_frame = ttk.Frame(section_frame)
        output_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(0, 10))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, width=60)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.RIGHT)
        
        # Options in two columns
        options_frame = ttk.Frame(section_frame)
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Left column - Processing options
        left_frame = ttk.LabelFrame(options_frame, text="Processing Options", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Checkbutton(left_frame, text="Headless Mode (Faster Processing)", 
                       variable=self.headless_mode).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(left_frame, text="Save Sample Frames", 
                       variable=self.save_frames).pack(anchor=tk.W, pady=2)
        
        # Right column - Display options
        right_frame = ttk.LabelFrame(options_frame, text="Display Options", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Checkbutton(right_frame, text="Show Footage During Analysis", 
                       variable=self.show_footage).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(right_frame, text="Real-time Display Updates", 
                       variable=self.real_time_display).pack(anchor=tk.W, pady=2)
        
        # Tips
        tips_frame = ttk.Frame(section_frame)
        tips_frame.pack(fill=tk.X)
        
        ttk.Label(tips_frame, text="💡 Tip: Uncheck 'Show Footage' for faster processing", 
                 style='Info.TLabel').pack(anchor=tk.W, pady=1)
        ttk.Label(tips_frame, text="💡 Tip: 'Real-time Display' shows live analysis overlay", 
                 style='Info.TLabel').pack(anchor=tk.W, pady=1)
        
    def create_analysis_section(self, parent):
        """Create analysis control section"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text="🚀 Analysis Control", padding="20")
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Control buttons with better spacing
        button_frame = ttk.Frame(section_frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        # First row - Main analysis buttons
        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(row1_frame, text="▶️ Start Analysis", 
                                     command=self.start_analysis, style='Action.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(row1_frame, text="⏹️ Stop Analysis", 
                                    command=self.stop_analysis, style='Danger.TButton', state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.results_button = ttk.Button(row1_frame, text="📁 Open Results", 
                                       command=self.open_results, state='disabled')
        self.results_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.preview_button = ttk.Button(row1_frame, text="👁️ Preview Video", 
                                       command=self.preview_video, state='disabled')
        self.preview_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row1_frame, text="❓ Help", command=self.show_help).pack(side=tk.RIGHT)
        
        # Second row - Advanced analysis features
        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(row2_frame, text="Advanced Features:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.view_analysis_button = ttk.Button(row2_frame, text="📊 View Analysis", 
                                             command=self.open_analysis_viewer, state='disabled')
        self.view_analysis_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.match_faces_button = ttk.Button(row2_frame, text="👥 Match Faces", 
                                           command=self.match_video_faces, state='disabled')
        self.match_faces_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.classify_lecture_button = ttk.Button(row2_frame, text="🎓 Classify Lecture", 
                                                command=self.classify_lecture_type, state='disabled')
        self.classify_lecture_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.attendance_button = ttk.Button(row2_frame, text="📋 Attendance Report", 
                                           command=self.generate_attendance_report, state='disabled')
        self.attendance_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Third row - Automated processing features (more prominent)
        row3_frame = ttk.LabelFrame(button_frame, text="🚀 Automated Processing", padding="10")
        row3_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Create a sub-frame for buttons
        auto_buttons_frame = ttk.Frame(row3_frame)
        auto_buttons_frame.pack(fill=tk.X)
        
        self.auto_process_button = ttk.Button(auto_buttons_frame, text="🚀 Auto Process Videos", 
                                            command=self.start_auto_processing, style='Action.TButton')
        self.auto_process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_auto_button = ttk.Button(auto_buttons_frame, text="⏹️ Stop Processing", 
                                         command=self.stop_auto_processing, style='Danger.TButton', state='disabled')
        self.stop_auto_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.history_button = ttk.Button(auto_buttons_frame, text="📚 View History", 
                                       command=self.show_historical_reports)
        self.history_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.folder_button = ttk.Button(auto_buttons_frame, text="📁 Open Video Folder", 
                                      command=self.open_video_folder)
        self.folder_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Firebase sync buttons
        self.sync_button = ttk.Button(auto_buttons_frame, text="☁️ Sync to Firebase", 
                                    command=self.sync_to_firebase)
        self.sync_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.sync_all_button = ttk.Button(auto_buttons_frame, text="📊 Sync All Data", 
                                        command=self.sync_all_historical_data)
        self.sync_all_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add options frame for automated processing
        auto_options_frame = ttk.Frame(row3_frame)
        auto_options_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Show footage option for automated processing
        self.auto_show_footage = tk.BooleanVar(value=False)
        self.auto_show_footage_cb = ttk.Checkbutton(auto_options_frame, text="👁️ Show Footage During Auto Processing", 
                                                   variable=self.auto_show_footage)
        self.auto_show_footage_cb.pack(side=tk.LEFT, padx=(0, 20))
        
        # Real-time display option for automated processing
        self.auto_realtime_display = tk.BooleanVar(value=True)
        self.auto_realtime_display_cb = ttk.Checkbutton(auto_options_frame, text="⚡ Real-time Display Updates", 
                                                       variable=self.auto_realtime_display)
        self.auto_realtime_display_cb.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add a separator and instructions
        ttk.Separator(row3_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 5))
        
        instructions_text = "💡 Add videos to video_processing/YYYY-MM-DD/input/ folder, then click 'Auto Process Videos'"
        ttk.Label(row3_frame, text=instructions_text, style='Info.TLabel', wraplength=600).pack(anchor=tk.W)
        
    def create_progress_section(self, parent):
        """Create progress tracking section"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text="📊 Progress", padding="15")
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(section_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Progress label
        self.progress_label = ttk.Label(section_frame, text="Ready to start analysis", style='Info.TLabel')
        self.progress_label.pack(anchor=tk.W)
        
    def create_results_section(self, parent):
        """Create results display section"""
        # Section frame
        section_frame = ttk.LabelFrame(parent, text="📋 Analysis Log", padding="15")
        section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(section_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Info.TLabel')
        status_label.pack(side=tk.LEFT)
        
    def browse_video_file(self):
        """Browse for video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path.set(filename)
            self.update_video_info(filename)
            self.preview_button.config(state='normal')
            
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            
    def update_video_info(self, video_path):
        """Update video information display"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                info_text = f"📹 {os.path.basename(video_path)} | {width}x{height} | {fps:.1f} FPS | {duration:.1f}s | {frame_count} frames"
                self.video_info_label.config(text=info_text, style='Success.TLabel')
                cap.release()
            else:
                self.video_info_label.config(text="❌ Invalid video file", style='Error.TLabel')
        except Exception as e:
            self.video_info_label.config(text=f"❌ Error reading video: {str(e)}", style='Error.TLabel')
            
    def preview_video(self):
        """Preview the selected video"""
        if not self.video_path.get():
            messagebox.showwarning("Warning", "Please select a video file first!")
            return
            
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open video file!")
                return
            
            # Create preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Video Preview")
            preview_window.geometry("800x600")
            
            # Video display
            video_frame = ttk.Frame(preview_window, padding="10")
            video_frame.pack(fill=tk.BOTH, expand=True)
            
            # Get first frame
            ret, frame = cap.read()
            if ret:
                # Resize frame for display
                height, width = frame.shape[:2]
                display_width = min(760, width)
                display_height = int(height * (display_width / width))
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(frame_pil)
                
                # Display frame
                label = ttk.Label(video_frame, image=photo)
                label.image = photo  # Keep a reference
                label.pack()
                
                # Video info
                info_text = f"First frame: {width}x{height} | Duration: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)):.1f}s"
                ttk.Label(video_frame, text=info_text, style='Info.TLabel').pack(pady=(10, 0))
            
            cap.release()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview video: {str(e)}")
            
    def start_analysis(self):
        """Start video analysis in a separate thread"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first!")
            return
            
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Video file does not exist!")
            return
            
        # Update UI
        self.is_analyzing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.results_button.config(state='disabled')
        self.preview_button.config(state='disabled')
        self.view_analysis_button.config(state='disabled')
        self.match_faces_button.config(state='disabled')
        self.classify_lecture_button.config(state='disabled')
        self.attendance_button.config(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text="Starting analysis...")
        self.status_var.set("Analyzing...")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 Starting classroom video analysis...")
        self.log_message(f"📹 Video: {os.path.basename(self.video_path.get())}")
        self.log_message(f"📁 Output: {self.output_dir.get()}")
        self.log_message(f"⚙️ Headless mode: {'Yes' if self.headless_mode.get() else 'No'}")
        self.log_message(f"👁️ Show footage: {'Yes' if self.show_footage.get() else 'No'}")
        self.log_message(f"🔄 Real-time display: {'Yes' if self.real_time_display.get() else 'No'}")
        self.log_message("=" * 50)
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        self.analysis_thread.start()
        
    def run_analysis(self):
        """Run the actual analysis"""
        try:
            # Check if models are loaded before starting
            if not any(self.model_status.values()):
                self.log_message("❌ No AI models are loaded! Cannot start analysis.")
                self.log_message("Please check the model status above and ensure model files are available.")
                self.root.after(0, lambda: self.analysis_error("No AI models loaded"))
                return
            
            self.log_message("🔍 Creating analyzer instance...")
            self.update_progress(5, "Initializing analyzer...")
            
            # Register video in database
            video_metadata = self.get_video_metadata(self.video_path.get())
            video_id = self.data_manager.register_video(self.video_path.get(), video_metadata)
            
            if video_id is None:
                raise Exception("Failed to register video in database")
            
            # Create analysis session
            self.current_session_id = self.data_manager.create_analysis_session(
                video_id=video_id,
                output_dir=self.output_dir.get(),
                headless_mode=self.headless_mode.get(),
                models_used=[k for k, v in self.model_status.items() if v]
            )
            
            # Create analyzer
            self.analyzer = RealtimeClassroomAnalyzer(
                video_path=self.video_path.get(),
                output_dir=self.output_dir.get(),
                headless_mode=self.headless_mode.get()
            )
            
            # Check if analyzer was created successfully
            if not self.analyzer:
                raise Exception("Failed to create analyzer instance")
            
            # Log model status
            loaded_models = [k for k, v in self.model_status.items() if v]
            self.log_message(f"✅ Loaded models: {', '.join(loaded_models)}")
            
            # Run analysis
            self.log_message("🔍 Loading AI models...")
            self.update_progress(10, "Loading models...")
            
            self.log_message("🎬 Starting video processing...")
            self.update_progress(20, "Processing video...")
            
            # Determine display mode based on user selection
            display_mode = self.show_footage.get() and not self.headless_mode.get()
            self.log_message(f"📺 Display mode: {'Enabled' if display_mode else 'Disabled'}")
            
            # Run the analysis
            results = self.analyzer.analyze_video_realtime(
                display=display_mode,
                save_frames=self.save_frames.get()
            )
            
            # Analysis complete
            self.update_progress(100, "Analysis complete!")
            self.log_message("✅ Analysis completed successfully!")
            self.log_message(f"📊 Processed {len(results)} frames")
            self.log_message(f"📁 Results saved to: {self.analyzer.output_dir}")
            
            # Save analysis results to database
            self.save_analysis_to_database(len(results))
            
            # Update UI
            self.root.after(0, self.analysis_complete)
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            self.log_message(error_msg)
            self.log_message(f"🔍 Error details: {type(e).__name__}")
            
            # Add specific handling for sys import errors
            if "sys" in str(e).lower() and "not defined" in str(e).lower():
                self.log_message("🔧 This appears to be a sys import issue in the executable")
                self.log_message("💡 Try running the Python script directly instead of the executable")
            
            try:
                import traceback
                self.log_message(f"📍 Traceback: {traceback.format_exc()}")
            except:
                self.log_message("📍 Could not get detailed traceback")
            
            self.root.after(0, lambda: self.analysis_error(str(e)))
            
    def stop_analysis(self):
        """Stop the analysis"""
        self.is_analyzing = False
        self.log_message("⏹️ Analysis stopped by user")
        self.root.after(0, self.analysis_stopped)
        
    def analysis_complete(self):
        """Handle analysis completion"""
        self.is_analyzing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.results_button.config(state='normal')
        self.preview_button.config(state='normal')
        self.view_analysis_button.config(state='normal')
        self.match_faces_button.config(state='normal')
        self.classify_lecture_button.config(state='normal')
        self.attendance_button.config(state='normal')
        self.status_var.set("Analysis complete")
        
    def analysis_stopped(self):
        """Handle analysis stop"""
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.preview_button.config(state='normal')
        self.view_analysis_button.config(state='normal')
        self.match_faces_button.config(state='normal')
        self.classify_lecture_button.config(state='normal')
        self.attendance_button.config(state='normal')
        self.status_var.set("Analysis stopped")
        
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.is_analyzing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.preview_button.config(state='normal')
        self.view_analysis_button.config(state='normal')
        self.match_faces_button.config(state='normal')
        self.classify_lecture_button.config(state='normal')
        self.attendance_button.config(state='normal')
        self.status_var.set("Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        
    def update_progress(self, value, text):
        """Update progress bar and label"""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=text))
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.log_text.see(tk.END))
        
    def open_results(self):
        """Open results directory"""
        if self.analyzer and os.path.exists(self.analyzer.output_dir):
            if sys.platform == "win32":
                os.startfile(self.analyzer.output_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", self.analyzer.output_dir])
            else:
                subprocess.run(["xdg-open", self.analyzer.output_dir])
        else:
            messagebox.showwarning("Warning", "No results directory found!")
            
    def show_help(self):
        """Show help dialog"""
        help_text = """
🎓 Classroom Video Analyzer - Help

This application analyzes classroom videos to detect:
• Student activities (writing, raising hand, listening, distracted)
• Face detection and tracking
• Engagement levels
• Classroom zones (front, middle, back)

How to use:
1. Select a video file using the Browse button
2. Choose output directory (optional)
3. Select analysis options:
   - Headless Mode: Faster processing without display
   - Save Sample Frames: Save sample frames during analysis
   - Show Footage: Display video during analysis
   - Real-time Display: Show live analysis overlay
4. Click "Start Analysis" to begin
5. Monitor progress in the log area
6. Click "Open Results" to view analysis results
7. Use "Preview Video" to see the first frame

Supported video formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM

Requirements:
• AI model weights must be in the correct directory
• Sufficient disk space for output files
• Good CPU/GPU for real-time processing

For support, check the documentation or contact the developer.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.configure(bg='#f0f0f0')
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=20, pady=20)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
    
    def open_analysis_viewer(self):
        """Open the advanced analysis viewer"""
        if not self.analyzer or not os.path.exists(self.analyzer.output_dir):
            messagebox.showwarning("Warning", "No analysis results found! Please run analysis first.")
            return
        
        try:
            # Create analysis viewer window
            viewer_window = tk.Toplevel(self.root)
            viewer_window.title("📊 Advanced Analysis Viewer")
            viewer_window.geometry("1400x900")
            
            # Create analysis viewer
            self.analysis_viewer = AnalysisViewer(
                viewer_window, 
                analysis_dir=self.analyzer.output_dir
            )
            
            self.log_message("📊 Opened advanced analysis viewer")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open analysis viewer: {str(e)}")
            self.log_message(f"❌ Error opening analysis viewer: {str(e)}")
    
    def match_video_faces(self):
        """Match faces from current video with existing database"""
        if not self.analyzer or not os.path.exists(self.analyzer.output_dir):
            messagebox.showwarning("Warning", "No analysis results found! Please run analysis first or use automated processing.")
            return
        
        try:
            self.log_message("👥 Starting face matching process...")
            self.update_progress(10, "Loading face database...")
            
            # Load analysis data
            comprehensive_file = os.path.join(self.analyzer.output_dir, "comprehensive_analysis_report.json")
            if not os.path.exists(comprehensive_file):
                messagebox.showerror("Error", "Comprehensive analysis file not found!")
                return
            
            with open(comprehensive_file, 'r') as f:
                analysis_data = json.load(f)
            
            self.update_progress(30, "Processing faces...")
            
            # Process faces with enhanced matcher
            # Get video ID from the analysis data or use a default
            video_id = "unknown"
            if hasattr(self, 'video_path') and self.video_path.get():
                video_id = os.path.basename(self.video_path.get())
            elif 'video_path' in analysis_data:
                video_id = os.path.basename(analysis_data['video_path'])
            
            # Debug: Check analysis_data type and content
            print(f"🔍 Analysis data type: {type(analysis_data)}")
            print(f"🔍 Analysis data keys: {list(analysis_data.keys()) if isinstance(analysis_data, dict) else 'Not a dict'}")
            
            # Check faces data specifically
            if 'faces' in analysis_data:
                faces_data = analysis_data['faces']
                print(f"🔍 Faces data type: {type(faces_data)}")
                if isinstance(faces_data, dict):
                    print(f"🔍 Faces dict keys: {list(faces_data.keys())[:5]}...")  # Show first 5 keys
                elif isinstance(faces_data, list):
                    print(f"🔍 Faces list length: {len(faces_data)}")
                else:
                    print(f"🔍 Faces data content: {faces_data}")
            else:
                print("🔍 No 'faces' key found in analysis data")
            
            matching_results = self.enhanced_face_matcher.process_video_faces(analysis_data, video_id)
            
            self.update_progress(90, "Generating report...")
            
            # Generate attendance report
            attendance_summary = self.enhanced_face_matcher.get_attendance_summary()
            
            # Generate engagement mapping by person ID
            engagement_summary = self.enhanced_face_matcher.get_person_engagement_summary(analysis_data, video_id)
            
            self.update_progress(100, "Face matching complete!")
            
            # Debug: Print matching results structure
            print(f"📊 Matching results type: {type(matching_results)}")
            print(f"📊 Matching results: {matching_results}")
            print(f"📊 Engagement summary: {engagement_summary}")
            
            # Show results
            self.show_face_matching_results(matching_results, attendance_summary, engagement_summary)
            
            # Safely access matching results - matching_results is a boolean
            if matching_results:
                self.log_message("✅ Face matching completed successfully!")
            else:
                self.log_message("⚠️ Face matching completed with warnings")
            
        except Exception as e:
            messagebox.showerror("Error", f"Face matching failed: {str(e)}")
            self.log_message(f"❌ Face matching error: {str(e)}")
    
    def show_face_matching_results(self, matching_results, attendance_summary, engagement_summary=None):
        """Show face matching results in a dialog"""
        results_window = tk.Toplevel(self.root)
        results_window.title("👥 Face Matching Results")
        results_window.geometry("800x600")
        
        # Create scrollable frame
        canvas = tk.Canvas(results_window)
        scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Results content
        ttk.Label(scrollable_frame, text="Face Matching Results", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Summary
        summary_frame = ttk.LabelFrame(scrollable_frame, text="Summary", padding="10")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Safely access matching results
        if isinstance(matching_results, dict):
            matched = matching_results.get('matched', 0)
            new = matching_results.get('new', 0)
            total_processed = matching_results.get('total_processed', 0)
        else:
            matched = 0
            new = 0
            total_processed = 0
        
        summary_text = f"""
        Total Faces Processed: {total_processed}
        Matched Faces: {matched}
        New Faces: {new}
        """
        ttk.Label(summary_frame, text=summary_text, font=('Arial', 10)).pack()
        
        # Engagement summary
        if engagement_summary:
            engagement_frame = ttk.LabelFrame(scrollable_frame, text="Engagement Analysis by Person ID", padding="10")
            engagement_frame.pack(fill=tk.X, padx=10, pady=5)
            
            total_persons = engagement_summary.get('total_persons', 0)
            avg_engagement = engagement_summary.get('average_engagement', 0.0)
            most_engaged = engagement_summary.get('most_engaged_person', {})
            least_engaged = engagement_summary.get('least_engaged_person', {})
            
            engagement_text = f"""
            Total Unique Persons: {total_persons}
            Average Engagement Score: {avg_engagement:.3f}
            Most Engaged Person: {most_engaged.get('person_id', 'N/A')} (Score: {most_engaged.get('engagement_score', 0):.3f})
            Least Engaged Person: {least_engaged.get('person_id', 'N/A')} (Score: {least_engaged.get('engagement_score', 0):.3f})
            """
            ttk.Label(engagement_frame, text=engagement_text, font=('Arial', 10)).pack()
            
            # Person details
            person_details = engagement_summary.get('person_details', {})
            if person_details:
                details_text = "\nPerson Details:\n"
                for person_id, details in person_details.items():
                    details_text += f"Person {person_id}: Engagement={details.get('engagement_score', 0):.3f}, Frames={details.get('total_frames', 0)}, Videos={len(details.get('videos', []))}\n"
                
                ttk.Label(engagement_frame, text=details_text, font=('Arial', 9)).pack(anchor=tk.W)
        
        # Attendance summary
        if attendance_summary:
            attendance_frame = ttk.LabelFrame(scrollable_frame, text="Attendance Summary", padding="10")
            attendance_frame.pack(fill=tk.X, padx=10, pady=5)
            
            total_people = len(attendance_summary)
            total_appearances = sum(person.get('total_appearances', 0) for person in attendance_summary.values())
            
            attendance_text = f"""
            Total People in Database: {total_people}
            Total Appearances: {total_appearances}
            Average Appearances per Person: {total_appearances/max(total_people, 1):.1f}
            """
            ttk.Label(attendance_frame, text=attendance_text, font=('Arial', 10)).pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def start_auto_processing(self):
        """Start automated video processing"""
        try:
            self.log_message("🚀 Starting automated video processing...")
            
            # Check if there are videos to process
            today = self.auto_processor.get_today_folder()
            videos = self.auto_processor.get_videos_for_date(today.name)
            
            self.log_message(f"📁 Checking folder: {today}/input")
            self.log_message(f"🎬 Found {len(videos)} videos to process")
            
            if not videos:
                messagebox.showinfo("Info", f"No videos found in {today}/input folder.\n\nPlease add videos to:\n{today}/input/")
                self.log_message("⚠️ No videos found in input folder")
                return
            
            # Confirm processing
            result = messagebox.askyesno(
                "Confirm Processing", 
                f"Found {len(videos)} video(s) to process:\n\n" + 
                "\n".join([f"• {v['name']}" for v in videos[:5]]) + 
                (f"\n... and {len(videos)-5} more" if len(videos) > 5 else "") +
                "\n\nStart processing?"
            )
            
            if not result:
                return
            
            # Start processing in background thread
            self.auto_process_button.config(state='disabled')
            self.stop_auto_button.config(state='normal')
            
            def process_thread():
                try:
                    self.log_message("🚀 Starting automated video processing...")
                    
                    # Get processing options from GUI
                    show_footage = self.auto_show_footage.get()
                    realtime_display = self.auto_realtime_display.get()
                    
                    self.log_message(f"📊 Processing options: Show footage={show_footage}, Real-time={realtime_display}")
                    
                    # Pass options to the processor - process all available dates
                    self.auto_processor.process_all_available_dates(
                        progress_callback=self.update_auto_progress,
                        show_footage=show_footage,
                        realtime_display=realtime_display
                    )
                    self.log_message("✅ Automated processing completed!")
                    
                    # Enable analysis buttons after processing
                    self.enable_analysis_buttons()
                    
                except Exception as e:
                    self.log_message(f"❌ Automated processing error: {e}")
                finally:
                    self.auto_process_button.config(state='normal')
                    self.stop_auto_button.config(state='disabled')
            
            threading.Thread(target=process_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start automated processing: {e}")
            self.auto_process_button.config(state='normal')
            self.stop_auto_button.config(state='disabled')
    
    def stop_auto_processing(self):
        """Stop automated video processing"""
        self.auto_processor.stop_processing()
        self.auto_process_button.config(state='normal')
        self.stop_auto_button.config(state='disabled')
        self.log_message("⏹️ Automated processing stopped")
    
    def update_auto_progress(self, progress, message):
        """Update progress for automated processing"""
        self.update_progress(progress, message)
        self.log_message(f"📊 {message} ({progress:.1f}%)")
    
    def show_historical_reports(self):
        """Show historical analysis reports"""
        try:
            reports = self.auto_processor.get_historical_reports()
            
            if not reports:
                messagebox.showinfo("No Reports", "No historical reports found.")
                return
            
            # Create reports window
            reports_window = tk.Toplevel(self.root)
            reports_window.title("📚 Historical Analysis Reports")
            reports_window.geometry("1000x700")
            
            # Create scrollable frame
            canvas = tk.Canvas(reports_window)
            scrollbar = ttk.Scrollbar(reports_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Title
            ttk.Label(scrollable_frame, text="Historical Analysis Reports", 
                     font=('Arial', 16, 'bold')).pack(pady=10)
            
            # Summary
            summary = self.auto_processor.get_report_summary()
            summary_frame = ttk.LabelFrame(scrollable_frame, text="Summary", padding="10")
            summary_frame.pack(fill=tk.X, padx=10, pady=5)
            
            summary_text = f"""
            Total Videos: {summary['total_videos']}
            Completed: {summary['completed']}
            Failed: {summary['failed']}
            Total Faces: {summary['total_faces']}
            Total Students: {summary['total_students']}
            Dates: {', '.join(summary['dates'])}
            """
            ttk.Label(summary_frame, text=summary_text, font=('Arial', 10)).pack()
            
            # Reports list
            reports_frame = ttk.LabelFrame(scrollable_frame, text="Reports", padding="10")
            reports_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            for i, report in enumerate(reports):
                report_frame = ttk.Frame(reports_frame)
                report_frame.pack(fill=tk.X, pady=2)
                
                # Report info
                status_icon = "✅" if report.get('status') == 'completed' else "❌"
                info_text = f"{status_icon} {report.get('video_name', 'Unknown')} - {report.get('date', 'Unknown')}"
                
                ttk.Label(report_frame, text=info_text, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
                
                # View button
                view_button = ttk.Button(report_frame, text="View", 
                                       command=lambda r=report: self.view_report(r))
                view_button.pack(side=tk.RIGHT, padx=5)
                
                # Open folder button
                folder_button = ttk.Button(report_frame, text="Open Folder", 
                                         command=lambda r=report: self.open_report_folder(r))
                folder_button.pack(side=tk.RIGHT, padx=5)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load historical reports: {e}")
    
    def view_report(self, report):
        """View a specific report"""
        try:
            report_path = report.get('report_path')
            if not report_path or not os.path.exists(report_path):
                messagebox.showerror("Error", "Report file not found!")
                return
            
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            # Create report viewer window
            viewer_window = tk.Toplevel(self.root)
            viewer_window.title(f"Report: {report.get('video_name', 'Unknown')}")
            viewer_window.geometry("800x600")
            
            # Create scrollable frame
            canvas = tk.Canvas(viewer_window)
            scrollbar = ttk.Scrollbar(viewer_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Report content
            ttk.Label(scrollable_frame, text=f"Report: {report.get('video_name', 'Unknown')}", 
                     font=('Arial', 14, 'bold')).pack(pady=10)
            
            # Report details
            details_frame = ttk.LabelFrame(scrollable_frame, text="Report Details", padding="10")
            details_frame.pack(fill=tk.X, padx=10, pady=5)
            
            details_text = f"""
            Video: {report_data.get('video_name', 'Unknown')}
            Status: {report_data.get('status', 'Unknown')}
            Processed: {report_data.get('processed_at', 'Unknown')}
            Face Count: {report_data.get('face_count', 0)}
            Student Count: {report_data.get('student_count', 0)}
            Processing Time: {report_data.get('processing_time', 0)}s
            """
            ttk.Label(details_frame, text=details_text, font=('Arial', 10)).pack()
            
            # Analysis summary
            if 'analysis_summary' in report_data:
                analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Summary", padding="10")
                analysis_frame.pack(fill=tk.X, padx=10, pady=5)
                
                analysis_text = json.dumps(report_data['analysis_summary'], indent=2)
                text_widget = tk.Text(analysis_frame, height=10, wrap=tk.WORD)
                text_widget.insert(tk.END, analysis_text)
                text_widget.config(state=tk.DISABLED)
                text_widget.pack(fill=tk.BOTH, expand=True)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view report: {e}")
    
    def open_report_folder(self, report):
        """Open the folder containing the report"""
        try:
            report_path = report.get('report_path')
            if report_path:
                folder_path = os.path.dirname(report_path)
                os.startfile(folder_path)
            else:
                messagebox.showerror("Error", "Report path not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {e}")
    
    def open_video_folder(self):
        """Open the video processing folder"""
        try:
            today_folder = self.auto_processor.get_today_folder()
            input_folder = today_folder / "input"
            input_folder.mkdir(exist_ok=True)
            os.startfile(str(input_folder))
            self.log_message(f"📁 Opened video folder: {input_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video folder: {e}")
    
    def enable_analysis_buttons(self):
        """Enable analysis buttons after automated processing"""
        try:
            # Find the most recent analysis results
            today = self.auto_processor.get_today_folder()
            analysis_folder = self.auto_processor.analysis_folder / today.name
            
            if analysis_folder.exists():
                # Look for the most recent analysis
                analysis_dirs = [d for d in analysis_folder.iterdir() if d.is_dir()]
                if analysis_dirs:
                    # Get the most recent analysis directory
                    latest_analysis = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
                    comprehensive_file = latest_analysis / "comprehensive_analysis_report.json"
                    
                    if comprehensive_file.exists():
                        # Set the analyzer output directory
                        self.analyzer = type('MockAnalyzer', (), {
                            'output_dir': str(latest_analysis)
                        })()
                        
                        # Enable analysis buttons
                        self.view_analysis_button.config(state='normal')
                        self.match_faces_button.config(state='normal')
                        self.classify_lecture_button.config(state='normal')
                        self.attendance_button.config(state='normal')
                        
                        self.log_message("✅ Analysis buttons enabled - results available!")
                        return True
            
            # Also check the regular analysis_results folder
            analysis_results_dir = Path("analysis_results")
            if analysis_results_dir.exists():
                comprehensive_file = analysis_results_dir / "comprehensive_analysis_report.json"
                if comprehensive_file.exists():
                    # Set the analyzer output directory
                    self.analyzer = type('MockAnalyzer', (), {
                        'output_dir': str(analysis_results_dir)
                    })()
                    
                    # Enable analysis buttons
                    self.view_analysis_button.config(state='normal')
                    self.match_faces_button.config(state='normal')
                    self.classify_lecture_button.config(state='normal')
                    self.attendance_button.config(state='normal')
                    
                    self.log_message("✅ Analysis buttons enabled - results available!")
                    return True
            
            self.log_message("⚠️ No analysis results found to enable buttons")
            return False
            
        except Exception as e:
            self.log_message(f"❌ Error enabling analysis buttons: {e}")
            return False
    
    def classify_lecture_type(self):
        """Classify the lecture type of the current video using vision LLM"""
        # Check if we have analysis results from automated processing
        if not self.analyzer or not os.path.exists(self.analyzer.output_dir):
            messagebox.showwarning("Warning", "No analysis results found! Please run analysis first or use automated processing.")
            return
        
        # For automated processing, we need to find the video file
        video_file = None
        if self.video_path.get() and os.path.exists(self.video_path.get()):
            video_file = self.video_path.get()
        else:
            # Try to find video from automated processing
            today = self.auto_processor.get_today_folder()
            processed_folder = today / "processed"
            if processed_folder.exists():
                video_files = list(processed_folder.glob("*.mp4")) + list(processed_folder.glob("*.avi"))
                if video_files:
                    video_file = str(video_files[0])  # Use the first video found
        
        if not video_file:
            messagebox.showerror("Error", "No video file found! Please select a video or use automated processing.")
            return
        
        try:
            self.log_message("🎓 Starting vision-based lecture classification...")
            self.update_progress(10, "Extracting frame from video...")
            
            # Use vision classifier for frame-based classification
            classification = self.vision_classifier.classify_video_frame(
                video_file, 
                frame_time=0.5  # Extract frame from middle of video
            )
            
            if classification is None:
                messagebox.showerror("Error", "Failed to classify video frame!")
                return
            
            self.update_progress(100, "Classification complete!")
            
            # Show results
            self.show_lecture_classification_results(classification)
            
            self.log_message(f"✅ Lecture classified as: {classification['lecture_type']} (confidence: {classification['confidence']:.3f})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Lecture classification failed: {str(e)}")
            self.log_message(f"❌ Lecture classification error: {str(e)}")
    
    def show_lecture_classification_results(self, classification):
        """Show lecture classification results"""
        results_window = tk.Toplevel(self.root)
        results_window.title("🎓 Lecture Classification Results")
        results_window.geometry("600x500")
        
        # Main content
        content_frame = ttk.Frame(results_window, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(content_frame, text="Lecture Classification Results", 
                 font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # Classification result
        result_frame = ttk.LabelFrame(content_frame, text="Classification", padding="15")
        result_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Type and confidence
        lecture_type = classification.get('lecture_type', classification.get('type', 'unknown'))
        confidence = classification.get('confidence', 0.0)
        method = classification.get('method', 'unknown')
        
        type_label = ttk.Label(result_frame, text=f"Type: {lecture_type.replace('_', ' ').title()}", 
                              font=('Arial', 14, 'bold'))
        type_label.pack(anchor=tk.W, pady=5)
        
        confidence_label = ttk.Label(result_frame, text=f"Confidence: {confidence:.3f}", 
                                   font=('Arial', 12))
        confidence_label.pack(anchor=tk.W, pady=5)
        
        method_label = ttk.Label(result_frame, text=f"Method: {method.replace('_', ' ').title()}", 
                                font=('Arial', 12))
        method_label.pack(anchor=tk.W, pady=5)
        
        # Reasoning
        reasoning_frame = ttk.LabelFrame(content_frame, text="Reasoning", padding="15")
        reasoning_frame.pack(fill=tk.X, pady=(0, 15))
        
        reasoning_text = classification.get('reasoning', 'No reasoning provided')
        ttk.Label(reasoning_frame, text=reasoning_text, font=('Arial', 10), 
                 wraplength=550).pack(anchor=tk.W)
        
        # Lecture type information
        if 'lecture_type_info' in classification:
            info = classification['lecture_type_info']
            info_frame = ttk.LabelFrame(content_frame, text="Lecture Type Information", padding="15")
            info_frame.pack(fill=tk.X, pady=(0, 15))
            
            if 'description' in info:
                ttk.Label(info_frame, text=f"Description: {info['description']}", 
                         font=('Arial', 10), wraplength=550).pack(anchor=tk.W, pady=5)
            
            if 'keywords' in info:
                keywords_text = f"Keywords: {', '.join(info['keywords'])}"
                ttk.Label(info_frame, text=keywords_text, font=('Arial', 10), 
                         wraplength=550).pack(anchor=tk.W, pady=5)
        
        # All scores (if available)
        if 'all_scores' in classification:
            scores_frame = ttk.LabelFrame(content_frame, text="All Classification Scores", padding="15")
            scores_frame.pack(fill=tk.X, pady=(0, 15))
            
            for lecture_type, score in classification['all_scores'].items():
                score_text = f"{lecture_type.replace('_', ' ').title()}: {score:.3f}"
                ttk.Label(scores_frame, text=score_text, font=('Arial', 10)).pack(anchor=tk.W, pady=2)
        
        # Close button
        ttk.Button(content_frame, text="Close", 
                  command=results_window.destroy).pack(pady=(20, 0))
    
    def generate_attendance_report(self):
        """Generate comprehensive attendance report"""
        try:
            self.log_message("📋 Generating attendance report...")
            self.update_progress(10, "Loading attendance data...")
            
            # Generate report using enhanced face matcher
            attendance_summary = self.enhanced_face_matcher.get_attendance_summary()
            
            # Create comprehensive attendance report
            attendance_report = {
                'summary': {
                    'total_persons': len(attendance_summary),
                    'total_videos_processed': len(set([video for person_data in attendance_summary.values() for video in person_data.get('videos', [])])),
                    'total_appearances': sum([person_data.get('total_appearances', 0) for person_data in attendance_summary.values()])
                },
                'persons': attendance_summary,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.update_progress(100, "Report generated!")
            
            # Show report
            self.show_attendance_report(attendance_report)
            
            self.log_message("✅ Attendance report generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate attendance report: {str(e)}")
            self.log_message(f"❌ Attendance report error: {str(e)}")
    
    def show_attendance_report(self, report):
        """Show attendance report in a detailed window"""
        report_window = tk.Toplevel(self.root)
        report_window.title("📋 Comprehensive Attendance Report")
        report_window.geometry("1400x800")
        
        # Create notebook for different report sections
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 Summary & KPIs")
        
        # Create scrollable frame for summary
        summary_canvas = tk.Canvas(summary_frame)
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_canvas.yview)
        summary_scrollable_frame = ttk.Frame(summary_canvas)
        
        summary_scrollable_frame.bind(
            "<Configure>",
            lambda e: summary_canvas.configure(scrollregion=summary_canvas.bbox("all"))
        )
        
        summary_canvas.create_window((0, 0), window=summary_scrollable_frame, anchor="nw")
        summary_canvas.configure(yscrollcommand=summary_scrollbar.set)
        
        # Summary statistics
        summary_stats_frame = ttk.LabelFrame(summary_scrollable_frame, text="📈 Overall Statistics", padding="15")
        summary_stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        total_persons = report['summary']['total_persons']
        total_videos = report['summary']['total_videos_processed']
        total_appearances = report['summary']['total_appearances']
        avg_appearances = total_appearances / max(total_persons, 1)
        
        summary_text = f"""
        👥 Total Students: {total_persons}
        🎬 Videos Processed: {total_videos}
        📊 Total Appearances: {total_appearances}
        📈 Average Appearances per Student: {avg_appearances:.1f}
        📅 Report Generated: {report['generated_at']}
        """
        
        ttk.Label(summary_stats_frame, text=summary_text, font=('Arial', 12)).pack()
        
        # KPI Analysis
        kpi_frame = ttk.LabelFrame(summary_scrollable_frame, text="🎯 Key Performance Indicators", padding="15")
        kpi_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate KPIs
        if total_persons > 0:
            attendance_rate = (total_persons / max(total_videos, 1)) * 100
            engagement_score = min(avg_appearances / 10 * 100, 100)  # Normalize to 0-100
            
            kpi_text = f"""
            📊 Attendance Rate: {attendance_rate:.1f}%
            🎯 Engagement Score: {engagement_score:.1f}/100
            📈 Most Active Student: {max(report['persons'].items(), key=lambda x: x[1].get('total_appearances', 0))[0] if report['persons'] else 'N/A'}
            📉 Least Active Student: {min(report['persons'].items(), key=lambda x: x[1].get('total_appearances', 0))[0] if report['persons'] else 'N/A'}
            """
        else:
            kpi_text = "No attendance data available"
        
        ttk.Label(kpi_frame, text=kpi_text, font=('Arial', 12)).pack()
        
        summary_canvas.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        # Detailed records tab
        records_frame = ttk.Frame(notebook)
        notebook.add(records_frame, text="👥 Student Details")
        
        # Create scrollable frame for records
        canvas = tk.Canvas(records_frame)
        scrollbar = ttk.Scrollbar(records_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add detailed student records
        if report['persons']:
            for person_id, person_data in report['persons'].items():
                person_frame = ttk.LabelFrame(scrollable_frame, text=f"👤 Student: {person_id}", padding="15")
                person_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Student information
                total_appearances = person_data.get('total_appearances', 0)
                videos = person_data.get('videos', [])
                first_seen = person_data.get('first_seen', 'Unknown')
                last_seen = person_data.get('last_seen', 'Unknown')
                
                # Calculate attendance metrics
                unique_videos = len(set(videos)) if videos else 0
                attendance_rate = (unique_videos / max(total_videos, 1)) * 100 if total_videos > 0 else 0
                
                person_text = f"""
                📊 Total Appearances: {total_appearances}
                🎬 Videos Attended: {unique_videos}
                📈 Attendance Rate: {attendance_rate:.1f}%
                📅 First Seen: {first_seen}
                📅 Last Seen: {last_seen}
                🎯 Videos: {', '.join(set(videos)) if videos else 'None'}
                """
                
                ttk.Label(person_frame, text=person_text, font=('Arial', 11)).pack(anchor=tk.W)
                
                # Add face matching details if available
                if hasattr(self.enhanced_face_matcher, 'face_database') and person_id in self.enhanced_face_matcher.face_database:
                    face_data = self.enhanced_face_matcher.face_database[person_id]
                    encodings_count = len(face_data.get('encodings', []))
                    features_count = len(face_data.get('image_features', []))
                    images_count = len(face_data.get('images', []))
                    
                    face_details = f"""
                    🔍 Face Matching Details:
                    • Face Encodings: {encodings_count}
                    • Image Features: {features_count}
                    • Stored Images: {images_count}
                    """
                    
                    ttk.Label(person_frame, text=face_details, font=('Arial', 10), foreground='blue').pack(anchor=tk.W)
        else:
            no_data_frame = ttk.LabelFrame(scrollable_frame, text="No Data", padding="15")
            no_data_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(no_data_frame, text="No attendance data available. Process some videos first!", 
                     font=('Arial', 12), foreground='red').pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Face Matching Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔍 Face Matching Analysis")
        
        # Create scrollable frame for analysis
        analysis_canvas = tk.Canvas(analysis_frame)
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=analysis_canvas.yview)
        analysis_scrollable_frame = ttk.Frame(analysis_canvas)
        
        analysis_scrollable_frame.bind(
            "<Configure>",
            lambda e: analysis_canvas.configure(scrollregion=analysis_canvas.bbox("all"))
        )
        
        analysis_canvas.create_window((0, 0), window=analysis_scrollable_frame, anchor="nw")
        analysis_canvas.configure(yscrollcommand=analysis_scrollbar.set)
        
        # Face matching statistics
        if hasattr(self.enhanced_face_matcher, 'face_database'):
            face_db = self.enhanced_face_matcher.face_database
            total_encodings = sum(len(person_data.get('encodings', [])) for person_data in face_db.values())
            total_features = sum(len(person_data.get('image_features', [])) for person_data in face_db.values())
            total_images = sum(len(person_data.get('images', [])) for person_data in face_db.values())
            
            stats_text = f"""
            📊 Face Database Statistics:
            
            👥 Total Students: {len(face_db)}
            🔍 Total Face Encodings: {total_encodings}
            🖼️ Total Image Features: {total_features}
            📸 Total Stored Images: {total_images}
            
            🎯 Matching Methods Used:
            • Face Recognition: Primary method
            • Image Similarity: Fallback method
            
            📅 Last Updated: {report['generated_at']}
            """
        else:
            stats_text = "Face database not available"
        
        stats_frame = ttk.LabelFrame(analysis_scrollable_frame, text="Database Statistics", padding="15")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(stats_frame, text=stats_text, font=('Arial', 12)).pack()
        
        analysis_canvas.pack(side="left", fill="both", expand=True)
        analysis_scrollbar.pack(side="right", fill="y")
    
    def sync_to_firebase(self):
        """Sync today's data to Firebase"""
        try:
            self.log_message("☁️ Starting Firebase sync...")
            self.update_progress(10, "Preparing data for sync...")
            
            # Sync today's data
            success = self.firebase_sync.sync_daily_data()
            
            if success:
                self.update_progress(100, "Firebase sync completed!")
                self.log_message("✅ Data synced to Firebase successfully!")
                messagebox.showinfo("Success", "Data has been synced to Firebase successfully!")
            else:
                self.log_message("⚠️ Firebase sync failed. Data saved locally as backup.")
                messagebox.showwarning("Warning", "Firebase sync failed. Data has been saved locally as backup.")
            
        except Exception as e:
            self.log_message(f"❌ Firebase sync error: {str(e)}")
            messagebox.showerror("Error", f"Firebase sync failed: {str(e)}")
    
    def auto_detect_videos(self):
        """Automatically detect and process videos in all date folders"""
        try:
            from datetime import date
            print("🔍 Auto-detecting videos in all date folders...")
            
            # Get all available date folders
            available_dates = self.auto_processor.get_available_dates()
            
            if not available_dates:
                print("📁 No date folders found")
                return
            
            print(f"📅 Found {len(available_dates)} date folder(s): {available_dates}")
            
            # Check each date folder for unprocessed videos
            total_videos_to_process = 0
            videos_by_date = {}
            
            for date_str in available_dates:
                videos = self.auto_processor.get_videos_for_date(date_str)
                processed_videos = self.auto_processor.get_processed_videos(date_str)
                processed_names = {p['video_name'] for p in processed_videos}
                
                # Filter out already processed videos
                unprocessed_videos = [v for v in videos if v['name'] not in processed_names]
                
                if unprocessed_videos:
                    videos_by_date[date_str] = unprocessed_videos
                    total_videos_to_process += len(unprocessed_videos)
                    print(f"📹 {date_str}: {len(unprocessed_videos)} unprocessed video(s)")
                else:
                    print(f"✅ {date_str}: All videos already processed")
            
            if total_videos_to_process == 0:
                print("📁 No unprocessed videos found in any date folder")
                return
            
            print(f"🚀 Found {total_videos_to_process} unprocessed video(s) across {len(videos_by_date)} date folder(s)")
            
            # Start auto-processing in background thread
            def auto_process_thread():
                try:
                    total_processed = 0
                    total_failed = 0
                    
                    for date_str, videos in videos_by_date.items():
                        print(f"\n📅 Processing videos for {date_str}...")
                        
                        for video in videos:
                            try:
                                print(f"🎬 Processing: {video['name']}")
                                
                                # Process individual video
                                from pathlib import Path
                                output_dir = self.auto_processor.analysis_folder / date_str / Path(video['name']).stem
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Get auto-processing preferences with defaults
                                show_footage = getattr(self, 'auto_show_footage', tk.BooleanVar(value=False)).get()
                                realtime_display = getattr(self, 'auto_realtime_display', tk.BooleanVar(value=True)).get()
                                
                                success = self.auto_processor.process_video(
                                    video['path'], 
                                    str(output_dir), 
                                    video['name'],
                                    show_footage,
                                    realtime_display
                                )
                                
                                if success:
                                    total_processed += 1
                                    print(f"✅ Processed: {video['name']}")
                                else:
                                    total_failed += 1
                                    print(f"❌ Failed: {video['name']}")
                                    
                            except Exception as e:
                                total_failed += 1
                                print(f"❌ Error processing {video['name']}: {e}")
                    
                    print(f"\n📊 Auto-processing completed!")
                    print(f"✅ Successfully processed: {total_processed} videos")
                    print(f"❌ Failed: {total_failed} videos")
                    
                    if total_processed > 0:
                        # Enable analysis buttons after auto-processing
                        self.root.after(0, self.enable_analysis_buttons)
                        
                except Exception as e:
                    print(f"❌ Auto-processing error: {e}")
            
            # Start background thread
            import threading
            thread = threading.Thread(target=auto_process_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"❌ Error in auto-detect: {e}")
    
    def sync_all_historical_data(self):
        """Sync all historical data to Firebase"""
        try:
            self.log_message("📊 Starting historical data sync...")
            self.update_progress(10, "Collecting historical data...")
            
            # Confirm sync
            result = messagebox.askyesno(
                "Confirm Sync", 
                "This will sync ALL historical data to Firebase. This may take some time.\n\nContinue?"
            )
            
            if not result:
                return
            
            # Sync all historical data
            success = self.firebase_sync.sync_all_historical_data()
            
            if success:
                self.update_progress(100, "Historical sync completed!")
                self.log_message("✅ All historical data synced to Firebase!")
                messagebox.showinfo("Success", "All historical data has been synced to Firebase!")
            else:
                self.log_message("⚠️ Historical sync failed. Check logs for details.")
                messagebox.showwarning("Warning", "Historical sync failed. Check logs for details.")
            
        except Exception as e:
            self.log_message(f"❌ Historical sync error: {str(e)}")
            messagebox.showerror("Error", f"Historical sync failed: {str(e)}")
    
    def get_video_metadata(self, video_path):
        """Get video metadata for database storage"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'resolution': f"{width}x{height}",
                    'frame_count': frame_count
                }
        except Exception as e:
            self.log_message(f"⚠️ Could not get video metadata: {e}")
        
        return {}
    
    def save_analysis_to_database(self, total_frames):
        """Save analysis results to database"""
        try:
            if not self.current_session_id:
                self.log_message("⚠️ No current session ID for database storage")
                return
            
            # Load comprehensive analysis data
            comprehensive_file = os.path.join(self.analyzer.output_dir, "comprehensive_analysis_report.json")
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Update session with results
                self.data_manager.update_analysis_session(
                    self.current_session_id,
                    total_frames=total_frames,
                    status='completed'
                )
                
                # Save analysis results
                self.data_manager.save_analysis_results(self.current_session_id, analysis_data)
                
                self.log_message("💾 Analysis results saved to database")
            else:
                self.log_message("⚠️ Comprehensive analysis file not found")
                
        except Exception as e:
            self.log_message(f"❌ Error saving to database: {str(e)}")

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set window properties before creating the app
    root.title("🎓 Classroom Video Analyzer")
    root.geometry("1400x1200")
    root.minsize(1300, 900)
    root.configure(bg='#f0f0f0')
    
    # Try to set icon immediately
    try:
        if os.path.exists('classroom_icon.ico'):
            root.iconbitmap('classroom_icon.ico')
            print("✅ Custom icon set for main window")
    except Exception as e:
        print(f"⚠️ Could not set custom icon: {e}")
    
    app = ClassroomAnalyzerGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
