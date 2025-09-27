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
import webbrowser
from datetime import datetime
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

class ClassroomAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Classroom Video Analyzer")
        self.root.geometry("1300x1000")  # Even larger window to prevent cropping
        self.root.configure(bg='#f0f0f0')
        self.root.minsize(1200, 800)  # Set minimum size to prevent cropping
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
        # Main container
        main_frame = ttk.Frame(self.root, padding="25")
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
        
        # First row of buttons
        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(row1_frame, text="▶️ Start Analysis", 
                                     command=self.start_analysis, style='Action.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 15))
        
        self.stop_button = ttk.Button(row1_frame, text="⏹️ Stop Analysis", 
                                    command=self.stop_analysis, style='Danger.TButton', state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 15))
        
        self.results_button = ttk.Button(row1_frame, text="📁 Open Results", 
                                       command=self.open_results, state='disabled')
        self.results_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Second row for additional buttons
        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.preview_button = ttk.Button(row2_frame, text="👁️ Preview Video", 
                                       command=self.preview_video, state='disabled')
        self.preview_button.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(button_frame, text="❓ Help", command=self.show_help).pack(side=tk.RIGHT)
        
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
        self.status_var.set("Analysis complete")
        
    def analysis_stopped(self):
        """Handle analysis stop"""
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.preview_button.config(state='normal')
        self.status_var.set("Analysis stopped")
        
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.is_analyzing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.preview_button.config(state='normal')
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

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set window properties before creating the app
    root.title("🎓 Classroom Video Analyzer")
    root.geometry("1200x900")
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
