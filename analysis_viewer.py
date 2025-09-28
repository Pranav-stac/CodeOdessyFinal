"""
Advanced Analysis Viewer for Classroom Analyzer
Provides detailed student-wise analysis with face images and comprehensive reports
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from pathlib import Path
import base64
from PIL import Image, ImageTk, ImageDraw, ImageFont
import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import Counter, defaultdict
import cv2

class AnalysisViewer:
    def __init__(self, root, analysis_data=None, analysis_dir=None):
        self.root = root
        self.analysis_data = analysis_data
        self.analysis_dir = analysis_dir
        self.current_student = None
        self.student_data = {}
        self.face_images = {}
        
        # Load analysis data if directory provided
        if analysis_dir and not analysis_data:
            self.load_analysis_data(analysis_dir)
        
        self.setup_ui()
        self.load_student_data()
        
    def setup_ui(self):
        """Setup the main UI for analysis viewer"""
        self.root.title("📊 Classroom Analysis Viewer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_overview_tab()
        self.create_student_details_tab()
        self.create_face_gallery_tab()
        self.create_statistics_tab()
        self.create_timeline_tab()
        
    def create_overview_tab(self):
        """Create overview tab with summary statistics"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Create scrollable frame
        canvas = tk.Canvas(overview_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(overview_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Overview content
        self.create_overview_content(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_overview_content(self, parent):
        """Create overview content with key metrics"""
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="📊 Analysis Overview", 
                 font=('Arial', 18, 'bold')).pack(side=tk.LEFT)
        
        # Load analysis button
        ttk.Button(title_frame, text="📁 Load Analysis", 
                  command=self.load_analysis_dialog).pack(side=tk.RIGHT)
        
        # Key metrics frame
        metrics_frame = ttk.LabelFrame(parent, text="Key Metrics", padding="15")
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        if self.analysis_data:
            overview = self.analysis_data.get('overview_statistics', {})
            
            metrics = [
                ("Total Students", overview.get('total_students_tracked', 0)),
                ("Unique Faces", overview.get('total_unique_faces_detected', 0)),
                ("Video Duration", f"{overview.get('video_duration_seconds', 0):.1f}s"),
                ("Analysis Success", overview.get('analysis_success_rate', 'N/A'))
            ]
            
            for i, (label, value) in enumerate(metrics):
                row = i // 2
                col = i % 2
                
                metric_frame = ttk.Frame(metrics_grid)
                metric_frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
                
                ttk.Label(metric_frame, text=label, font=('Arial', 10)).pack()
                ttk.Label(metric_frame, text=str(value), font=('Arial', 14, 'bold'), 
                         foreground='#2c3e50').pack()
            
            # Configure grid weights
            metrics_grid.columnconfigure(0, weight=1)
            metrics_grid.columnconfigure(1, weight=1)
        
        # Activity distribution
        self.create_activity_chart(parent)
        
        # Zone analysis
        self.create_zone_analysis(parent)
        
    def create_activity_chart(self, parent):
        """Create activity distribution chart"""
        chart_frame = ttk.LabelFrame(parent, text="Activity Distribution", padding="15")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        if self.analysis_data and 'activity_analysis' in self.analysis_data:
            activity_data = self.analysis_data['activity_analysis']
            overall_stats = activity_data.get('overall_activity_summary', {})
            activity_dist = overall_stats.get('activity_distribution', {})
            
            if activity_dist:
                # Create matplotlib chart
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#f0f0f0')
                
                activities = list(activity_dist.keys())
                counts = list(activity_dist.values())
                
                # Color mapping for activities
                colors = {
                    'listening': '#3498db',
                    'writing': '#2ecc71',
                    'raising_hand': '#e74c3c',
                    'distracted': '#f39c12',
                    'unknown': '#95a5a6'
                }
                
                chart_colors = [colors.get(activity, '#95a5a6') for activity in activities]
                
                wedges, texts, autotexts = ax.pie(counts, labels=activities, autopct='%1.1f%%',
                                                 colors=chart_colors, startangle=90)
                
                ax.set_title('Student Activity Distribution', fontsize=14, fontweight='bold')
                
                # Embed chart in tkinter
                canvas = FigureCanvasTkAgg(fig, chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(chart_frame, text="No activity data available").pack()
        else:
            ttk.Label(chart_frame, text="No analysis data loaded").pack()
    
    def create_zone_analysis(self, parent):
        """Create classroom zone analysis"""
        zone_frame = ttk.LabelFrame(parent, text="Classroom Zone Analysis", padding="15")
        zone_frame.pack(fill=tk.X, pady=(0, 20))
        
        if self.analysis_data and 'classroom_zones' in self.analysis_data:
            zones_data = self.analysis_data['classroom_zones']
            
            zones_grid = ttk.Frame(zone_frame)
            zones_grid.pack(fill=tk.X)
            
            for i, (zone_name, zone_data) in enumerate(zones_data.items()):
                zone_frame_item = ttk.Frame(zones_grid)
                zone_frame_item.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
                
                # Zone color indicator
                color_frame = ttk.Frame(zone_frame_item)
                color_frame.pack(fill=tk.X)
                
                zone_colors = {
                    'front_zone': '#2ecc71',
                    'middle_zone': '#f39c12', 
                    'back_zone': '#e74c3c'
                }
                
                color = zone_colors.get(zone_name, '#95a5a6')
                ttk.Label(color_frame, text="●", foreground=color, font=('Arial', 20)).pack(side=tk.LEFT)
                ttk.Label(color_frame, text=zone_name.replace('_', ' ').title(), 
                         font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=(5, 0))
                
                # Zone metrics
                ttk.Label(zone_frame_item, text=f"Students: {zone_data.get('student_count', 0)}").pack()
                ttk.Label(zone_frame_item, text=f"Engagement: {zone_data.get('engagement_rate', 'N/A')}").pack()
            
            # Configure grid weights
            for i in range(len(zones_data)):
                zones_grid.columnconfigure(i, weight=1)
        else:
            ttk.Label(zone_frame, text="No zone analysis data available").pack()
    
    def create_student_details_tab(self):
        """Create student details tab with individual analysis"""
        student_frame = ttk.Frame(self.notebook)
        self.notebook.add(student_frame, text="👥 Students")
        
        # Split into student list and details
        paned_window = ttk.PanedWindow(student_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Student list frame
        list_frame = ttk.LabelFrame(paned_window, text="Student List", padding="10")
        paned_window.add(list_frame, weight=1)
        
        # Student listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_list = ttk.Scrollbar(listbox_frame)
        scrollbar_list.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.student_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar_list.set)
        self.student_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_list.config(command=self.student_listbox.yview)
        
        self.student_listbox.bind('<<ListboxSelect>>', self.on_student_select)
        
        # Student details frame
        details_frame = ttk.LabelFrame(paned_window, text="Student Details", padding="10")
        paned_window.add(details_frame, weight=2)
        
        self.student_details_frame = details_frame
        
        # Populate student list
        self.populate_student_list()
        
    def populate_student_list(self):
        """Populate the student listbox"""
        if self.analysis_data and 'students' in self.analysis_data:
            students = self.analysis_data['students']
            
            for student_id, student_data in students.items():
                # Create display text with key info
                behavior = student_data.get('behavior_summary', {})
                engagement = behavior.get('engagement_rate', 'N/A')
                dominant_activity = behavior.get('dominant_activity', 'unknown')
                
                display_text = f"{student_id} | {engagement} engaged | {dominant_activity}"
                self.student_listbox.insert(tk.END, display_text)
                
                # Store student data for quick access
                self.student_data[student_id] = student_data
        else:
            self.student_listbox.insert(tk.END, "No student data available")
    
    def on_student_select(self, event):
        """Handle student selection"""
        selection = self.student_listbox.curselection()
        if selection:
            index = selection[0]
            student_id = list(self.student_data.keys())[index]
            self.current_student = student_id
            self.display_student_details(student_id)
    
    def display_student_details(self, student_id):
        """Display detailed information for selected student"""
        # Clear existing details
        for widget in self.student_details_frame.winfo_children():
            widget.destroy()
        
        if student_id not in self.student_data:
            ttk.Label(self.student_details_frame, text="No student data available").pack()
            return
        
        student_data = self.student_data[student_id]
        
        # Create scrollable frame for details
        canvas = tk.Canvas(self.student_details_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(self.student_details_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Student header with face image
        self.create_student_header(scrollable_frame, student_id, student_data)
        
        # Behavior analysis
        self.create_behavior_analysis(scrollable_frame, student_data)
        
        # Activity timeline
        self.create_activity_timeline(scrollable_frame, student_data)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_student_header(self, parent, student_id, student_data):
        """Create student header with face image and basic info"""
        header_frame = ttk.LabelFrame(parent, text=f"Student: {student_id}", padding="15")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Split header into image and info
        header_content = ttk.Frame(header_frame)
        header_content.pack(fill=tk.X)
        
        # Face image frame
        image_frame = ttk.Frame(header_content)
        image_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        # Try to load face image
        face_data = student_data.get('face_data')
        if face_data and face_data.get('base64_image'):
            try:
                # Decode base64 image
                image_data = base64.b64decode(face_data['base64_image'])
                image = Image.open(io.BytesIO(image_data))
                
                # Resize image for display
                image = image.resize((120, 120), Image.Resampling.LANCZOS)
                
                # Create rounded image
                rounded_image = self.create_rounded_image(image)
                
                photo = ImageTk.PhotoImage(rounded_image)
                
                image_label = ttk.Label(image_frame, image=photo)
                image_label.image = photo  # Keep reference
                image_label.pack()
                
                # Confidence info
                confidence = face_data.get('confidence', 'N/A')
                ttk.Label(image_frame, text=f"Confidence: {confidence}", 
                         font=('Arial', 9)).pack(pady=(5, 0))
                
            except Exception as e:
                ttk.Label(image_frame, text="No face image", 
                         font=('Arial', 10)).pack()
        else:
            ttk.Label(image_frame, text="No face image", 
                     font=('Arial', 10)).pack()
        
        # Student info
        info_frame = ttk.Frame(header_content)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tracking_info = student_data.get('tracking_summary', {})
        behavior_info = student_data.get('behavior_summary', {})
        
        info_items = [
            ("Total Appearances", tracking_info.get('total_appearances', 'N/A')),
            ("Position Zone", tracking_info.get('position_zone', 'N/A')),
            ("Duration in Class", tracking_info.get('duration_in_class', 'N/A')),
            ("Engagement Rate", behavior_info.get('engagement_rate', 'N/A')),
            ("Engagement Level", behavior_info.get('engagement_level', 'N/A')),
            ("Dominant Activity", behavior_info.get('dominant_activity', 'N/A'))
        ]
        
        for label, value in info_items:
            info_row = ttk.Frame(info_frame)
            info_row.pack(fill=tk.X, pady=2)
            
            ttk.Label(info_row, text=f"{label}:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
            ttk.Label(info_row, text=str(value), font=('Arial', 10)).pack(side=tk.LEFT, padx=(10, 0))
    
    def create_behavior_analysis(self, parent, student_data):
        """Create detailed behavior analysis section"""
        behavior_frame = ttk.LabelFrame(parent, text="Behavior Analysis", padding="15")
        behavior_frame.pack(fill=tk.X, pady=(0, 15))
        
        behavior_info = student_data.get('behavior_summary', {})
        
        # Activity distribution
        activity_dist = behavior_info.get('activity_distribution', {})
        if activity_dist:
            activity_frame = ttk.Frame(behavior_frame)
            activity_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(activity_frame, text="Activity Distribution:", 
                     font=('Arial', 11, 'bold')).pack(anchor=tk.W)
            
            for activity, count in activity_dist.items():
                activity_row = ttk.Frame(activity_frame)
                activity_row.pack(fill=tk.X, pady=1)
                
                # Activity color bar
                color_frame = ttk.Frame(activity_row)
                color_frame.pack(side=tk.LEFT, padx=(0, 10))
                
                colors = {
                    'listening': '#3498db',
                    'writing': '#2ecc71',
                    'raising_hand': '#e74c3c',
                    'distracted': '#f39c12',
                    'unknown': '#95a5a6'
                }
                
                color = colors.get(activity, '#95a5a6')
                ttk.Label(color_frame, text="■", foreground=color, font=('Arial', 12)).pack()
                
                ttk.Label(activity_row, text=f"{activity}: {count} times").pack(side=tk.LEFT)
        
        # Attention distribution
        attention_dist = behavior_info.get('attention_distribution', {})
        if attention_dist:
            attention_frame = ttk.Frame(behavior_frame)
            attention_frame.pack(fill=tk.X)
            
            ttk.Label(attention_frame, text="Attention Distribution:", 
                     font=('Arial', 11, 'bold')).pack(anchor=tk.W)
            
            for attention, count in attention_dist.items():
                attention_row = ttk.Frame(attention_frame)
                attention_row.pack(fill=tk.X, pady=1)
                
                ttk.Label(attention_row, text=f"{attention}: {count} times").pack(side=tk.LEFT)
    
    def create_activity_timeline(self, parent, student_data):
        """Create activity timeline visualization"""
        timeline_frame = ttk.LabelFrame(parent, text="Activity Timeline", padding="15")
        timeline_frame.pack(fill=tk.X, pady=(0, 15))
        
        # This would show a timeline of activities over time
        # For now, show a placeholder
        ttk.Label(timeline_frame, text="Activity timeline visualization would go here", 
                 font=('Arial', 10)).pack()
        
        # In a full implementation, this would show:
        # - Timeline chart with activities over time
        # - Engagement patterns
        # - Zone movements
        # - Key moments (raised hand, writing, etc.)
    
    def create_face_gallery_tab(self):
        """Create face gallery tab showing all detected faces"""
        gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(gallery_frame, text="👤 Face Gallery")
        
        # Gallery controls
        controls_frame = ttk.Frame(gallery_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Face Gallery", font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        # Gallery canvas with scrollbar
        canvas_frame = ttk.Frame(gallery_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.gallery_canvas = tk.Canvas(canvas_frame, bg='#f0f0f0')
        gallery_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_scrollable_frame = ttk.Frame(self.gallery_canvas)
        
        self.gallery_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        
        self.gallery_canvas.create_window((0, 0), window=self.gallery_scrollable_frame, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        gallery_scrollbar.pack(side="right", fill="y")
        
        # Populate gallery
        self.populate_face_gallery()
    
    def populate_face_gallery(self):
        """Populate the face gallery with all detected faces"""
        if self.analysis_data and 'faces' in self.analysis_data:
            faces_data = self.analysis_data['faces']
            
            # Create grid layout for faces
            faces_per_row = 4
            row = 0
            col = 0
            
            for face_id, face_data in faces_data.items():
                face_frame = ttk.LabelFrame(self.gallery_scrollable_frame, 
                                          text=f"Face {face_data['face_id']}", 
                                          padding="10")
                face_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                
                # Face image
                best_image_info = face_data.get('best_image_info', {})
                if best_image_info.get('base64_image'):
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(best_image_info['base64_image'])
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Resize for gallery
                        image = image.resize((150, 150), Image.Resampling.LANCZOS)
                        
                        # Create rounded image
                        rounded_image = self.create_rounded_image(image)
                        
                        photo = ImageTk.PhotoImage(rounded_image)
                        
                        image_label = ttk.Label(face_frame, image=photo)
                        image_label.image = photo  # Keep reference
                        image_label.pack()
                        
                    except Exception as e:
                        ttk.Label(face_frame, text="Image Error", 
                                 font=('Arial', 10)).pack()
                else:
                    ttk.Label(face_frame, text="No Image", 
                             font=('Arial', 10)).pack()
                
                # Face info
                detection_summary = face_data.get('detection_summary', {})
                
                info_items = [
                    f"Appearances: {detection_summary.get('total_appearances', 'N/A')}",
                    f"Avg Confidence: {detection_summary.get('average_confidence', 'N/A')}",
                    f"Best Confidence: {best_image_info.get('confidence', 'N/A')}",
                    f"Quality Score: {best_image_info.get('quality_score', 'N/A')}"
                ]
                
                for info in info_items:
                    ttk.Label(face_frame, text=info, font=('Arial', 9)).pack()
                
                # Update grid position
                col += 1
                if col >= faces_per_row:
                    col = 0
                    row += 1
            
            # Configure grid weights
            for i in range(faces_per_row):
                self.gallery_scrollable_frame.columnconfigure(i, weight=1)
        else:
            ttk.Label(self.gallery_scrollable_frame, text="No face data available", 
                     font=('Arial', 12)).pack()
    
    def create_statistics_tab(self):
        """Create statistics tab with comprehensive charts"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 Statistics")
        
        # Create scrollable frame for statistics
        canvas = tk.Canvas(stats_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Statistics content
        self.create_statistics_content(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_statistics_content(self, parent):
        """Create comprehensive statistics content"""
        if not self.analysis_data:
            ttk.Label(parent, text="No analysis data available", 
                     font=('Arial', 12)).pack()
            return
        
        # Engagement over time chart
        self.create_engagement_chart(parent)
        
        # Zone comparison chart
        self.create_zone_comparison_chart(parent)
        
        # Activity frequency chart
        self.create_activity_frequency_chart(parent)
    
    def create_engagement_chart(self, parent):
        """Create engagement over time chart"""
        chart_frame = ttk.LabelFrame(parent, text="Engagement Over Time", padding="15")
        chart_frame.pack(fill=tk.X, pady=(0, 20))
        
        if 'frame_analysis' in self.analysis_data:
            frame_data = self.analysis_data['frame_analysis']
            
            if frame_data.get('sample_frames'):
                # Create matplotlib chart
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#f0f0f0')
                
                frames = [f['frame_number'] for f in frame_data['sample_frames']]
                engagement_counts = [f['engagement_count'] for f in frame_data['sample_frames']]
                
                ax.plot(frames, engagement_counts, marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Engaged Students')
                ax.set_title('Student Engagement Over Time')
                ax.grid(True, alpha=0.3)
                
                # Embed chart
                canvas = FigureCanvasTkAgg(fig, chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(chart_frame, text="No frame data available").pack()
        else:
            ttk.Label(chart_frame, text="No engagement data available").pack()
    
    def create_zone_comparison_chart(self, parent):
        """Create zone comparison chart"""
        chart_frame = ttk.LabelFrame(parent, text="Zone Comparison", padding="15")
        chart_frame.pack(fill=tk.X, pady=(0, 20))
        
        if 'classroom_zones' in self.analysis_data:
            zones_data = self.analysis_data['classroom_zones']
            
            # Create matplotlib chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('#f0f0f0')
            
            zone_names = [name.replace('_', ' ').title() for name in zones_data.keys()]
            student_counts = [data.get('student_count', 0) for data in zones_data.values()]
            engagement_rates = [float(data.get('engagement_rate', '0%').replace('%', '')) 
                              for data in zones_data.values()]
            
            # Student count bar chart
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            bars1 = ax1.bar(zone_names, student_counts, color=colors)
            ax1.set_title('Students per Zone')
            ax1.set_ylabel('Number of Students')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Engagement rate bar chart
            bars2 = ax2.bar(zone_names, engagement_rates, color=colors)
            ax2.set_title('Engagement Rate by Zone')
            ax2.set_ylabel('Engagement Rate (%)')
            ax2.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Embed chart
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(chart_frame, text="No zone data available").pack()
    
    def create_activity_frequency_chart(self, parent):
        """Create activity frequency chart"""
        chart_frame = ttk.LabelFrame(parent, text="Activity Frequency", padding="15")
        chart_frame.pack(fill=tk.X, pady=(0, 20))
        
        if 'activity_analysis' in self.analysis_data:
            activity_data = self.analysis_data['activity_analysis']
            overall_stats = activity_data.get('overall_activity_summary', {})
            activity_dist = overall_stats.get('activity_distribution', {})
            
            if activity_dist:
                # Create matplotlib chart
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#f0f0f0')
                
                activities = list(activity_dist.keys())
                counts = list(activity_dist.values())
                
                # Color mapping
                colors = {
                    'listening': '#3498db',
                    'writing': '#2ecc71',
                    'raising_hand': '#e74c3c',
                    'distracted': '#f39c12',
                    'unknown': '#95a5a6'
                }
                
                chart_colors = [colors.get(activity, '#95a5a6') for activity in activities]
                
                bars = ax.bar(activities, counts, color=chart_colors)
                ax.set_title('Activity Frequency Distribution')
                ax.set_xlabel('Activity Type')
                ax.set_ylabel('Frequency')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
                
                # Rotate x-axis labels if needed
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Embed chart
                canvas = FigureCanvasTkAgg(fig, chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(chart_frame, text="No activity data available").pack()
        else:
            ttk.Label(chart_frame, text="No activity analysis available").pack()
    
    def create_timeline_tab(self):
        """Create timeline tab showing analysis timeline"""
        timeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(timeline_frame, text="⏰ Timeline")
        
        # Timeline content
        ttk.Label(timeline_frame, text="Timeline Analysis", 
                 font=('Arial', 14, 'bold')).pack(pady=20)
        
        ttk.Label(timeline_frame, text="Timeline visualization would show:", 
                 font=('Arial', 12)).pack(pady=10)
        
        timeline_features = [
            "• Frame-by-frame analysis progression",
            "• Key moments and events",
            "• Student behavior changes over time",
            "• Engagement patterns",
            "• Zone movements",
            "• Activity transitions"
        ]
        
        for feature in timeline_features:
            ttk.Label(timeline_frame, text=feature, font=('Arial', 10)).pack(anchor=tk.W, padx=20)
    
    def create_rounded_image(self, image):
        """Create a rounded version of the image"""
        # Create a mask for rounded corners
        size = image.size
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        
        # Create output image with transparency
        output = Image.new('RGBA', size, (255, 255, 255, 0))
        output.paste(image, (0, 0))
        output.putalpha(mask)
        
        return output
    
    def load_analysis_data(self, analysis_dir):
        """Load analysis data from directory"""
        try:
            comprehensive_file = os.path.join(analysis_dir, "comprehensive_analysis_report.json")
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, 'r') as f:
                    self.analysis_data = json.load(f)
                print(f"✅ Loaded analysis data from {comprehensive_file}")
            else:
                print(f"❌ Comprehensive analysis file not found: {comprehensive_file}")
        except Exception as e:
            print(f"❌ Error loading analysis data: {e}")
    
    def load_student_data(self):
        """Load and process student data"""
        if self.analysis_data and 'students' in self.analysis_data:
            self.student_data = self.analysis_data['students']
            print(f"✅ Loaded {len(self.student_data)} students")
    
    def load_analysis_dialog(self):
        """Open dialog to load analysis data"""
        directory = filedialog.askdirectory(title="Select Analysis Directory")
        if directory:
            self.load_analysis_data(directory)
            self.load_student_data()
            
            # Refresh UI
            self.refresh_ui()
    
    def refresh_ui(self):
        """Refresh the UI with new data"""
        # Clear existing content and reload
        for i in range(self.notebook.index("end")):
            tab_frame = self.notebook.nth(i)
            for widget in tab_frame.winfo_children():
                widget.destroy()
        
        # Recreate tabs with new data
        self.create_overview_tab()
        self.create_student_details_tab()
        self.create_face_gallery_tab()
        self.create_statistics_tab()
        self.create_timeline_tab()

def main():
    """Main function to run the analysis viewer"""
    root = tk.Tk()
    app = AnalysisViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

