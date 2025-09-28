# 🎓 Classroom Analyzer - GUI Features Guide

## 📍 Where to Find Attendance Mapping & Video Classification

### 🖥️ Main GUI Interface

When you run `python classroom_analyzer_gui.py`, you'll see these new buttons in the **Analysis Control** section:

#### **After Video Analysis is Complete:**

1. **👥 Match Faces** - Shows face matching across videos
2. **🎓 Classify Lecture** - Shows video type classification  
3. **📋 Attendance Report** - Shows comprehensive attendance mapping
4. **📊 View Analysis** - Opens detailed analysis viewer

---

## 🔍 Step-by-Step Usage Guide

### **Step 1: Process First Video**
1. Select your video file
2. Click "▶️ Start Analysis"
3. Wait for analysis to complete
4. **New buttons will become enabled** ✅

### **Step 2: View Attendance Mapping**
1. Click **"👥 Match Faces"** button
2. This will show:
   - How many faces were processed
   - How many faces matched existing database
   - How many new faces were added
   - Detailed face matching results

### **Step 3: Classify Video Type**
1. Click **"🎓 Classify Lecture"** button
2. This will show:
   - **Lecture Type**: (Lecture, Group Discussion, Hands-on Activity, etc.)
   - **Confidence Score**: How confident the classification is
   - **Method**: LLM-based or rule-based
   - **Reasoning**: Why it classified as that type

### **Step 4: View Comprehensive Attendance**
1. Click **"📋 Attendance Report"** button
2. This will show:
   - **Total persons** in database
   - **Total videos** processed
   - **Individual attendance records**
   - **Cross-video tracking** (same person in multiple videos)

### **Step 5: Detailed Analysis Viewer**
1. Click **"📊 View Analysis"** button
2. This opens a new window with tabs:
   - **Overview**: Charts and statistics
   - **Students**: Individual student analysis with face images
   - **Face Gallery**: All detected faces
   - **Statistics**: Comprehensive charts
   - **Timeline**: Activity over time

---

## 🎯 What You'll See

### **Face Matching Results Window:**
```
Face Matching Results
====================

Summary:
- Processed Faces: 15
- Matched Faces: 12  
- New Faces: 3

Face Matches:
- Face 01 → Person 001 (similarity: 0.89)
- Face 02 → Person 002 (similarity: 0.92)
- Face 03 → Person 001 (similarity: 0.85)

New Persons:
- Person 003 from Face 04
- Person 004 from Face 05

Attendance Summary:
- Total Persons: 25
- Total Videos Processed: 3
```

### **Lecture Classification Results Window:**
```
Lecture Classification Results
=============================

Classification:
- Type: Group Discussion
- Confidence: 0.87
- Method: rule_based

Reasoning:
- High interaction level detected
- Group activities observed  
- Active movement suggests discussion

Lecture Type Information:
- Description: Interactive group discussion or collaborative learning
- Keywords: discussion, group, collaborative, interactive, debate
```

### **Attendance Report Window:**
```
Attendance Report
================

Summary Tab:
- Total Persons: 25
- Total Videos Processed: 3
- Report Generated: 2024-01-15T10:30:00

Detailed Records Tab:
Person 001:
- Total Appearances: 8
- Videos Attended: 3
- Attendance Rate: 100%
- First Seen: 2024-01-15T09:00:00
- Last Seen: 2024-01-15T10:15:00

Person 002:
- Total Appearances: 6
- Videos Attended: 2  
- Attendance Rate: 66.7%
- First Seen: 2024-01-15T09:05:00
- Last Seen: 2024-01-15T09:45:00
```

---

## 🔄 Cross-Video Workflow

### **Process Multiple Videos:**
1. **Video 1**: Process normally → Click "Match Faces" → See new faces added
2. **Video 2**: Process normally → Click "Match Faces" → See faces matched to Video 1
3. **Video 3**: Process normally → Click "Match Faces" → See cumulative matching
4. **Attendance Report**: Shows attendance across ALL videos

### **Attendance Tracking:**
- **Same person in Video 1 + Video 2** = Attendance count = 2
- **Same person in Video 1 + Video 2 + Video 3** = Attendance count = 3
- **New person in Video 2** = Attendance count = 1

---

## 🎨 Visual Layout

```
┌─────────────────────────────────────────────────────────┐
│  🎓 Classroom Video Analyzer                           │
├─────────────────────────────────────────────────────────┤
│  📹 Video Selection                                    │
│  [Browse] [Video File Path]                            │
├─────────────────────────────────────────────────────────┤
│  ⚙️ Analysis Options                                   │
│  [Output Directory] [Headless Mode] [Save Frames]      │
├─────────────────────────────────────────────────────────┤
│  🚀 Analysis Control                                   │
│  [▶️ Start Analysis] [⏹️ Stop] [📁 Open Results]       │
│  [📊 View Analysis]                                    │
│  [👁️ Preview Video] [👥 Match Faces] [🎓 Classify]     │
│  [📋 Attendance Report] [❓ Help]                      │
├─────────────────────────────────────────────────────────┤
│  📊 Progress                                           │
│  [████████████████████████████████████████] 100%       │
│  Analysis complete!                                    │
├─────────────────────────────────────────────────────────┤
│  📋 Analysis Log                                       │
│  [10:30:15] ✅ Analysis completed successfully!        │
│  [10:30:16] 📊 Processed 1500 frames                   │
│  [10:30:17] 💾 Analysis results saved to database     │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

1. **Run the GUI**: `python classroom_analyzer_gui.py`
2. **Select video**: Click Browse and choose your video
3. **Start analysis**: Click "▶️ Start Analysis"
4. **Wait for completion**: Progress bar will show status
5. **Explore results**: Use the new buttons to see:
   - Face matching results
   - Lecture classification
   - Attendance reports
   - Detailed analysis viewer

---

## 💡 Tips

- **Face matching works best** with clear, front-facing faces
- **Lecture classification** uses video features (movement, interaction, presentation elements)
- **Attendance tracking** is automatic across all processed videos
- **All data is stored** in SQLite database for persistence
- **Analysis viewer** provides comprehensive visualizations

---

## 🔧 Troubleshooting

If buttons are disabled:
- Make sure analysis is complete first
- Check that video file exists and is valid
- Ensure all dependencies are installed

If no results appear:
- Check the Analysis Log for error messages
- Verify video has detectable faces and activities
- Try with a different video file

