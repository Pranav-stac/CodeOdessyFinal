# Face Tracking with Engagement Analysis

This script tracks faces across video frames, assigns unique IDs to each face, calculates engagement metrics, and generates detailed reports.

## Features

- **Face Tracking**: Assigns unique IDs to each face and tracks them across frames
- **Engagement Analysis**: Calculates engagement scores based on:
  - Activity detection (writing, listening, raising hand, etc.)
  - Attention levels (focused, partially focused, distracted)
  - Position zones (front, middle, back of classroom)
  - Detection confidence
- **Detailed Reports**: Generates comprehensive reports for each tracked face ID
- **Output Video**: Creates annotated video with bounding boxes and engagement information

## Usage

### Basic Usage

```bash
python face_track_engagement_analyzer.py <video_path>
```

### With Custom Output Directory

```bash
python face_track_engagement_analyzer.py <video_path> --output-dir <output_directory>
```

### Example

```bash
python face_track_engagement_analyzer.py test.mp4 --output-dir analysis_results
```

## Output Files

The script generates the following files in the output directory:

1. **`output_with_tracking.mp4`**: Annotated video with:
   - Bounding boxes around each tracked face
   - Face ID labels
   - Engagement state (engaged/partially_engaged/not_engaged)
   - Activity type
   - Engagement score

2. **`face_XXX_report.json`**: Individual detailed report for each face ID containing:
   - Tracking summary (duration, frames, timestamps)
   - Engagement metrics (average, max, min scores)
   - Activity breakdown
   - Attention analysis
   - Position zone analysis
   - Detection quality metrics

3. **`summary_report.json`**: Overall summary report with:
   - Video information
   - Face tracking summary
   - Overall engagement statistics

4. **`detailed_report.txt`**: Human-readable text report with all face details

## Report Structure

### Individual Face Report (JSON)

```json
{
  "face_id": 0,
  "tracking_summary": {
    "total_frames": 150,
    "total_time_seconds": 5.0,
    "first_seen_seconds": 1.2,
    "last_seen_seconds": 6.2
  },
  "engagement_metrics": {
    "average_engagement_score": 0.75,
    "max_engagement_score": 0.95,
    "min_engagement_score": 0.45,
    "engagement_percentages": {
      "engaged": 60.0,
      "partially_engaged": 30.0,
      "not_engaged": 10.0
    }
  },
  "activity_analysis": {
    "most_common_activity": "writing",
    "activity_breakdown": {
      "writing": 80,
      "listening": 50,
      "raising_hand": 20
    }
  },
  "attention_analysis": {
    "most_common_attention": "focused",
    "attention_breakdown": {
      "focused": 100,
      "partially_focused": 40,
      "distracted": 10
    }
  },
  "position_analysis": {
    "most_common_zone": "front",
    "zone_breakdown": {
      "front": 120,
      "middle": 30
    }
  }
}
```

## Engagement Score Calculation

The engagement score (0.0 to 1.0) is calculated based on:

1. **Activity** (weighted):
   - Raising hand: 1.0
   - Writing: 0.9
   - Reading: 0.7
   - Listening: 0.6
   - Talking: 0.5
   - Unknown: 0.2
   - Distracted/Using phone/Sleeping: 0.0-0.1

2. **Attention Level** (weighted):
   - Focused: +0.3
   - Partially focused: +0.15
   - Distracted/Not visible: +0.0

3. **Position Zone** (weighted):
   - Front: +0.2
   - Middle: +0.1
   - Back: +0.05

4. **Detection Confidence** (weighted):
   - Confidence score × 0.1

**Engagement States:**
- **Engaged**: Score ≥ 0.8
- **Partially Engaged**: Score 0.5 - 0.8
- **Not Engaged**: Score < 0.5

## Requirements

The script uses existing models from the project:
- YOLOv8s for person detection
- YOLOv8n-pose for pose estimation
- YOLOv12s-face for face detection

Make sure these model files are available in:
- `AI_Model_Weights/AI_Model_Weights/` directory

## Dependencies

All dependencies should already be installed. The script uses:
- OpenCV (cv2)
- NumPy
- Ultralytics (YOLO)
- tqdm (for progress bar)

## Notes

- The script processes videos frame by frame, so processing time depends on video length and resolution
- Face tracking uses centroid-based matching, which works well for most scenarios
- Engagement scores are calculated per frame and aggregated over the entire track
- The output video maintains the same resolution and FPS as the input video
