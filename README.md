# 🚪 Tailgating Detection System (YOLOv8 + SORT)

A computer vision pipeline to detect **tailgating behavior** in surveillance footage using real-time object detection and tracking. Tailgating is defined as unauthorized entry by closely following an authorized person through a secured door.

---

## 🧠 Overview

This system uses:
- **YOLOv8** for detecting persons and doors
- **SORT (Simple Online and Realtime Tracking)** for multi-person tracking
- A rule-based logic for detecting tailgating events based on:
  - **Person movement into the door zone**
  - **Time difference between consecutive entries**

---

## 🎯 Goal

Detect and log events where **two individuals enter a door region within a short time window** (default: 3 seconds), potentially indicating a tailgating violation.

---

## 🧩 System Components

### 📦 Model Setup

| Component       | Description                               |
|----------------|-------------------------------------------|
| `yolov8n.pt`    | YOLOv8 model pretrained to detect persons |
| `best_door.pt`  | Custom YOLOv11 model trained to detect doors |
| `Sort`          | Tracker for assigning persistent IDs to people |

---

### ⚙️ Parameters

- `TAILGATE_THRESHOLD_SEC = 3`  
  Time window in which two people entering the door zone triggers a tailgating alert.

- `INPUT_DIR`, `OUTPUT_DIR`, `LOGS_DIR`  
  Input videos, processed video outputs, and tailgating event logs.

---

## 🎥 Frame Processing Workflow

### 1. **Video Ingestion**
Each `.MOV` or `.mp4` file in the input directory is processed.

### 2. **Detection**
- Detects people in the frame using YOLOv8.
- Detects the first visible door.
- Confidence threshold for both: `> 0.3`.

### 3. **Tailgating Zone Definition**
- **No padding** is applied.
- The tailgating zone is **exactly the bounding box of the detected door**.

### 4. **Tracking and Logic**
- Each person is tracked using SORT and assigned a `track_id`.
- Person centroid is checked against the door zone.
- If a person enters the zone:
  - Their entry timestamp is stored.
  - If another person enters within 3 seconds, a **tailgating event is triggered**.
- A set `already_inside` tracks people currently within the door region to avoid repeated logging.

### 5. **Annotation and Rotation**
- Frames are rotated **90° clockwise** to correct `.MOV` metadata orientation.
- Drawn overlays:
  - Blue box: Door region
  - Green boxes: Person bounding boxes
  - Yellow dot: Person centroid
  - Red "TAILGATING!" warning if triggered

---

## 📊 Tailgating Event Logging

A single row is logged per video **only if** tailgating events are detected:

| Column                 | Description                                      |
|------------------------|--------------------------------------------------|
| `video_path`           | Path to the video file                           |
| `num_tailgating_events`| Number of events detected in that video          |
| `timestamp_first_event`| Time of the first event (in seconds)             |

Saved to:

```bash
logs/tailgating_summary.csv

## Future Improvements
 - Replacing SORT with DeepSORT for more reliable tracking
 - Replacing the rule-based system with a classifier trained on trajectories obtained from DeepSORT and door position.