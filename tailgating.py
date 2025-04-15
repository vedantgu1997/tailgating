import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sort.sort import Sort
import os
from glob import glob
from collections import deque

# ---- Device Setup ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Configuration ----
PERSON_MODEL_PATH = "weights/yolov8n.pt"
DOOR_MODEL_PATH = "weights/best_door.pt"
INPUT_DIR = "input_data"
OUTPUT_DIR = "output_data"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load models ----
person_model = YOLO(PERSON_MODEL_PATH)
door_model = YOLO(DOOR_MODEL_PATH)

# ---- Trackers ----
person_tracker = Sort()
door_tracker = Sort()

# ---- Tailgating Detection Parameters ----
TAILGATE_THRESHOLD_SEC = 3

# ---- Helper: Check if a point is inside a bounding box ----
def is_in_zone(centroid, zone):
    if not zone:
        return False
    x, y = centroid
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

# ---- Process each video ----
video_paths = glob(os.path.join(INPUT_DIR, "*.MOV")) + glob(os.path.join(INPUT_DIR, "*.mp4"))
summary_rows = []
for video_path in video_paths:
    filename = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    track_history = {}
    tailgating_events = []

    entry_times = {}      # track_id -> last entry timestamp
    already_inside = set()  # track_ids currently in the zone

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time_sec = frame_idx / fps

        # ---- Detect people ----
        person_results = person_model(frame, verbose=False)[0]
        person_detections = [
            list(det.xyxy[0].cpu().numpy()) + [float(det.conf[0])]
            for det in person_results.boxes
            if int(det.cls[0]) == 0 and float(det.conf[0]) > 0.3
        ]
        person_dets = np.array(person_detections) if person_detections else np.empty((0, 5))
        person_tracks = person_tracker.update(person_dets)

        # ---- Detect door and define tailgate zone ----
        door_results = door_model(frame, verbose=False)[0]
        door_box = None
        tailgate_zone = None

        for det in door_results.boxes:
            conf = float(det.conf[0])
            if conf > 0.3:
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                door_box = (x1, y1, x2, y2)
                tailgate_zone = (x1, y1, x2, y2)
                break  # only take one door

        # ---- Draw door region ----
        if door_box:
            cv2.rectangle(frame, door_box[:2], door_box[2:], (255, 0, 0), 2)
            cv2.putText(frame, "Door", (door_box[0], door_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        
        # ---- Check person entries into region ----
        for track in person_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, centroid, 4, (0, 255, 255), -1)

            in_zone = is_in_zone(centroid, tailgate_zone)
            was_inside = track_id in already_inside

            if in_zone and not was_inside:
                # print(track_id, "entered the zone at", curr_time_sec)
                # New entry detected
                entry_times[track_id] = curr_time_sec
                already_inside.add(track_id)

                # Check against all other entries
                for other_id, other_time in entry_times.items():
                    if other_id != track_id and abs(curr_time_sec - other_time) <= TAILGATE_THRESHOLD_SEC:
                        tailgating_events.append((curr_time_sec, track_id, other_id))
                        cv2.putText(frame, "TAILGATING!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            elif not in_zone and was_inside:
                already_inside.remove(track_id)

            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=50)
            track_history[track_id].append((curr_time_sec, centroid))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if '.MOV' in filename:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(frame)
        cv2.imshow("Tailgating Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Processed {filename} | Tailgating events: {len(tailgating_events)}")
    if tailgating_events:
        first_event_time = min(e[0] for e in tailgating_events)
        summary_rows.append({
            "video_path": video_path,
            "num_tailgating_events": len(tailgating_events),
            "timestamp_first_event": round(first_event_time, 2)
        })


cv2.destroyAllWindows()

if summary_rows:    
    log_path = os.path.join(LOGS_DIR , "tailgating_summary.csv")
    log_df = pd.DataFrame(summary_rows)

    if os.path.exists(log_path):
        existing_df = pd.read_csv(log_path)
        final_df = pd.concat([existing_df, log_df], ignore_index=True)
    else:
        final_df = log_df

    final_df.drop_duplicates(subset="video_path", keep="last", inplace=True)
    final_df.to_csv(log_path, index=False)
    print(f"Tailgating summary updated: {log_path}")
