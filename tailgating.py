import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # Ensure sort.py is in your project
import os

# ---- Configuration ----
MODEL_PATH = "weights/yolov8n.pt"  # Replace with 'best.pt' for custom model
VIDEO_PATH = "input_data/vid1.mov"
OUTPUT_DIR = "output_data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tracked_output.mp4")

# ---- Create output directory if needed ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load model and tracker ----
model = YOLO(MODEL_PATH)
tracker = Sort()

# ---- Load video ----
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ---- Main Loop ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, verbose=False)[0]

    # Extract only 'person' detections (class 0 in COCO)
    detections = []
    for det in results.boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        conf = det.conf[0].cpu().numpy()
        cls = int(det.cls[0])  # class ID

        if cls == 0 and conf > 0.3:  # Only 'person'
            detections.append([x1, y1, x2, y2, conf])

    # Convert to NumPy array
    dets = np.array(detections)

    # Update tracker
    tracks = tracker.update(dets)

    # Draw tracked bounding boxes
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write frame
    out.write(frame)

    # Display (optional)
    cv2.imshow("YOLOv8 + SORT (Person)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# ---- Cleanup ----
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Tracked video saved at: {OUTPUT_PATH}")
