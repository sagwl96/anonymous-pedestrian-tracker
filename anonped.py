# Anonymous pedestrian tracking and recording
# Python version 3.9.19

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator
import time

# Privacy mode variable
privacy_mode = True

track_history = defaultdict(lambda: [])

model = YOLO("yolo11m.pt")
names = model.model.names

# Open a connection to the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the current time for the file name
start_time = int(time.time())
file_name = f'tracking_data_{start_time}.txt'

# Open the file to store tracking data
with open(file_name, 'w') as file:
    file.write('timeframe,id,position_x,position_y\n')  # Write the header

    while cap.isOpened():
        success, frame = cap.read()

        if success:

            results = model.track(frame, persist=True)

            boxes = results[0].boxes.xywh.cpu()  # Bounding boxes in xywh format
            clss = results[0].boxes.cls.cpu().tolist()  # Class IDs

            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            except:
                continue

            if privacy_mode:
                display_frame = np.zeros_like(frame)  # Black background
            else:
                display_frame = frame.copy()  # Original frame

            annotator = Annotator(display_frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls == 0:  # Filter to detect only people (class ID 0)
                    x, y, w, h = box  # x, y are the center coordinates of the bounding box
                    x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)  # Convert to corner coordinates
                    label = str(names[cls]) + " : " + str(track_id)
                    annotator.box_label([x1, y1, x2, y2], label, (218, 100, 255))

                    # Tracking Lines plot
                    track = track_history[track_id]
                    track.append((float(box[0]), float(box[1])))  # Append center coordinates
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [points], isClosed=False, color=(37, 255, 225), thickness=2)

                    # Center circle
                    cv2.circle(display_frame, (int(track[-1][0]), int(track[-1][1])), 5, (235, 219, 11), -1)

                    # Store tracking data in the file
                    current_time_ms = int(time.time() * 1000)
                    file.write(f'{current_time_ms},{track_id},{float(box[0])},{float(box[1])}\n')  # Save center coordinates

            cv2.imshow("YOLOv8 Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
