import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define counting zones
counting_regions = [
    {
        "name": "Zone 1",
        "polygon": Polygon([(0, 0), (0, 1600), (2560, 1600), (2560, 0)]),
        "counts": 0,
        "color": (0, 255, 0),
    }
]

# Define the vertical counting line
counting_line = [(325, 0), (325, 480)]
crossed_objects = set()  # Track IDs of counted humans
track_history = defaultdict(list)  # Store previous positions of tracked objects
human_count = 0  # Count of humans who crossed the line

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object tracking
    frame = cv2.resize(frame, (650, 480))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()  # Bounding box coordinates
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Object tracking IDs
        print("track_id:", track_ids)
        clss = results[0].boxes.cls.cpu().tolist()  # Object classes

        for box, track_id, clss_id in zip(boxes, track_ids, clss):
            # Only process humans (class 0)
            if clss_id != 0:
                continue  # Ignore all non-human objects

            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = box

            # Calculate the center of the bounding box
            bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

            # Add the position to tracking history
            track_history[track_id].append(bbox_center)
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Draw bounding box on the image
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Check if the person has crossed the line
            if len(track_history[track_id]) > 1:
                prev_position = track_history[track_id][-2]
                current_position = bbox_center

                # Detect crossing event
                if (prev_position[0] < counting_line[0][0] and current_position[0] >= counting_line[0][0]) or \
                   (prev_position[0] > counting_line[0][0] and current_position[0] <= counting_line[0][0]):
                    if track_id not in crossed_objects:
                        crossed_objects.add(track_id)  # Mark this human as counted
                        print('track_id:', track_id)
                        human_count += 1  # Increment human count

            # Check if the person enters a counting zone
            for region in counting_regions:
                if region["polygon"].contains(Point(bbox_center)):
                    region["counts"] += 1

    # Draw the counting line
    cv2.line(frame, counting_line[0], counting_line[1], (0, 0, 255), 2)

    # Draw the counting zones
    for region in counting_regions:
        poly_pts = np.array(region["polygon"].exterior.coords, dtype=np.int32)
        cv2.polylines(frame, [poly_pts], isClosed=True, color=region["color"], thickness=2)
        cv2.putText(frame, f"{region['counts']}", (poly_pts[0][0], poly_pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, region["color"], 2)

    # Display the human count on the image
    cv2.putText(frame, f"Humans crossed: {human_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the output frame
    cv2.imshow("People Counting & Tracking", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
