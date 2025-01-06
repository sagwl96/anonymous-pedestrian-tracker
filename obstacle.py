# Python program to mark the obstacles
# Use left mouse click to mark the points around the obstacles
# Press spacebar at the end to store the file

import cv2
import time

# List to store points
points = []

# Mouse callback function to capture click coordinates
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the current time for the file name
start_time = int(time.time())
file_name = f'obstacle_data_{start_time}.txt'

cv2.namedWindow("Webcam View")
cv2.setMouseCallback("Webcam View", mouse_click)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw the points on the frame
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow("Webcam View", frame)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # Close on space key
    if key == 32:  # Spacebar key ASCII
        break

# Save points to a text file
with open(file_name, "w") as file:
    for point in points:
        file.write(f"{point[0]}, {point[1]}\n")

print(f"Obstacle pixel values stored in {file_name}")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
