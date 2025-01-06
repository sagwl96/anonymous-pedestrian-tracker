import cv2
import time

# List to store points
points = []
lengths = []

# Mouse callback function to capture click coordinates
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the current time for the file name
start_time = int(time.time())
file_name = f'homography_data_{start_time}.txt'

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
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow("Webcam View", frame)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # Close on space key when 4 points are marked
    if key == 32 and len(points) == 4:  # Spacebar key ASCII
        break

if len(points) < 4:
    print("Error: You must mark exactly 4 points.")
else:
    # Ask for 4 length inputs
    for i in range(4):
        while True:
            try:
                length = float(input(f"Enter length {i+1}: "))
                lengths.append(length)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Save points and lengths to a text file
    with open(file_name, "w") as file:
        for point in points:
            file.write(f"{point[0]}, {point[1]}\n")
        for length in lengths:
            file.write(f"{length}\n")

    print(f"Points and lengths saved to {file_name}")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
