import cv2
import os

# Path to your saved video file
video_path = r'C:\GitHub\CIM-Summer-Project-Pedestrian-Detection-Using-Haar-Cascades\Input_Data\sample_video.mp4'

# Output video file path
output_path = r'C:\GitHub\CIM-Summer-Project-Pedestrian-Detection-Using-Haar-Cascades\Input_Data\output.mp4'

# Check if video exists
if not os.path.exists(video_path):
    print("Error: Video not found at the specified path.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize width if too large
new_width = min(600, width)
scale = new_width / width
new_height = int(height * scale)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize frame for faster processing
    frame = cv2.resize(frame, (new_width, new_height))

    # Detect people
    (regions, _) = hog.detectMultiScale(frame,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Draw rectangles and labels
    for i, (x, y, w, h) in enumerate(regions, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Person {i}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

    # Show the frame in a window
    cv2.imshow("Pedestrian Detection (Video)", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at: {output_path}")
