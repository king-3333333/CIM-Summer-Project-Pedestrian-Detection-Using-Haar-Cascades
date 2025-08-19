import cv2

# Initialize the webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Detect pedestrians
    (regions, _) = hog.detectMultiScale(frame,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Draw bounding boxes and labels
    for i, (x, y, w, h) in enumerate(regions, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Person {i}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Webcam Pedestrian Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
