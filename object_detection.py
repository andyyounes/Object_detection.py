import cv2

# Load the pre-trained Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Configure the camera
camera = cv2.VideoCapture(0)  # Use the default camera

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    # Check for the 'Esc' key (key code 27) to exit
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
