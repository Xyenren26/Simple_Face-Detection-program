import cv2

# Load the pre-trained face detection classifier, in this program I used haar cascade with the used of open cv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform face detection on each frame, take note this will detect by frame which means it might used some memories
def detect_faces(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

# Open a video capture object (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    
    # If frame is read correctly
    if ret:
        # Perform face detection on the frame
        frame_with_faces = detect_faces(frame)
        
        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame_with_faces)
    
    # Check for key press
    key = cv2.waitKey(1)
    
    # Exit loop if 'q' key is pressed
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
