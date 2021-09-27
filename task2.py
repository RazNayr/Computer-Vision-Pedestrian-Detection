import cv2 as cv

# Function to apply the captured frame to the classifiers and draw ROIs where detections are made.
def detect_faces_and_eyes(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    face_line_color = (0, 255, 0)
    face_line_type = cv.LINE_4

    eye_line_color = (0, 0, 255)
    eye_line_type = cv.LINE_4

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:
        top_left = (x, y)
        bottom_right = (x + w, y + h)

        # Draw face ROI
        frame = cv.rectangle(frame, top_left, bottom_right, face_line_color, lineType=face_line_type)
        faceROI = frame_gray[y:y+h, x:x+w]

        # Detect eyes within detected face
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_top_left = (x+x2, y+y2)
            eye_bottom_right = (x+x2 + w2, y+y2 + h2)

            # Draw eye ROI
            frame = cv.rectangle(frame, eye_top_left, eye_bottom_right, eye_line_color, lineType=eye_line_type)

    cv.imshow('Capture - Face detection', frame)


# Initialise Face Cascade
face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# Initialise Eye Cascade
eyes_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# Capture from camera
cap = cv.VideoCapture(0)

if not cap.isOpened:
    print('ERROR: Cannot open video capture.')
    exit(0)

while True:
    ret, frame = cap.read()

    if frame is None:
        print("WARNING: No frame captured.")
        break

    detect_faces_and_eyes(frame)

    if cv.waitKey(10) == 27:
        break