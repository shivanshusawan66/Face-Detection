import cv2
import numpy as np

# Mapping of IDs to names
# id_to_name = {
#     1: "Shivanshu",
#     3: "Divyansh",
#     4: "nandan"
#     # Add more mappings here if you have more IDs and names
# }

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        face = gray_image[y:y+h, x:x+w]  # Corrected extraction of face region
        id, pred = clf.predict(face)

        # Confidence is derived from the prediction distance
        confidence = int(100 * (1 - pred / 400))  # Adjust the divisor based on your training data

        print(f"ID: {id}, Confidence: {confidence}")  # Debugging line

        if confidence > 77:  # Set a reasonable confidence threshold
            if id == 1:
                cv2.putText(img, "Shivanshu", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id == 2:
                cv2.putText(img, "Divyansh", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
            # if id == 3:
            #     cv2.putText(img, "Third_user", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        else:
            cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
    return img

# Load the pre-trained face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the LBPH face recognizer model
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Open the video capture (webcam)
video_capture = cv2.VideoCapture(0)

# Main loop to capture video frames
while True:
    ret, img = video_capture.read()  # Capture frame-by-frame
    img = recognize(img, clf, faceCascade)  # Recognize faces in the frame
    cv2.imshow("Face Detection", img)  # Show the result in a window

    if cv2.waitKey(1) == 13:  # Break on pressing 'Enter' key (ASCII code 13)
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
