import cv2

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None
        
        for (x, y, w, h) in faces:
            margin = 30
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + 2 * margin
            h = h + 2 * margin
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    #Everytime you have to change it
    id = 2
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        face = face_cropped(frame)
        if face is not None:
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Cropped face", face)

            img_id += 1  # Increment image ID

            if cv2.waitKey(1) == 13 or img_id >= 200:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")

generate_dataset()
