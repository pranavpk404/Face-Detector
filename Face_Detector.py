import cv2

face_file = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)
while True:
    su_frame_read, frame = webcam.read()

    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_file.detectMultiScale(bw_img)

    for (x, y, z, h) in faces:
        cv2.rectangle(frame, (x, y), (x + z, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Face", (x, y + h + 40), fontScale=2, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 255))

    cv2.imshow("Face Detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()
