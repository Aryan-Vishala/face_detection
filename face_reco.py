import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv.VideoCapture(0)
while True:
    _,img = webcam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,225,0), 3)
    cv.imshow("Face detection", img)
    key = cv.waitKey(10)
    if key == 27:
        break
webcam.release()
cv.destroyAllWindows()