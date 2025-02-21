# Smile detector using haar cascades
# loading the file from https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_smile.xml

import cv2

def detect():
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
  camera = cv2.VideoCapture(0)
  while (True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()