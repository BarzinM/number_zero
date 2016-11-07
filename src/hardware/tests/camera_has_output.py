from __future__ import print_function
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(sum(sum(gray))!=0.0)

cap.release()
cv2.destroyAllWindows()