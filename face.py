# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 21:49
# @Author  : https://blog.csdn.net/monster663/article/details/118341515
# @FileName: face.py.py
# @Software: IntelliJ IDEA
# 文档：face.py.note
# 链接：http://note.youdao.com/noteshare?id=64d8b933445a6c54362d1ec209007bc4&sub=8CB15C3D1E764F29A8DACE824F39BB66
# 人脸的 landmark 画上去

import cv2
import dlib
from math import sqrt
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
model_path="model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)

# read the image
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        #for n in range(0, 68):
        x67 = landmarks.part(67).x
        y67 = landmarks.part(67).y
        for n in range(0, 67):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            x2 = landmarks.part(n+1).x
            y2 = landmarks.part(n+1).y
            ptStart=(x,y)
            ptEnd=(x2,y2)
            point_color=(0,255,0)
            thickness=1
            lineType=4
            cv2.circle(img=frame, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
            if(n==16 or n==26 or n==35 or n==41 or n==47):
                continue
            cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType)

    cv2.imshow(winname="human face test", mat=frame)

    if cv2.waitKey(delay=1) == 27:
        break

cap.release()

cv2.destroyAllWindows()
