# coding=utf-8
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
# from sklearn.externals import joblib
# import joblib
VECTOR_SIZE = 3


def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue


def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)
# ModuleNotFoundError: No module named 'sklearn.svm.classes'
# 导入模型
# svm_path="ear_svm2021_09_21_19_09_06.m"
svm_path = "train/ear_svm2021_09_21_19_18_05.m"
svm_path_angry = "train/train_eye_brow_angry_2021_09_22_08_53_41.m"
clf = joblib.load(svm_path)
clf_angry = joblib.load(svm_path_angry)
# clf = joblib.load("ear_svm.m")

EYE_AR_THRESH = 0.3  # EAR阈值
EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0
blink_counter = 0
ear_vector = []
# file_path=0
# file_path=r"G:\download\zhengyan_video2021_09_21_18_53_08.avi"
# file_path=r"G:\download\biyan_video2021_09_21_18_53_08.avi"
file_path = r"G:\download\眨眼video2021_09_21_14_49_57.avi"


# cap = cv2.VideoCapture(1)
# svm 训练的 戴眼镜 和不带 眼镜  都比较好

def blink_detect_svm(ear_vector, ear):
    ret, ear_vector = queue_in(ear_vector, ear)
    if len(ear_vector) == VECTOR_SIZE:
        print(ear_vector)
        input_vector = [ear_vector]
        res = clf.predict(input_vector)
        print(res)
        return res
    return None
    # # 1 是闭眼
    # if res == 1:
    #     frame_counter += 1
    # else:
    #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
    #         blink_counter += 1
    #     frame_counter = 0


cap = cv2.VideoCapture(file_path)

while (1):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        print('-' * 20)
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        # points = shape.parts()
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        res = blink_detect_svm(ear_vector, ear)
        if res is not None:
            # continue
        # ret, ear_vector = queue_in(ear_vector, ear)
        # if (len(ear_vector) == VECTOR_SIZE):
        #     print(ear_vector)
        #     input_vector = []
        #     input_vector.append(ear_vector)
        #     res = clf.predict(input_vector)
        #     print(res)

            # 1 是闭眼
            if res == 1:
                frame_counter += 1
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0

        # train/train_eye_brow_angry_2021_09_22_08_53_41.m
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
