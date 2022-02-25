# coding=utf-8
import numpy as np
import os
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
import pickle

# from blink_detect import eyebrow_aspect_ratio

VECTOR_SIZE = 3

import util

now_time_str = util.get_now_time_str()


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

# cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
# 没有这句 变成 左边 白色 右边黑色 更加不对了
# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1



# txt = open('train_open.txt', 'wb')
# 需要配置类型
import commons

# train_file_prefix = "train_eye_brow_angry_"
# 首先配置类型
train_file_prefix = commons.train_type
# train_yes_file='train_yes_{}.txt'.format(now_time_str)
train_yes_file = 'train/{}_yes_{}.txt'.format(train_file_prefix, now_time_str)
# train_no_file = train_file_prefix + '_no_{}.txt'.format(now_time_str)
train_no_file = 'train/{}_no_{}.txt'.format(train_file_prefix, now_time_str)
# train_open_file = 'train_open{}.txt'.format(now_time_str)
# train_close_file = 'train_eye_brow_angry_{}.txt'.format(now_time_str)

# train_yes_file = train_yes_file
# train_no_file = train_no_file

# txt = open('train_open2.txt', "w+")
# txt = open(train_yes_file, "w+")
# data_counter = 0
# ear_vector = []


# sad  完全不准
# 要 s 才是开始 收集  还有 识别

def data_get(ear_vector, data_counter,frame,txt):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # print("rects")
    # print(rects)
    for rect in rects:
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        # points = shape.parts()
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # print('leftEAR = {0}'.format(leftEAR))
        # print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouth_points = points[48:68]  # 设置嘴巴特征点，其实也可以取出
        mouthHull = cv2.convexHull(mouth_points)  # 寻找嘴巴轮廓
        nosewing_points = points[31:36]  # 设置间距鼻孔特征点，其实也可以取出
        # 鼻翼
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        # mouth_points = points[48:68]  # 设置嘴巴特征点，其实也可以取出
        # eyebrow_ear = util.eyebrow_aspect_ratio(shape, rect)  # 眉毛
        # mouth_ear = util.mouth_aspect_ratio(mouth_points)  # 嘴巴
        # mouth_slope_left = util.mouth_slope_left(points)  # 眉毛

        # mouth_slope_left = util.mouth_slope_left(shape)  # 眉毛
        # mouth_slope_right = util.mouth_slope_right(shape)  # 眉毛
        # avg_mouth_slope = abs(mouth_slope_right) + abs(mouth_slope_left)
        # avg_mouth_slope /= 2

        # 伤心的表情
        # avg_mouth_slope_below=util.avg_mouth_slope_below(shape)
        # nose_wing_distance = util.nose_wing_distance(nosewing_points)
        # nose_bridge_distance = util.nose_bridge_distance(nosewing_points)
        # nose_bridge_distance = util.nose_bridge_distance(shape)
        nose_bridge_distance = util.nose_bridge_distance(points)
        # TypeError: 'dlib.full_object_detection' object is not subscriptable
        # shape 是要 part 的

        # ret, ear_vector = queue_in(ear_vector, ear)
        # ret, ear_vector = queue_in(ear_vector, eyebrow_ear)
        # ret, ear_vector = queue_in(ear_vector, mouth_ear)
        # ret, ear_vector = queue_in(ear_vector, avg_mouth_slope)
        # ret, ear_vector = queue_in(ear_vector, avg_mouth_slope_below)
        # ret, ear_vector = queue_in(ear_vector, nose_wing_distance)
        # 也就是说 我们先根据 比如眼睛的 上下距离和左右距离，来算出一个比例，这个比例在眼睛睁大闭眼的情况下，是不同的
        # 然后我们视频摄像头实时的检测，用dlib去实时的获取他的脸部轮廓，去计算这么一个比例
        # 然后把他放到一个列表里面，我们是以连续的几个值作为一个特征量的，所以再把这么一个列表写入到txt里面
        # 这样经过一段时间，我们就有好多个列表了，他们里面放的是 眼睛纵宽比的数值
        ret, ear_vector = queue_in(ear_vector, nose_bridge_distance)
        if (len(ear_vector) == VECTOR_SIZE):
            # print(ear_vector)
            # input_vector = []
            # input_vector.append(ear_vector)

            txt.write(str(ear_vector))
            txt.write('\n')

            data_counter += 1
            print(data_counter)

        # cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(eyebrow_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(mouth_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(avg_mouth_slope), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(avg_mouth_slope_below), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR:{:.2f}".format(nose_bridge_distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        # nose_wing_distance

        # ret, ear_vector = queue_in(ear_vector, ear)
        # if (len(ear_vector) == VECTOR_SIZE):
        #     # print(ear_vector)
        #     # input_vector = []
        #     # input_vector.append(ear_vector)
        #     # print("ear_vector")
        #     # print(ear_vector)
        #     txt.write(str(ear_vector))
        #     txt.write('\n')
        #     data_counter += 1
        #     print(data_counter)
        #
        # cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return data_counter



def change_status():
    key = cv2.waitKey(1)
    if key & 0xFF == ord("s"):
        print('Start collecting images.')
        flag = 1
        return flag
    elif key & 0xFF == ord("e"):
        print('Stop collecting images.')
        flag = 0
        # break
        return flag
    elif key & 0xFF == ord("q"):
        print('quit')
        return

video_win_name = "video"
cv2.namedWindow(video_win_name)

def get_yes_data_from_video():
    # opencv 打开视频 不显示
    print('Prepare to collect images with your eyes open')
    print('Press s to start collecting images.')
    print('Press e to end collecting images.')
    print('Press q to quit')
    # txt = open('train_open2.txt', "w+")
    txt = open(train_yes_file, "w+")
    data_counter = 0
    ear_vector = []


    # video_path = 0
    # video_path = r"G:\FFOutput\digustVID_20211003_120342.mp4"
    # video_path = r"G:\FFOutput\你为什么总腰疼呢？腰疼自测与康复训练 00_01_22-00_03_14.mp4"
    video_path = r"G:\download\emotion_video2021_09_21_17_22_30.avi"
    # 不行
    # video_path=0
    cap = cv2.VideoCapture(video_path)
    # cap.set(3, 640)
    # cap.set(4, 480)

    # flag = 0
    flag = 1
    # opencv 读取 视频 空的
    #  opencv  cap.read()  retval null
    # https://blog.csdn.net/cwcww1314/article/details/51615564/
    # while (1):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            # break
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord("s"):
        #     print('Start collecting images.')
        #     flag = 1
        # elif key & 0xFF == ord("e"):
        #     print('Stop collecting images.')
        #     flag = 0
        #     # break
        # elif key & 0xFF == ord("q"):
        #     print('quit')
        #     break

        # if flag == 1:
        #     data_counter = data_get(ear_vector, data_counter)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rects = detector(gray, 0)
        # for rect in rects:
        #     shape = predictor(gray, rect)
        #     points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        #     # points = shape.parts()
        #     leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
        #     rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
        #     leftEAR = eye_aspect_ratio(leftEye)
        #     rightEAR = eye_aspect_ratio(rightEye)
        #     # print('leftEAR = {0}'.format(leftEAR))
        #     # print('rightEAR = {0}'.format(rightEAR))
        #
        #     ear = (leftEAR + rightEAR) / 2.0
        #
        #     leftEyeHull = cv2.convexHull(leftEye)
        #     rightEyeHull = cv2.convexHull(rightEye)
        #     cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #     cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #
        #     ret, ear_vector = queue_in(ear_vector, ear)
        #     if (len(ear_vector) == VECTOR_SIZE):
        #         # print(ear_vector)
        #         # input_vector = []
        #         # input_vector.append(ear_vector)
        #         # print("ear_vector")
        #         # print(ear_vector)
        #         txt.write(str(ear_vector))
        #         txt.write('\n')
        #         data_counter += 1
        #         print(data_counter)
        #
        #     cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow("frame", frame)
        cv2.imshow(video_win_name, frame)
    txt.close()
    cap.release()

# get_yes_data_from_video()

def get_no_data_from_video():
    print('-' * 40)
    print('Prepare to collect images with your eyes close')
    print('Press s to start collecting images.')
    print('Press e to end collecting images.')
    print('Press q to quit')
    flag = 0
    # txt = open('train_close.txt', 'wb')
    # train_close_file = 'train_close{}.txt'.format(now_time_str)
    # train_close_file = 'train_eye_brow_angry_{}.txt'.format(now_time_str)
    # txt = open('train_close2.txt', 'w+')
    txt = open(train_no_file, 'w+')
    data_counter = 0
    ear_vector = []

    no_video_path = r"G:\FFOutput\normalVID_20211003_120355.mp4"
    # video_path=0
    cap = cv2.VideoCapture(no_video_path)

    while (1):
        ret, frame = cap.read()
        if not ret:
            continue
        key = cv2.waitKey(1)
        if key & 0xFF == ord("s"):
            print('Start collecting images.')
            flag = 1
        elif key & 0xFF == ord("e"):
            print('Stop collecting images.')
            flag = 0
            # break
        elif key & 0xFF == ord("q"):
            print('quit')
            break

        if flag == 1:
            data_counter = data_get(ear_vector, data_counter,frame,txt)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rects = detector(gray, 0)
            # for rect in rects:
            #     shape = predictor(gray, rect)
            #     points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
            #     # points = shape.parts()
            #     leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
            #     rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
            #     leftEAR = eye_aspect_ratio(leftEye)
            #     rightEAR = eye_aspect_ratio(rightEye)
            #     # print('leftEAR = {0}'.format(leftEAR))
            #     # print('rightEAR = {0}'.format(rightEAR))
            #
            #     ear = (leftEAR + rightEAR) / 2.0
            #
            #     leftEyeHull = cv2.convexHull(leftEye)
            #     rightEyeHull = cv2.convexHull(rightEye)
            #     cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            #     cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            #
            #     eyebrow_ear = eyebrow_aspect_ratio(shape, rect)  # 眉毛
            #
            #     # ret, ear_vector = queue_in(ear_vector, ear)
            #     ret, ear_vector = queue_in(ear_vector, eyebrow_ear)
            #     if (len(ear_vector) == VECTOR_SIZE):
            #         # print(ear_vector)
            #         # input_vector = []
            #         # input_vector.append(ear_vector)
            #
            #         txt.write(str(ear_vector))
            #         txt.write('\n')
            #
            #         data_counter += 1
            #         print(data_counter)
            #
            #     # cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     cv2.putText(frame, "EAR:{:.2f}".format(eyebrow_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow("frame", frame)
        cv2.imshow(video_win_name, frame)
    txt.close()

    cap.release()
    cv2.destroyAllWindows()


get_yes_data_from_video()
get_no_data_from_video()
print("train_yes_file", train_yes_file)
print("train_no_file", train_no_file)
print("now_time_str")
print(now_time_str)
