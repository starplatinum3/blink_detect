# author：郭昆仑
# coding=utf-8
import joblib
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
import time
# import torch

import commons
import util
# from data_collect import train_yes_file
from logger import Logger

import commons

import util

now_time_str = util.get_now_time_str()
# time_str = "2021_10_03_13_13_59"
time_str = now_time_str
# mode = "no"
# train_file_prefix = "train_eye_brow_angry_"
# 首先配置类型
train_file_prefix = commons.train_type
# train_yes_file='train_yes_{}.txt'.format(now_time_str)
train_yes_file = 'train/{}_yes_{}.txt'.format(train_file_prefix, time_str)
# train_no_file = train_file_prefix + '_no_{}.txt'.format(now_time_str)
train_no_file = 'train/{}_no_{}.txt'.format(train_file_prefix, time_str)


def angry_predict(svm_angry, eyebrow_ear, mouth_ear, MAR_THRESH, use_svm=True):
    if use_svm:
        return svm_angry.detect_svm(eyebrow_ear)
        # if angry_predict==1:
        #     cv2.putText(img, "angry", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
        #                 2, 4)

    # angry_limit = 0.02
    angry_limit = 0.2
    # if mouth_ear < MAR_THRESH and eyebrow_ear < 0.03:
    print("eyebrow_ear")
    print(eyebrow_ear)
    if mouth_ear < MAR_THRESH and eyebrow_ear < angry_limit:
        return 1
        # cv2.putText(img, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
        # cv2.putText(img, "angry", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
        #             2, 4)
        # print("angry")
    return 0


def amazing_predict(mouth_ear, mCOUNT, ear, svm_mouth_amazing=None):
    # 嘴巴闭合阈值，大于0.5认为张开嘴巴
    # 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
    # 睁开嘴巴 睁开 眼睛就算是 AMAZING

    # if mouth_ear >commons. MAR_THRESH and mCOUNT >=commons. MOUTH_AR_CONSEC_FRAMES:
    #     if ear > commons.EYE_AR_THRESH:
    #         cv2.putText(img, "AMAZING", (rect.left(), rect.bottom() + 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #         print("AMAZING")
    if svm_mouth_amazing is None:
        if mouth_ear > commons.MAR_THRESH and mCOUNT >= commons.MOUTH_AR_CONSEC_FRAMES:
            if ear > commons.EYE_AR_THRESH:
                return 1
                # cv2.putText(img, "AMAZING", (rect.left(), rect.bottom() + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # print("AMAZING")
        return 0
    return svm_mouth_amazing.detect_svm(mouth_ear)
    # if mouth_ear > MAR_THRESH and mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
    #     if ear > EYE_AR_THRESH:
    #         cv2.putText(img, "AMAZING", (rect.left(), rect.bottom() + 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #         print("AMAZING")
    # 如果闭上嘴巴，可能是生气或者正常
    # 0.03 好像更加不准了
    # 0.02
    # 0.01 特别不准


def eyebrow_aspect_ratio(shape, d):
    # 创立眉毛x坐标和y坐标列表
    line_brow_x = []
    line_brow_y = []
    # cap.isOpened（） 返回true/false 检查初始化是否成功
    # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
    brow_sum = 0  # 高度之和
    frown_sum = 0  # 两边眉毛距离之和
    # 以矩形框上线为横轴，高度之和即每一个坐标与矩形框左上角坐标的纵坐标之差的求和
    # 长度之和即每一个坐标与矩形框左上角坐标的横坐标之差的求和，所以最后所求斜率需要转换一下。
    for j in range(17, 21):  # 17到21即为左边眉毛的五个点的坐标
        # d 是眉毛的框子
        brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
        frown_sum += shape.part(j + 5).x - shape.part(j).x
        # 17+5==22 是另外一个眉毛的最里面
        line_brow_x.append(shape.part(j).x)
        line_brow_y.append(shape.part(j).y)

    # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y)  # 计算眉毛的倾斜程度
    tempx = np.array(line_brow_x)
    tempy = np.array(line_brow_y)
    z1 = np.polyfit(tempx, tempy, 1)
    eyebrowtilt = -round(z1[0], 3)
    # 保留 3位
    # https://www.runoob.com/python/func-number-round.html
    return eyebrowtilt


# 斜率
def slope(face, points):
    # 创立嘴巴 x坐标和y坐标列表
    line_x = []
    line_y = []
    # 62 63 64  54  一个 斜线的 斜率
    # for i in range(62, 21):  # 17到21即为左边眉毛的五个点的坐标
    # point_lst = [62, 63, 64, 54]
    # point_lst = [62, 61, 60, 48]
    for i in points:
        line_x.append(face.part(i).x)
        line_y.append(face.part(i).y)
        # line_mouth_x.append(face.part(i).x)
        # line_mouth_x.append(face.part(62).x)
        # line_mouth_x.append(face.part(62).x)
    tempx = np.array(line_x)
    tempy = np.array(line_y)
    z1 = np.polyfit(tempx, tempy, 1)
    tilt = -round(z1[0], 3)
    return tilt


def mouth_slope_left(face):
    # point_lst = [62, 63, 64, 54]
    point_lst_left = [62, 61, 60, 48]
    return slope(face, point_lst_left)


# https://blog.csdn.net/qq_24946843/article/details/92764528
def mouth_slope_right(face):
    point_lst = [62, 63, 64, 54]
    return slope(face, point_lst)
    # 创立嘴巴 x坐标和y坐标列表
    # line_mouth_x = []
    # line_mouth_y = []
    # # 62 63 64  54  一个 斜线的 斜率
    # # for i in range(62, 21):  # 17到21即为左边眉毛的五个点的坐标
    # point_lst = [62, 63, 64, 54]
    # point_lst = [62, 61, 60, 48]
    # for i in point_lst:
    #     line_mouth_x.append(face.part(i).x)
    #     line_mouth_y.append(face.part(i).y)
    #     # line_mouth_x.append(face.part(i).x)
    #     # line_mouth_x.append(face.part(62).x)
    #     # line_mouth_x.append(face.part(62).x)
    # tempx = np.array(line_mouth_x)
    # tempy = np.array(line_mouth_y)
    # z1 = np.polyfit(tempx, tempy, 1)
    # tilt= -round(z1[0], 3)
    # return tilt


# https://www.5axxw.com/questions/content/fypm99
# 计算EAR
def eye_aspect_ratio(eye):
    # print(eye)计算阈值函数
    A = distance.euclidean(eye[1], eye[5])  # A,B是计算两组垂直眼睛标志的距离，而C是计算水平眼睛标志之间的距离
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    # ear = (A + B) / (3.0 * C)
    return ear


# 鼻子上 4个点 对于鼻子的中间的点 的距离
# 鼻翼
def nose_aspect_ratio(nose):
    # https://blog.csdn.net/weixin_38659482/article/details/85045470
    A = distance.euclidean(nose[0], nose[2])
    B = distance.euclidean(nose[1], nose[2])
    C = distance.euclidean(nose[3], nose[2])
    D = distance.euclidean(nose[4], nose[2])
    E = A + B + C + D
    return E


def mouth_aspect_ratio(mouth):  # 计算人嘴巴张开闭合的阈值
    # 垂直点位
    A = np.linalg.norm(mouth[2] - mouth[10])  # 取得嘴巴上50
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


def pain_aspect_ratio1(eye):
    A = np.linalg.norm(eye[1] - eye[8])
    B = np.linalg.norm(eye[5] - eye[10])
    return A


def pain_aspect_ratio2(mouth):
    A = np.linalg.norm(mouth[0] - mouth[6])
    return A


# dlib 点
def happy_aspect_ratio(mouth):
    # c2.
    A = np.linalg.norm(mouth[14] - mouth[18])
    return A


# def emotion_recognition():
# 	# #如果张开嘴巴，可能是惊讶或者开心，
# 	# if mouth_ear>MAR_THRESH and mCOUNT>=MOUTH_AR_CONSEC_FRAMES:
# 	# 	if ear>EYE_AR_THRESH:
# 	# 		cv2.putText(img, "AMAZING{0}", (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# 	# #如果闭上嘴巴，可能是生气或者正常
# 	# else:
# 	# 	if ear>EYE_AR_THRESH and eyebrow_ear<0.1:
# 	# 		cv2.putText(img, "ANGRY{0}", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#      #


# https://blog.csdn.net/qq_29750461/article/details/103466269
class Config():
    def __init__(self):
        # self.face_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        # self.eye_cascade_path=cv2.data.haarcascades + 'haarcascade_eye.xml'

        self.face_cascade_path = "model/" + 'haarcascade_frontalface_default.xml'
        self.eye_cascade_path = 'model/haarcascade_eye.xml'

        # self.shape_detector_path='model/haarcascade_eye.xml'


# config = Config()
# 导入opnecv的级联分类器
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 从这里注释
# face_cascade = cv2.CascadeClassifier(config.face_cascade_path)
# # smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
# eye_cascade = cv2.CascadeClassifier(config.eye_cascade_path)
#
# pwd = os.getcwd()  # 获取当前路径
# model_path = os.path.join(pwd, 'model')  # 模型文件夹路径
# shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测模型路径
#
# detector = dlib.get_frontal_face_detector()  # 人脸检测器
# predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器
#
# EYE_AR_THRESH = 0.3  # EAR判断阈值，默认0.3，如果大于0.3则认为眼睛是睁开的；小于0.3则认为眼睛是闭上的
# EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
#
# # 嘴巴闭合阈值，大于0.5认为张开嘴巴
# # 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
# MAR_THRESH = 0.5
# MOUTH_AR_CONSEC_FRAMES = 3
# mCOUNT = 0
# mTOTAL = 0
# # 对应特征点的序号
# RIGHT_EYE_START = 37 - 1
# RIGHT_EYE_END = 42 - 1
# LEFT_EYE_START = 43 - 1
# LEFT_EYE_END = 48 - 1
#
# frame_counter = 0
# blink_counter = 0  # 眨眼计数
# count = 0
# # detect_danger = 0
# detect_danger = 1
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# show_it = False
# 注释结束

def draw_characteristic_point(img, shape):
    for i in range(68):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)


# 记数的麻烦，传递的不是引用
from Cnt import Cnt


# 太多数字有问题
# def eye_cnt(earhz, frame_counter, EYE_AR_CONSEC_FRAMES, count, frame_cnt_blink_refresh, blink_counter):
#     earhz_thresh = 4.5
#     # 小的是睁眼 3多
#     if earhz >= earhz_thresh:
#         frame_counter += 1
#     else:
#         # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
#         # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
#         if frame_counter >= EYE_AR_CONSEC_FRAMES:
#             if count < frame_cnt_blink_refresh:
#                 blink_counter += 1
#         frame_counter = 0


# 这是 视频， 有好几针 有cnt 所以不能单独图片

# def emotion_img():

def nothing(emp):
    pass


# def process_bar():

from processBar import ProcessBar
import svm

use_beep = True
try:
    import beep
except ModuleNotFoundError:
    use_beep = False


def beep_sec(sec):
    if use_beep:
        beep.beep_of(sec)


def eye_cnt(res_eye_close, frame_counter, EYE_AR_CONSEC_FRAMES,
            count, frame_cnt_blink_refresh, blink_counter):
    if res_eye_close == 1:
        frame_counter += 1
    else:
        # else:
        # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
        # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
        print("frame_counter")
        print(frame_counter)
        # 连续三帧感觉条件太苛刻
        if frame_counter >= EYE_AR_CONSEC_FRAMES:
            if count < frame_cnt_blink_refresh:
                blink_counter += 1
        frame_counter = 0
        # count 不会更新 不用返回
    return frame_counter, blink_counter


if use_beep:
    beep.setup(beep.Buzzer)


def data_get(ear_vector, data_counter, frame, txt, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # print("rects")
    # print(rects)
    for rect in rects:
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        # points = shape.parts()
        leftEye = points[commons.LEFT_EYE_START:commons.LEFT_EYE_END + 1]
        rightEye = points[commons.RIGHT_EYE_START:commons.RIGHT_EYE_END + 1]
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
        # # nose_bridge_distance = util.nose_bridge_distance(nosewing_points)
        # # nose_bridge_distance = util.nose_bridge_distance(shape)
        # nose_bridge_distance = util.nose_bridge_distance(points)
        # TypeError: 'dlib.full_object_detection' object is not subscriptable
        # shape 是要 part 的
        # eye_aspect_ratio()
        # nose_ar=util.nose_ar(points,nosewing_points)
        # avg_mouth_slope_up=util.avg_mouth_slope_up(points)
        avg_mouth_slope_up = util.avg_mouth_slope_up(shape)

        # ret, ear_vector = queue_in(ear_vector, ear)
        # ret, ear_vector = queue_in(ear_vector, eyebrow_ear)
        # ret, ear_vector = queue_in(ear_vector, mouth_ear)
        # ret, ear_vector = queue_in(ear_vector, avg_mouth_slope)
        # ret, ear_vector = queue_in(ear_vector, avg_mouth_slope_below)
        # ret, ear_vector = queue_in(ear_vector, nose_wing_distance)
        # ret, ear_vector = svm.queue_in(ear_vector, nose_bridge_distance)
        # ret, ear_vector = svm.queue_in(ear_vector, ear)
        # ret, ear_vector = svm.queue_in(ear_vector, nose_ar)
        ret, ear_vector = svm.queue_in(ear_vector, avg_mouth_slope_up)
        # 之前就是因为 眼睛 睁大了 他也识别不出 所以卡住了
        # 输入的 也不是整数啊  不是 0 1
        # svm 多参数
        if (len(ear_vector) == commons.VECTOR_SIZE):
            # print(ear_vector)
            # input_vector = []
            # input_vector.append(ear_vector)

            txt.write(str(ear_vector))
            txt.write('\n')

            data_counter += 1
            # print(data_counter)

        # cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(eyebrow_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(mouth_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(avg_mouth_slope), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(avg_mouth_slope_below), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR:{:.2f}".format(nose_bridge_distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return data_counter


# 视频 是可以放的 ，是因为没有初始化吗？
# beep.setup(beep.Buzzer)
def data_collect_from_video(log, file_path, mode):
    # 导入opnecv的级联分类器
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # # smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # svm_path = "train/ear_svm2021_09_21_19_18_05.m"
    # clf = joblib.load(svm_path)

    config = Config()
    face_cascade = cv2.CascadeClassifier(config.face_cascade_path)
    # smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(config.eye_cascade_path)

    pwd = os.getcwd()  # 获取当前路径
    model_path = os.path.join(pwd, 'model')  # 模型文件夹路径
    shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测模型路径

    detector = dlib.get_frontal_face_detector()  # 人脸检测器
    predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器

    # EYE_AR_THRESH = 0.3
    # EYE_AR_THRESH = 0.25
    # EYE_AR_THRESH = 0.2
    EYE_AR_THRESH = 0.3

    # eye_open_thresh = 0.28
    # eye_open_thresh = 0.3
    # eye_open_thresh = 0.27
    eye_open_thresh = 0.3
    # 树莓派清晰 大一点
    # eye_open_thresh = 0.25
    # 感觉闭上眼睛 ear 也不会小 反而好像有点大
    # eye_open_thresh = 0.2
    # EYE_AR_THRESH = 0.15
    # EAR判断阈值，默认0.3，如果大于0.3则认为眼睛是睁开的；小于0.3则认为眼睛是闭上的
    EYE_AR_CONSEC_FRAMES = 3
    # EYE_AR_CONSEC_FRAMES = 1
    # EYE_AR_CONSEC_FRAMES = 2
    # glass = True
    glass = False
    if glass:
        EYE_AR_THRESH = 0.2
        eye_open_thresh = 0.25
        # 低下头之后就比较不准了
    # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
    # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
    # EYE_AR_CONSEC_FRAMES=2
    # EYE_AR_CONSEC_FRAMES = 1

    # 嘴巴闭合阈值，大于0.5认为张开嘴巴
    # 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
    MAR_THRESH = 0.5
    MOUTH_AR_CONSEC_FRAMES = 3
    # MOUTH_AR_CONSEC_FRAMES = 2
    mCOUNT = 0
    mTOTAL = 0
    # 对应特征点的序号
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END = 42 - 1
    LEFT_EYE_START = 43 - 1
    LEFT_EYE_END = 48 - 1

    frame_counter = 0
    blink_counter = 0  # 眨眼计数
    # 一段时间没有眨眼过10次，就要置为0 吧
    count = 0
    detect_danger = 0
    # detect_danger = 1
    if file_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)
    cap.set(3, 640)
    cap.set(4, 480)
    show_it = False

    # blink_cnt_sleep = 5
    blink_cnt_sleep = 4
    fps_guess = 25
    frame_cnt_blink_refresh = fps_guess * 3
    video_win_name = "video"
    cv2.namedWindow(video_win_name)
    print("start")
    start_time = time.time()
    # https://blog.csdn.net/m0_37606112/article/details/79590012
    processBar = ProcessBar(cap, video_win_name)
    # frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # loop_flag = 0
    # pos = 0
    # cv2.createTrackbar('time', 'video', 0, frames, nothing)
    happy_ear_lst = []
    ear_lst = []
    mouth_ear_lst = []
    pain_ear1_lst = []
    pain_ear2_lst = []
    # nose_bridge_distance_disgust_yes_2021_10_03_13_13_59.txt
    svm_path = "train/ear_svm2021_09_21_19_18_05.m"
    svm_eye = svm.Svm(svm_path)
    svm_path_angry = "train/train_eye_brow_angry_2021_09_22_08_53_41.m"
    svm_mouth_amazing_path = "train/train_eye_amazing_2021_09_22_09_17_24.m"
    svm_mouth_sad_path = "train/train_mouth_sad_2021_10_01_22_27_18.m"
    svm_mouth_below_sad_path = "train/train_mouth_sad_2021_10_02_11_12_46.m"
    # 搞错了，这里弄得是嘴巴
    # svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_11_20_49.m"
    svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_11_31_39.m"

    data_counter = 0
    ear_vector = []
    if mode == "yes":
        txt = open(train_yes_file, 'w+')
    elif mode == "no":
        txt = open(train_no_file, 'w+')
    while (1):
   
        ret, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
        # or not img
        if not ret:
            # print ("danger")
            # continue
            break

        # if commons.flip:
        #     img = cv2.flip(img, 1)  # 水平翻转图像

        data_counter = data_get(ear_vector, data_counter, img, txt, detector, predictor)
      

        cv2.imshow(video_win_name, img)
        if util.wait_esc():
            break
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # 应该是跑起来了 ，但是没有显示屏
    cap.release()
    cv2.destroyAllWindows()


# 生气的没有
beep_sec(1)
# if use_beep:
#     beep_sec(1)
# do_log = False
#
# if do_log:

# now_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

# log_path = FLAGS.log
# --log  log/mqp5.log
# log_path = "log/"+now_time_str+".log"
now_time_str = util.get_now_time_str()
log_path = "log/data_collect2_%s.log" % now_time_str
print("log_path", log_path)
# 笔记本摄像头 用 5 作为阈值是很不错的
# 因为不太清晰
# log = Logger('all.log', level='debug')
# log = Logger(log_path, level='debug')

import argparse

# 一旦识别不到眼睛了就报警 有点夸张了
#
#

# video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"
# video_path = r"G:\FFOutput\diguestVID_20211003_150659.mp4"

# yes_video_path = r"G:\FFOutput\disgust_no_glass_VID_20211003_151356.mp4"
# no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"

# yes_video_path = r"G:\FFOutput\fear_no_glassVID_20211003_151540.mp4"
# no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"


# yes_video_path = r"G:\FFOutput\close_eye_VID_20211005_172828.mp4"
# no_video_path = r"G:\FFOutput\open_eye_VID_20211005_172529.mp4"

# yes_video_path = r"G:\FFOutput\sad_VID_20211005_201110.mp4"
# no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"

# yes_video_path = r"G:\FFOutput\sad_many_pos_VID_20211005_203557.mp4"
# no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"

# yes_video_path = r"G:\FFOutput\sad_VID_20211006_155448.mp4"
# no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"

yes_video_path = r"G:\FFOutput\disgust_no_glass_VID_20211003_151356.mp4"
no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--video_file_path", required=False, type=str, help="Absolute path to video path.",
#     default=video_path)
# # "G:\FFOutput\手指肌腱断裂康复训练记录Day1&2 00_00_10-00_01_22.flv"
# FLAGS, unparsed = parser.parse_known_args()
# hand_detect(FLAGS)
print("emotion_detect start")
# "G:\FFOutput\normalVID_20211003_120355.mp4"
# mode="yes"
# data_collect_from_video(log, yes_video_path, mode)
# mode="no"
# # data_collect_from_video(log, FLAGS.video_file_path, mode)
# data_collect_from_video(log, no_video_path, mode)

mode = "yes"
data_collect_from_video(None, yes_video_path, mode)
mode = "no"
# data_collect_from_video(log, FLAGS.video_file_path, mode)
data_collect_from_video(None, no_video_path, mode)

# if use_beep:
#     beep.destroy()

print("time_str", time_str)


import numpy as np
from sklearn import svm

import pandas as pd

# from sklearn.externals import joblib
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
import util

# time_str = "2021_09_21_19_09_06"
# tarin_str= "train_open2021_09_21_19_18_05.txt"
# time_str = "2021_09_21_19_18_05"
# time_str = "2021_09_21_22_01_12"
# time_str = "2021_09_21_22_01_12"
# time_str = "2021_09_22_08_48_21"
# time_str = "2021_09_22_08_53_41"
# time_str = "2021_09_22_09_17_24"
# time_str = "2021_10_01_22_27_18"
# time_str = "2021_10_02_10_44_20"
# time_str = "2021_10_02_10_49_42"
# time_str = "2021_10_02_11_12_46"
# time_str = "2021_10_03_11_06_37"
# time_str = "2021_10_03_11_20_49"
# time_str = "2021_10_03_11_31_39"
# time_str = "2021_10_03_13_13_59"
# time_str = "2021_10_03_15_22_24"
# time_str = "2021_10_03_17_01_07"
# time_str = "2021_10_05_17_31_42"
# time_str = "2021_10_05_17_48_49"
# time_str = "2021_10_05_20_16_42"
# time_str = "2021_10_05_20_45_55"
# time_str= "2022_02_26_09_37_27"
# time_str= "2022_02_26_10_03_23"
time_str= now_time_str
# train_mouth_sad_yes_2021_10_01_22_27_18.txt
# now_time_str = util.get_now_time_str()
# train_yes_path = 'train/train_open{}.txt'.format(time_str)
# train_no_path = 'train/train_close{}.txt'.format(time_str)

# train_yes_path = 'train/train_open{}.txt'.format(time_str)
# train_no_path = 'train/train_close{}.txt'.format(time_str)
import commons
prefix="train/"
train_yes_path =prefix+ commons.train_type + '_yes_{}.txt'.format(time_str)
train_no_path =prefix+ commons.train_type + '_no_{}.txt'.format(time_str)
# svm_file = "train/ear_svm{}.m".format(time_str)
svm_file = "train/{}_{}.m".format(commons.train_type, time_str)

data_dic = {"yes": [], "no": []}
# https://www.gairuo.com/p/pandas-plot-scatter
# df = pd.DataFrame(data_dic)

# train_open_path='train_open.txt'
# train_close_path='train_close.txt'

# train_open_txt = open(train_yes_path, 'r')
# train_close_txt = open(train_no_path, 'r')

# train_open_txt = open('train_open.txt', 'rb')
# train_close_txt = open('train_close.txt', 'rb')

train = []
labels = []

# 在这里打了 标签 是 0 1
def read_file(train, labels, file_path,label=1):
    # train = []
    # labels = []
    train_txt = open(file_path, 'r')
    train_vars=[]

    # print('Reading train_open.txt...')
    line_ctr = 0
    for txt_str in train_txt.readlines():
        temp = []
        # print(txt_str)
        datas = txt_str.strip()
        datas = datas.replace('[', '')
        datas = datas.replace(']', '')
        datas = datas.split(',')
        print(datas)
        # 一行的 datas
        for data in datas:
            # print(data)
            data = float(data)
            temp.append(data)
            # 每一行是一个列表
        # print(temp)
        train.append(temp)
        # train 是二维的
        train_vars.append(temp)
        # labels.append(0)
        labels.append(label)
        # 所有的数字作为训练
    train_txt.close()
    return train_vars


def push_train_vars(train, data_dic, key):
    vars = []
    for i in train:
        # avg_var=util.avg(i)
        vars.append(util.avg(i))

    data_dic[key] = vars
    # data_dic["yes"] = yes_vars


print('Reading train_open.txt...')
train_vars=read_file(train, labels, train_yes_path,1)

push_train_vars(train_vars, data_dic, "yes")

# data = float(txt_str)

# if line_ctr <= 12:
# 	line_ctr += 1
# 	temp.append(data)
# elif line_ctr == 13:
# 	# print(temp)
# 	# print(len(temp))
# 	train.append(temp)
# 	labels.append(0)
# 	temp = []
# 	line_ctr = 1
# 	temp.append(data)

print('Reading train_close.txt...')
train_vars=read_file(train, labels, train_no_path,0)
# line_ctr = 0
# temp = []
# for txt_str in train_close_txt.readlines():
#     temp = []
#     # print(txt_str)
#     datas = txt_str.strip()
#     datas = datas.replace('[', '')
#     datas = datas.replace(']', '')
#     datas = datas.split(',')
#     print(datas)
#     for data in datas:
#         # print(data)
#         data = float(data)
#         temp.append(data)
#         # 所有数字放进temp
#     # print(temp)
#     # train 就两个列表，不是的
#     train.append(temp)
#     labels.append(1)

# data = float(txt_str)

# if line_ctr <= 12:
# 	line_ctr += 1
# 	temp.append(data)
# elif line_ctr == 13:
# 	# print(temp)
# 	# print(len(stemp))
# 	train.append(temp)
# 	labels.append(1)
# 	temp = []
# 	line_ctr = 1
# 	temp.append(data)

push_train_vars(train_vars, data_dic, "no")
# 这里就有问题了
# 因为是引用 所以会变的
print("data_dic")
print(data_dic)

def cut_lst(lst1, lst2):
    len_1 = len(lst1)
    len_2 = len(lst2)
    if len_1 < len_2:
        # 3 取得 ：3 得到 0,1
        # 赋值  要返回
        lst2 = lst2[:len_1]
    elif len_1 > len_2:
        lst1 = lst1[:len_2]
    return lst1, lst2


data_dic["yes"], data_dic["no"] = cut_lst(data_dic["yes"], data_dic["no"])

import matplotlib.pyplot as plt
# https://blog.csdn.net/qq_27825451/article/details/83057541
def show(data_dic):
    df = pd.DataFrame(data_dic)
    print("df")
    print(df)
    # df.plot.scatter(x='')
    df.plot.bar()
    plt.title(svm_file)

    plt.show()

    # plt 名字
# ['0.123', ' 0.111', ' 0.098']
# ['0.155', ' 0.187', ' 0.187']

def start_train(train,labels):
    for i in range(len(labels)):
        print("{0} --> {1}".format(train[i], labels[i]))
    # 多维的数据 压缩为1维度的而且 独特性还要高
    # train_close_txt.close()
    # train_open_txt.close()

    print(train)
    print(labels)
    clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
    clf.fit(train, labels)
    # svm_file = "train/ear_svm{}.m".format(time_str)
    # svm_file = "ear_svm{}.m".format(now_time_str)
    # joblib.dump(clf, "ear_svm.m")
    joblib.dump(clf, svm_file)
    print("dump at",svm_file)

    # print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]]')
    # res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]])
    # print(res)

    # print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]]')
    # res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]])
    # print(res)

    # print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]]')
    # res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]])
    # print(res)

    # print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]]')
    # res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]])
    # print(res)

    print('predicting [[0.34, 0.34, 0.31]]')
    res = clf.predict([[0.34, 0.34, 0.31]])
    print(res)

    print('predicting [[0.19, 0.18, 0.18]]')
    res = clf.predict([[0.19, 0.18, 0.18]])
    print(res)


# print('predicting [[0.34]]')
# res = clf.predict([[0.34]])
# print(res)

# print('predicting [[0.19]]')
# res = clf.predict([[0.19]])
# print(res)


def test_cut_lst():
    # [1, 1, 1, 1, 1] [1, 1, 1, 2, 2]
    lst1 = [1, 1, 1, 1, 1]
    lst2 = [1, 1, 1, 2, 2, 2, 2, 2]
    lst1, lst2 = cut_lst(lst1, lst2)
    print(lst1, lst2)

# 眼睛睁大了 框也没有增大 dlib
show(data_dic)
start_train(train,labels)
# https://zhuanlan.zhihu.com/p/361835285
# 悲伤