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
# from blink_detect_svm import RIGHT_EYE_END
from NoFaceWarnUtil import NoFaceWarnUtil
from logger import Logger


def no_face_warn(start_time, rects, img, should_warn):
    now_time = time.time()
    # print ("now_time")
    # print (now_time)
    # print ("now_time-start_time")
    # print (now_time - start_time)
    # warn_time = 1
    # 每隔一秒 检查一下人在不在
    # if now_time - start_time >= warn_time:
    # 不是这个逻辑，应该是 每次过了 3s 就测试一次
    # print(start_time,start_time)
    # print(now_time,now_time)
    time_gap = now_time - start_time
    # print("time_gap",time_gap)

    if time_gap >= commons.warn_time:
        should_warn = True
        # check_danger(rects)
        # if len(rects) == 0:
        #     print("danger")
        #     cv2.putText(img, "TIRED!!Warnning", (150, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     # if use_beep:
        #     #     beep.beep_of(1)
        #     beep_sec(1)
        # 不是这个时候 才能叫  是这个时候 改变了状态
        print("start_time", start_time)
        start_time = now_time

    if len(rects) == 0:
        if should_warn:
            print("danger")
            cv2.putText(img, "TIRED!!Warnning", (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # if use_beep:
            #     beep.beep_of(1)
            beep_sec(1)
            should_warn = False
    return start_time, should_warn


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

def eye_aspect_ratio_from_face(points):
    leftEye = points[commons.LEFT_EYE_START:commons.LEFT_EYE_END + 1]  # 取出左眼特征点

    rightEye = points[commons.RIGHT_EYE_START:commons.RIGHT_EYE_END + 1]  # 取出右眼特征点
    leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR阈值
    rightEAR = eye_aspect_ratio(rightEye)  # 计算左眼EAR阈值
    ear = (leftEAR + rightEAR) / 2.0
    return ear


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


def close_eye_cal(count, frame_counter, img, svm_eye, ear, blink_counter):
    # draw_characteristic_point(img,shape)
    # for i in range(68):
    # 	cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
    # eye_aspect_ratio_from_face
    count += 1
    # for pt in leftEye:
    # 	pt_pos = (pt[0], pt[1])
    # 	cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

    # for pt in rightEye:
    # 	pt_pos = (pt[0], pt[1])
    # 	cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    # 如果少于的次数 > 某个数字，就是说他闭眼的时间太多了
    # close_eye_long_frame_cnt=10

    # close_eye_long_frame_cnt = 25
    # if frame_counter >= close_eye_long_frame_cnt:
    # disguts 会 当作闭眼的时间长 因为 眼睛有点 变小了
    # 如果他闭眼的帧数超过这个值，那么就说明他闭眼的时间太长了
    if frame_counter >= commons.close_eye_long_frame_cnt:
        print("frame_counter")
        print(frame_counter)
        print("danger")
        # 闭眼的时间太长了
        cv2.putText(img, "TIRED!!Warnning", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # if use_beep:
        #     beep.beep_of(1)
        beep_sec(1)

    # 注释开始
    # if ear < EYE_AR_THRESH:
    #     frame_counter += 1
    # elif ear >= eye_open_thresh:
    #     # else:
    #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
    #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
    #     print("frame_counter")
    #     print(frame_counter)
    #     # 连续三帧感觉条件太苛刻
    #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
    #         if count < frame_cnt_blink_refresh:
    #             blink_counter += 1
    #     frame_counter = 0
    # 注释 结束

    # svm.blink_detect_svm(ea)
    # svm_eye.blink_detect_svm()
    res_eye_close = svm_eye.blink_detect_svm(ear)
    # 准确
    frame_counter, blink_counter = eye_cnt(res_eye_close, frame_counter,
                                           commons.EYE_AR_CONSEC_FRAMES, count,
                                           commons.frame_cnt_blink_refresh,
                                           blink_counter)
    # 注释开始
    # if res_eye_close == 1:
    #     frame_counter += 1
    # else:
    #     # else:
    #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
    #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
    #     print("frame_counter")
    #     print(frame_counter)
    #     # 连续三帧感觉条件太苛刻
    #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
    #         if count < frame_cnt_blink_refresh:
    #             blink_counter += 1
    #     frame_counter = 0
    # 注释结束

    # earhz_thresh = 4.5
    # # 树莓派的摄像头的话 ，就 不用这个了，ear 好像也挺好的
    # # 小的是睁眼 3多
    # # log.info("earhz: %f"%earhz)
    # # log.info(earhz)
    # if earhz >= earhz_thresh:
    #     frame_counter += 1
    # else:
    #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
    #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
    #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
    #         if count < frame_cnt_blink_refresh:
    #             blink_counter += 1
    #     frame_counter = 0

    # if mouth_ear > commons.MAR_THRESH:
    #     mCOUNT += 1
    # else:
    #     if mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
    #         mTOTAL += 1
    #     mCOUNT = 0

    if count >= commons.frame_cnt_blink_refresh:
        count = 0
        blink_counter = 0
    # if blink_counter >= 10:
    #     detect_danger=1
    # if detect_danger == 1
    # 一秒钟大概25帧
    # 3秒眨眼5次 就当作他困了
    # 这样 跟 嘴巴有什么关系
    if blink_counter >= commons.blink_cnt_sleep:
        cv2.putText(img, "TIRED!!Warnning", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("TIRED!!Warnning")
        # beep.beep_of(1)
        beep_sec(1)
    # if blink_counter >= 10:
    #     cv2.putText(img, "TIRED!!Warnning".format(ear), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
    #                 2)
    #     print ("TIRED!!Warnning")
    else:
        cv2.putText(img, "Blinks:{0}".format(blink_counter),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(img, "MOUTH{0}".format(mTOTAL), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("Blinks:{0}".format(blink_counter))
        print("EAR:{:.2f}".format(ear))
        # print("MOUTH{0}".format(mTOTAL))
    return count, frame_counter, blink_counter

def check_face_rects(rects,bundle):
    # 好多变量 放在函数里不好吧
    if noFaceWarnUtil.no_face_warn(rects):
        print("danger")
        cv2.putText(img, "TIRED!!Warnning", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # if use_beep:
        #     beep.beep_of(1)
        beep_sec(1)



# 视频 是可以放的 ，是因为没有初始化吗？
# beep.setup(beep.Buzzer)
def emotion_detect(log, file_path):
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
    noFaceWarnUtil = NoFaceWarnUtil()
    # start_time = time.time()
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
    # should_warn = True
    # svm_path = "train/ear_svm2021_09_21_19_18_05.m"
    svm_path = "train/close_eye_2021_10_05_17_31_42.m"
    svm_eye = svm.Svm(svm_path)
    svm_path_angry = "train/train_eye_brow_angry_2021_09_22_08_53_41.m"
    svm_mouth_amazing_path = "train/train_eye_amazing_2021_09_22_09_17_24.m"
    svm_mouth_sad_path = "train/train_mouth_sad_2021_10_01_22_27_18.m"
    # svm_mouth_below_sad_path = "train/train_mouth_sad_2021_10_02_11_12_46.m"
    svm_mouth_below_sad_path = "train/mouth_up_sad_2021_10_05_20_45_55.m"
    # 搞错了，这里弄得是嘴巴
    # svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_11_20_49.m"
    # svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_11_31_39.m"
    # svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_13_13_59.m"
    # svm_nose_bridge_distance_disgust_path = "train/nose_bridge_distance_disgust_2021_10_03_15_22_24.m"
    svm_nose_bridge_distance_disgust_path = "train/disgust_nose_ar_2021_10_05_17_48_49.m"

    # clf = joblib.load(svm_path)
    # clf_angry = joblib.load(svm_path_angry)
    svm_angry = svm.Svm(svm_path_angry)
    svm_mouth_amazing = svm.Svm(svm_mouth_amazing_path)
    # svm_mouth_sad = svm.Svm(svm_mouth_sad_path)
    svm_mouth_below_sad = svm.Svm(svm_mouth_below_sad_path)
    svm_nose_bridge_distance_disgust = svm.Svm(svm_nose_bridge_distance_disgust_path)
    print("start read")
    while (1):
        # processBar.control()

        # if loop_flag == pos:
        #     loop_flag = loop_flag + 1
        #     cv2.setTrackbarPos('time', 'video', loop_flag)
        #     # 这句话是让视频随着进度条动，进度条也随着数字动
        # else:
        #     pos = cv2.getTrackbarPos('time', 'video')
        #     loop_flag = pos
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        # #    这个不写的话 调整进度条就没用了
        ret, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
        # or not img
        # print("read")
        if not ret:
            # print("no ret")
            # print ("danger")
            # continue
            # break
            if file_path == "0":
                # print("video continue")
                continue
            else:
                print("break")
                break
        if commons.flip:
            img = cv2.flip(img, 1)  # 水平翻转图像
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 把读取到Img转换为2进制灰度图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 0)  # 调用检测器人脸检测
        # print ("detector")
        # print (detector)
        # 识别不出来 树莓派 dlib
        # faces = face_cascade.detectMultiScale(gray, 1.3, 2)  # 调用级联分类器对gray进行人脸识别并且返回一个矩形列表
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print ("set up")
        # 没有人脸
        # print ("rects")
        # print (rects)
        # print ("img")
        # print (img)
        # 注意摄像头不要拿倒过来了,摄像头是在上面的
        # print ("rects")
        # print (rects)
        # 注释开始
        # for k, d in enumerate(rects):
        # # for idx, content in enumerate(rects):
        #     # print("第", k + 1, "个人脸d的坐标：",
        #     # 	  "left:", d.left(),
        #     # 	  "right:", d.right(),
        #     # 	  "top:", d.top(),
        #     # 	  "bottom:", d.bottom())
        #     width = d.right() - d.left()
        #     heigth = d.bottom() - d.top()
        #     # print ("d",d)、
        # 注释结束
        # print ("rects")
        # print (rects)
        # print (len(rects))
        #         rectangles[]
        # 0
        # cnt_helper=Cnt()
        if noFaceWarnUtil.no_face_warn(rects):
            print("danger")
            cv2.putText(img, "TIRED!!Warnning", (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # if use_beep:
            #     beep.beep_of(1)
            beep_sec(1)

        # start_time, should_warn = no_face_warn(start_time, rects, img, should_warn)
        #
        # now_time = time.time()
        # # print ("now_time")
        # # print (now_time)
        # # print ("now_time-start_time")
        # # print (now_time - start_time)
        # # warn_time = 1
        # # 每隔一秒 检查一下人在不在
        # # if now_time - start_time >= warn_time:
        # # 不是这个逻辑，应该是 每次过了 3s 就测试一次
        # # print(start_time,start_time)
        # # print(now_time,now_time)
        # time_gap = now_time - start_time
        # # print("time_gap",time_gap)
        #
        # if time_gap >= commons.warn_time:
        #     should_warn = True
        #     # check_danger(rects)
        #     # if len(rects) == 0:
        #     #     print("danger")
        #     #     cv2.putText(img, "TIRED!!Warnning", (150, 30),
        #     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     #     # if use_beep:
        #     #     #     beep.beep_of(1)
        #     #     beep_sec(1)
        #     # 不是这个时候 才能叫  是这个时候 改变了状态
        #     print("start_time", start_time)
        #     start_time = now_time
        #
        # if len(rects) == 0:
        #     if should_warn:
        #         print("danger")
        #         cv2.putText(img, "TIRED!!Warnning", (150, 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #         # if use_beep:
        #         #     beep.beep_of(1)
        #         beep_sec(1)
        #         should_warn = False

        for rect in rects:
            # print ("have pic")
            # print('-'*20)
            # landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(im_rd, faces[i]).parts()])
            shape = predictor(gray, rect)  # 返回68个人脸特征点的位置
            # 在一张图上 框出一个脸
            # __call__(self: dlib.shape_predictor, image: array, box: dlib.rectangle)
            points = face_utils.shape_to_np(shape)  # 将facial landmark (x, y)转换为NumPy 数组
            # points = shape.parts()
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼特征点

            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼特征点
            # util.get_one_eye(leftEye, img_rgb, img, log)
            # util.get_one_eye(rightEye, img_rgb, img, log)

            earhz_l = util.eye_aspect_ratio_hz(leftEye)
            earhz_r = util.eye_aspect_ratio_hz(rightEye)
            earhz = (earhz_l + earhz_r) / 2.0

            mouth_points = points[48:68]  # 设置嘴巴特征点，其实也可以取出
            # 48+14==62 , +4==66
            rmeimao_points = points[17:22]  # 设置右眉毛特征点，其实也可以取出
            lmeimao_points = points[22:27]  # 设置左眉毛特征点，其实也可以取出
            nose1_points = points[27:31]  # 设置鼻梁间距特征点，其实也可以取出
            # 鼻梁
            # bridge of the nose  ,nose_bridge

            nose2_points = points[31:36]  # 设置间距鼻孔特征点，其实也可以取出
            # 鼻翼
            eye_points = points[36:48]
            leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR阈值
            rightEAR = eye_aspect_ratio(rightEye)  # 计算左眼EAR阈值
            mouth_ear = mouth_aspect_ratio(mouth_points)  # 计算嘴巴EAR阈值
            # print ("d")
            # print (d)
            # eyebrow_ear = eyebrow_aspect_ratio(shape,d)  # 眉毛
            eyebrow_ear = eyebrow_aspect_ratio(shape, rect)  # 眉毛
            nose_ear = nose_aspect_ratio(nose2_points)
            pain_ear1 = pain_aspect_ratio1(eye_points)
            pain_ear2 = pain_aspect_ratio2(mouth_points)
            happy_ear = happy_aspect_ratio(mouth_points)
            # 用嘴巴这里的点 反而不清楚了
            cv2.circle(img, tuple(mouth_points[14]), 2, (0, 255, 0), -1, 8)
            cv2.circle(img, tuple(mouth_points[18]), 2, (0, 255, 0), -1, 8)

            # A = np.linalg.norm(eye[1] - eye[8])
            # B = np.linalg.norm(eye[5] - eye[10])
            DeepPink = (147, 255, 20)
            cv2.circle(img, tuple(eye_points[1]), 2, DeepPink, -1, 8)
            cv2.circle(img, tuple(eye_points[8]), 2, DeepPink, -1, 8)
            # opencv 颜色
            # BRG
            # 255,20,147  rgb
            #
            # cv2.
            # mouth[14] - mouth[18]
            # print('leftEAR = {0}'.format(leftEAR))
            # print('rightEAR = {0}'.format(rightEAR))
            # print('openMouth = {0}'.format(mouth_ear))
            # print('eyeborwtilt = {0}'.format(eyebrow_ear))
            # print('happy_ear = {0}'.format(happy_ear))
            # print('noselength ={0}'.format(nose_ear))
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)  # 寻找左眼轮廓
            rightEyeHull = cv2.convexHull(rightEye)  # 寻找右眼轮廓
            mouthHull = cv2.convexHull(mouth_points)  # 寻找嘴巴轮廓

            lmeimaoHull = cv2.convexHull(lmeimao_points)  # 寻找左眉毛轮廓
            rmeimaoHull = cv2.convexHull(rmeimao_points)  # 寻找左眉毛轮廓
            nose1Hull = cv2.convexHull(nose1_points)  # 寻找鼻梁轮廓
            nose2Hull = cv2.convexHull(nose2_points)  # 寻找鼻孔轮廓

            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓
            cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)  # 绘制嘴巴轮廓
            # cv2.drawContours(img, [lmeimaoHull], -1, (0, 255, 0), 2)  # 绘制眉毛轮廓
            # cv2.drawContours(img, [rmeimaoHull], -1, (0, 255, 0), 2)  # 绘制眉毛轮廓
            # cv2.drawContours(img, [nose1Hull], -1, (0, 255, 0), 2)  # 绘制鼻梁轮廓
            # cv2.drawContours(img, [nose2Hull], -1, (0, 255, 0), 2)  # 绘制鼻孔轮廓

            count, frame_counter, blink_counter = close_eye_cal(
                count, frame_counter, img, svm_eye, ear, blink_counter)
            # draw_characteristic_point(img,shape)
            # for i in range(68):
            # 	cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
            # 开始注释 [2]
            # count += 1
            # # for pt in leftEye:
            # # 	pt_pos = (pt[0], pt[1])
            # # 	cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
            #
            # # for pt in rightEye:
            # # 	pt_pos = (pt[0], pt[1])
            # # 	cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
            # # 如果少于的次数 > 某个数字，就是说他闭眼的时间太多了
            # # close_eye_long_frame_cnt=10
            #
            # # close_eye_long_frame_cnt = 25
            # # if frame_counter >= close_eye_long_frame_cnt:
            # # disguts 会 当作闭眼的时间长 因为 眼睛有点 变小了
            # if frame_counter >= commons.close_eye_long_frame_cnt:
            #     print("frame_counter")
            #     print(frame_counter)
            #     print("danger")
            #     # 闭眼的时间太长了
            #     cv2.putText(img, "TIRED!!Warnning", (150, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     # if use_beep:
            #     #     beep.beep_of(1)
            #     beep_sec(1)
            #
            # # 注释开始
            # # if ear < EYE_AR_THRESH:
            # #     frame_counter += 1
            # # elif ear >= eye_open_thresh:
            # #     # else:
            # #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
            # #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
            # #     print("frame_counter")
            # #     print(frame_counter)
            # #     # 连续三帧感觉条件太苛刻
            # #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
            # #         if count < frame_cnt_blink_refresh:
            # #             blink_counter += 1
            # #     frame_counter = 0
            # # 注释 结束
            #
            # # svm.blink_detect_svm(ea)
            # # svm_eye.blink_detect_svm()
            # res_eye_close = svm_eye.blink_detect_svm(ear)
            # # 准确
            # frame_counter, blink_counter = eye_cnt(res_eye_close, frame_counter,
            #                                        EYE_AR_CONSEC_FRAMES, count,
            #                                        frame_cnt_blink_refresh, blink_counter)
            # # 注释开始
            # # if res_eye_close == 1:
            # #     frame_counter += 1
            # # else:
            # #     # else:
            # #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
            # #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
            # #     print("frame_counter")
            # #     print(frame_counter)
            # #     # 连续三帧感觉条件太苛刻
            # #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
            # #         if count < frame_cnt_blink_refresh:
            # #             blink_counter += 1
            # #     frame_counter = 0
            # # 注释结束
            #
            # # earhz_thresh = 4.5
            # # # 树莓派的摄像头的话 ，就 不用这个了，ear 好像也挺好的
            # # # 小的是睁眼 3多
            # # # log.info("earhz: %f"%earhz)
            # # # log.info(earhz)
            # # if earhz >= earhz_thresh:
            # #     frame_counter += 1
            # # else:
            # #     # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，
            # #     # 才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
            # #     if frame_counter >= EYE_AR_CONSEC_FRAMES:
            # #         if count < frame_cnt_blink_refresh:
            # #             blink_counter += 1
            # #     frame_counter = 0
            #
            # if mouth_ear > MAR_THRESH:
            #     mCOUNT += 1
            # else:
            #     if mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
            #         mTOTAL += 1
            #     mCOUNT = 0
            #
            # if count >= frame_cnt_blink_refresh:
            #     count = 0
            #     blink_counter = 0
            # # if blink_counter >= 10:
            # #     detect_danger=1
            # # if detect_danger == 1
            # # 一秒钟大概25帧
            # # 3秒眨眼5次 就当作他困了
            # if blink_counter >= blink_cnt_sleep:
            #     cv2.putText(img, "TIRED!!Warnning".format(ear), (150, 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     print("TIRED!!Warnning")
            #     # beep.beep_of(1)
            #     beep_sec(1)
            # # if blink_counter >= 10:
            # #     cv2.putText(img, "TIRED!!Warnning".format(ear), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
            # #                 2)
            # #     print ("TIRED!!Warnning")
            # else:
            #     cv2.putText(img, "Blinks:{0}".format(blink_counter),
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     cv2.putText(img, "MOUTH{0}".format(mTOTAL), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     print("Blinks:{0}".format(blink_counter))
            #     print("EAR:{:.2f}".format(ear))
            #     print("MOUTH{0}".format(mTOTAL))
            # 结束注释[2]

            # 嘴巴闭合阈值，大于0.5认为张开嘴巴
            # 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
            # 睁开嘴巴 睁开 眼睛就算是 AMAZING
            amazing_yes = amazing_predict(mouth_ear, mCOUNT, ear, svm_mouth_amazing)
            if amazing_yes == 1:
                cv2.putText(img, "AMAZING", (rect.left(), rect.bottom() + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("AMAZING")

                # 注释开始
            # if mouth_ear > MAR_THRESH and mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
            #     if ear > EYE_AR_THRESH:
            #         cv2.putText(img, "AMAZING", (rect.left(), rect.bottom() + 20),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("AMAZING")
            # 注释结束
            # 伤心的表情 可以根据嘴巴 往下撇嘴

            # 如果闭上嘴巴，可能是生气或者正常
            # 0.03 好像更加不准了
            # 0.02
            # 0.01 特别不准

            angry_yes = angry_predict(svm_angry, eyebrow_ear, mouth_ear, MAR_THRESH, True)
            if angry_yes == 1:
                cv2.putText(img, "angry", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2, 4)
                # amazing  经常出现

            # 开始注释
            # angry_predict=svm_angry.detect_svm(eyebrow_ear)
            # if angry_predict==1:
            #     cv2.putText(img, "angry", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
            #                 2, 4)

            # angry_limit = 0.02
            # angry_limit = 0.2
            # # if mouth_ear < MAR_THRESH and eyebrow_ear < 0.03:
            # print("eyebrow_ear")
            # print(eyebrow_ear)
            # if mouth_ear < MAR_THRESH and eyebrow_ear < angry_limit:
            #     # cv2.putText(img, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
            #     cv2.putText(img, "angry", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
            #                 2, 4)
            #     print("angry")
            # 结束注释

            # else:
            # 	cv2.putText(img, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
            # nature 也打印出来的话 就挺烦的
            # 不过 angry 还是不准
            print("happy_ear")
            print(happy_ear)
            # 正常是 3--4，张开嘴是 5以上
            if happy_ear < 5:
                if ear < 0.18:
                    if pain_ear2 > pain_ear1 * 3 / 4:
                        cv2.putText(img, "Pain", (rect.left(), rect.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        print("Pain")
            if happy_ear > 5:
                if ear > 0.2:
                    if pain_ear2 > pain_ear1 * 4 / 5:
                        cv2.putText(img, "Happy", (rect.left(), rect.bottom() + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255),
                                    2)
                        print("Happy")
            # cv2.imshow("Frame", img)

            #     svm_mouth_sad_ear=util.avg_mouth_slope(shape)
            #     # util.mouth_slope_left()
            #     res_mouth_sad=svm_mouth_sad.detect_svm(svm_mouth_sad_ear)
            #     if res_mouth_sad==1:
            #         cv2.putText(img, "sad", (rect.left(), rect.bottom() + 20),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #                     (0, 0, 255),2)
            #         print("sad")

            # avg_mouth_slope_below = util.avg_mouth_slope_below(shape)
            # avg_mouth_slope_below = util.avg_mouth_slope_below(shape)
            # 斜坡;
            avg_mouth_slope_up = util.avg_mouth_slope_up(shape)
            # util.mouth_slope_left()
            # sad 有点准确了
            # 文档：伤心的 嘴 的 斜率 好像没什么意义.not...
            # 链接：http://note.youdao.com/noteshare?id=824a590c16441c7931c0d3b0601740e1&sub=0F1DEFF24BC7485FAB5C62BD4E886755
            #             res_mouth_below_sad = svm_mouth_below_sad.detect_svm(avg_mouth_slope_below)
            res_mouth_below_sad = svm_mouth_below_sad.detect_svm(avg_mouth_slope_up)
            if res_mouth_below_sad == 1:
                cv2.putText(img, "sad", (rect.left(), rect.bottom() + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                print("sad")
            # disgust 完全不准
            #             文档：鼻翼  厌恶.note
            # 链接：http://note.youdao.com/noteshare?id=33d34d85b2999b155feea377ee998ef0&sub=BA4B7EEC05494B868DD64F6270FBB934
            #             nose_bridge_distance=util.nose_bridge_distance(points)
            #             nose_bridge_distance=util.nose_ar(points,nose1_points)
            nose_bridge_distance = util.nose_ar(points, nose2_points)
            res_nose_disgust = svm_nose_bridge_distance_disgust.detect_svm(nose_bridge_distance)
            if res_nose_disgust == 1:
                cv2.putText(img, "disgust", (rect.right(), rect.bottom() + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                print("disgust")

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
now_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
# log_path = FLAGS.log
# --log  log/mqp5.log
# log_path = "log/"+now_time_str+".log"
log_path = "log/%s.log" % now_time_str
# print("log_path", log_path)
# 笔记本摄像头 用 5 作为阈值是很不错的
# 因为不太清晰
# log = Logger('all.log', level='debug')
# log = Logger(log_path, level='debug')

import argparse

# 一旦识别不到眼睛了就报警 有点夸张了
#
#
# 离开 远一点 可以找到 人
# video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"
# video_path = r"G:\FFOutput\digustVID_20211003_120342.mp4"
# 可以 不过会当作 生气
# video_path = r"G:\FFOutput\diguestVID_20211003_150659.mp4"
# 可以 不过会当作 生气

# video_path =  r"G:\FFOutput\disgust_no_glass_VID_20211003_151356.mp4"
# 识别不出

# video_path =  r"G:\FFOutput\normalVID_20211003_144749.mp4"
# 可以识别 就是正常
# video_path = r"G:\FFOutput\fear_no_glassVID_20211003_151540.mp4"

# fear 不行

# video_path = r"G:\FFOutput\sleep_VID_20211007_201227.mp4"
# 没有显示 危险？
# video_path = r"G:\FFOutput\sleep_beep_VID_20211007_202046.mp4"
# 无效
# video_path = r"G:\FFOutput\sleep_VID_20211007_203000.mp4"
# video_path = 0

camera="0"
video_path = camera
# 闭眼 没用了
# video_path = r"G:\download\眨眼video2021_09_21_14_49_57.avi"
# 视频不行

# video_path = r"G:\download\emotion_video2021_09_21_17_22_30.avi"

# 害怕 被当作 amzing  因为 嘴巴张开
# happy 没有吗

# video_path = r"G:\emotion\sad_VID_20211005_161024.mp4"

# sad 嘴巴 跑到 鼻子上去了

# video_path = r"G:\FFOutput\sad_VID_20211005_201110.mp4"

# sad 不准确

# video_path = r"G:\FFOutput\sad_VID_20211005_203129.mp4"

# 嘴巴下面轮廓还是 圆形的
# 可能要下巴抬起来?

# video_path = r"G:\FFOutput\sad_VID_20211005_203129.mp4"

# 头低下的
# video_path = r"G:\FFOutput\sad_many_pos_VID_20211005_203557.mp4"
# 下巴不抬起来 就没有用处
# 换了 m 之后 更加 不准确了

# video_path = "0"

# video_path = r"G:\FFOutput\disgust_no_glass_VID_20211003_151356.mp4"
# 这个视频 也识别不出了
# 会当作 生气 甚至 还有sad

# video_path = r"G:\emotion\disgust_VID_20211005_161143.mp4"

# 其他的视频 来显示 disgust_  貌似也是没问题
# disgust_ 闭眼的 问题好像没有出现了
# video_path = r"G:\emotion\angry_far_VID_20211005_161930.mp4"

# 生气 和 恶心 经常是一起出现的 ,因为 恶心 我是当作 皱鼻子就是了,但是生气也会皱鼻子
# video_path = r"G:\emotion\angry_far_VID_20211005_161930.mp4"
# video_path = r"G:\FFOutput\close_eye_VID_20211005_172828.mp4"
# 会是 恶心
# video_path = r"G:\FFOutput\normalVID_20211003_120355.mp4"
# 太近了

# video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"
# 没有 disguts
# video_path = r"G:\FFOutput\amazing_far_VID_20211005_162639.mp4"

# 鼻梁 只是 长度的 话,只要人 远离了 就会有问题,需要是比例
# 远离 的 恶心 问题 好像没有
# 真正的 disgust 也识别不出来了
# 也有 disguest 也有 sad

# video_path = r"G:\project\pythonProj\dlib_test\face\gitignore\sad_test_bandicam 2021-10-06 16-08-53-098.mp4"
# 这是大的视频
# video_path = r"G:\FFOutput\sad_VID_20211005_203112.mp4"
# 会 当作闭眼睛啊
# video_path = r"G:\FFOutput\amazing_far_VID_20211005_162639.mp4"

# disgust 现在 还是距离 不是比例吗
# video_path = r"G:\download\眨眼video2021_09_21_14_49_57.avi"
# 这视频 不行 打不卡了
# 眨眼 好像有点准确 又有点不准
# video_path = r"G:\FFOutput\blink_VID_20211007_155744.mp4"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_file_path", required=False, type=str,
    help="Absolute path to video path.", default=video_path)
# "G:\FFOutput\手指肌腱断裂康复训练记录Day1&2 00_00_10-00_01_22.flv"
FLAGS, unparsed = parser.parse_known_args()
# hand_detect(FLAGS)
print("emotion_detect start")
# 训练 dlib
# emotion_detect(log, FLAGS.video_file_path)
emotion_detect(None, FLAGS.video_file_path)
if use_beep:
    beep.destroy()
