# -*- coding: utf-8 -*-
# @Time    : 2021/9/21 10:10
# @Author  : 喵奇葩
# @FileName: util.py
# @Software: IntelliJ IDEA
import time

from scipy.spatial import distance
import cv2


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


def get_one_eye(eye, img_rgb, img, log, draw=False):
    # flag += 1
    # eye_width,eye_height=get_one_eye_size(eye)
    left_side_eye = eye[0]  # left edge of eye
    right_side_eye = eye[3]  # right edge of eye
    top_side_eye = eye[1]  # top side of eye
    bottom_side_eye = eye[4]  # bottom side of eye
    # 可以当做点 画出来
    # 树莓派不行
    # print ("left_side_eye")
    # print (left_side_eye)
    # print ("right_side_eye")
    # print (right_side_eye)
    # print ("top_side_eye")
    # print (top_side_eye)
    # print ("bottom_side_eye")
    # print (bottom_side_eye)
    # top  0 相减

    # point_size = 1
    # point_color = (0, 0, 255)  # BGR
    # bottom_color = (255, 0, 0)  # BGR
    # thickness = 4  # 可以为 0 、4、8

    # 要画的点的坐标
    # points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]
    # left_side_eye

    # cv2.circle(img_rgb, left_side_eye, point_size, point_color, thickness)
    # cv2.circle(img_rgb, right_side_eye, point_size, point_color, thickness)
    # cv2.circle(img_rgb, top_side_eye, point_size, bottom_color, thickness)
    # cv2.circle(img_rgb, bottom_side_eye, point_size, bottom_color, thickness)
    if draw:
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        bottom_color = (255, 0, 0)  # BGR
        thickness = 4  # 可以为 0 、4、8
        cv2.circle(img, tuple(left_side_eye), point_size, point_color, thickness)
        cv2.circle(img, tuple(right_side_eye), point_size, point_color, thickness)
        cv2.circle(img, tuple(top_side_eye), point_size, bottom_color, thickness)
        cv2.circle(img, tuple(bottom_side_eye), point_size, bottom_color, thickness)

    # for point in points_list:
    #     cv2.circle(img_rgb, point, point_size, point_color, thickness)
    # ————————————————
    # 版权声明：本文为CSDN博主「Igor Sun」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/deflypig/article/details/103081649

    # top bottom_side_eye
    #     roi_eye1 = img_rgb[top_side_eye:bottom_side_eye, left_side_eye:right_side_eye]
    # desired EYE Region(RGB)
    # cv2.imshow("roi_eye1", roi_eye1)

    # calculate height and width of dlib eye keypoints
    eye_width = right_side_eye[0] - left_side_eye[0]
    eye_height = bottom_side_eye[1] - top_side_eye[1]

    # print("eye_width/eye_height")
    # print(eye_width / eye_height)
    # log = Logger('all.log',level='debug')
    log.logger.info("eye_width/eye_height")
    log.logger.info(eye_width / eye_height)
    bi = eye_width / eye_height
    if bi < 5:
        # print("open")
        # log.logger.info("eye_width/eye_height")
        log.logger.info("open")
    else:
        # print("close")
        log.logger.info("close")
        # log.logger.info(eye_width/eye_height)
    # 这个很准确吧
    #     eye_width/eye_height
    # 7.0

    # create bounding box with buffer around keypoints
    eye_x1 = int(left_side_eye[0] - 0 * eye_width)
    eye_x2 = int(right_side_eye[0] + 0 * eye_width)

    eye_y1 = int(top_side_eye[1] - 1 * eye_height)
    eye_y2 = int(bottom_side_eye[1] + 0.75 * eye_height)

    # draw bounding box around eye roi

    # cv2.rectangle(img_rgb,(eye_x1, eye_y1), (eye_x2, eye_y2),(0,255,0),2)

    roi_eye = img_rgb[eye_y1:eye_y2, eye_x1:eye_x2]  # desired EYE Region(RGB)
    # print ("roi_eye")
    # print (roi_eye)

    # if flag == 1:
    #     print ("flag")
    #     print (flag)
    #     # 这里跳出 是为了在外面画图吧
    #     break

    # x = roi_eye.shape
    # row = x[0]
    # col = x[1]
    # # this is the main part,
    # # where you pick RGB values from the area just below pupil
    # array1 = roi_eye[row // 2:(row // 2) + 1, int((col // 3) + 3):int((col // 3)) + 6]
    #
    # try:
    #     array1 = array1[0][2]
    #     array1 = tuple(array1)  # store it in tuple and pass this tuple to "find_color" Funtion
    #
    #     print(find_color(array1))
    # except Exception:
    #     pass
    return roi_eye

    # cv2.imshow("frame"+str(index), roi_eye)


# aspect ratio
def get_one_eye_size(eye):
    # flag += 1
    left_side_eye = eye[0]  # left edge of eye
    right_side_eye = eye[3]  # right edge of eye
    top_side_eye = eye[1]  # top side of eye
    bottom_side_eye = eye[4]  # bottom side of eye
    # 可以当做点 画出来
    # 树莓派不行
    # print ("left_side_eye")
    # print (left_side_eye)
    # print ("right_side_eye")
    # print (right_side_eye)
    # print ("top_side_eye")
    # print (top_side_eye)
    # print ("bottom_side_eye")
    # print (bottom_side_eye)
    # top  0 相减

    # point_size = 1
    # point_color = (0, 0, 255)  # BGR
    # bottom_color = (255, 0, 0)  # BGR
    # thickness = 4  # 可以为 0 、4、8

    # 要画的点的坐标
    # points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]
    # left_side_eye

    # cv2.circle(img_rgb, left_side_eye, point_size, point_color, thickness)
    # cv2.circle(img_rgb, right_side_eye, point_size, point_color, thickness)
    # cv2.circle(img_rgb, top_side_eye, point_size, bottom_color, thickness)
    # cv2.circle(img_rgb, bottom_side_eye, point_size, bottom_color, thickness)

    # for point in points_list:
    #     cv2.circle(img_rgb, point, point_size, point_color, thickness)
    # ————————————————
    # 版权声明：本文为CSDN博主「Igor Sun」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/deflypig/article/details/103081649

    # top bottom_side_eye
    #     roi_eye1 = img_rgb[top_side_eye:bottom_side_eye, left_side_eye:right_side_eye]
    # desired EYE Region(RGB)
    # cv2.imshow("roi_eye1", roi_eye1)

    # calculate height and width of dlib eye keypoints
    eye_width = right_side_eye[0] - left_side_eye[0]
    eye_height = bottom_side_eye[1] - top_side_eye[1]
    return eye_width, eye_height


def get_now_time_str():
    now_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    return now_time_str


import numpy as np


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
        brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
        frown_sum += shape.part(j + 5).x - shape.part(j).x
        line_brow_x.append(shape.part(j).x)
        line_brow_y.append(shape.part(j).y)

    # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y)  # 计算眉毛的倾斜程度
    tempx = np.array(line_brow_x)
    tempy = np.array(line_brow_y)
    z1 = np.polyfit(tempx, tempy, 1)
    eyebrowtilt = -round(z1[0], 3)
    return eyebrowtilt


def avg(lst):
    return sum(lst) / len(lst)


def mouth_aspect_ratio(mouth):  # 计算人嘴巴张开闭合的阈值
    # 垂直点位
    A = np.linalg.norm(mouth[2] - mouth[10])  # 取得嘴巴上50
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


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
    # 这个是 我看图片的 左边 应该是脸的右边才对。。。
    return slope(face, point_lst_left)
# 57 58 --60 嘴下面
# 嘴巴的 下面的两块的 斜率的平均
# https://blog.csdn.net/qq_24946843/article/details/92764528
def avg_mouth_slope(face):

    avg_mouth_slope = abs(mouth_slope_right(face)) + abs(mouth_slope_left(face))
    avg_mouth_slope /= 2
    return avg_mouth_slope

# 鼻子上 4个点 对于鼻子的中间的点 的距离
# 鼻翼
# def nose_aspect_ratio(nose):
def nose_wing_distance(nose):
    # https://blog.csdn.net/weixin_38659482/article/details/85045470
    A = distance.euclidean(nose[0], nose[2])
    B = distance.euclidean(nose[1], nose[2])
    C = distance.euclidean(nose[3], nose[2])
    D = distance.euclidean(nose[4], nose[2])
    E = A + B + C + D
    return E

def nose_bridge_distance(face):
    # return distance.euclidean(face[28], face[34])
    return distance.euclidean(face[29], face[31])

def nose_ar(face,nose):
    return  nose_bridge_distance(face)/nose_wing_distance(nose)

def avg_mouth_slope_below(face):
    point_lst_mouth_below_right = [57, 58, 59, 60]
    point_lst_mouth_below_left = [57, 56, 55, 54]
    avg_mouth_slope = abs(slope(face,point_lst_mouth_below_right)) \
                      + abs(slope(face,point_lst_mouth_below_left))
    avg_mouth_slope /= 2
    return avg_mouth_slope


def avg_mouth_slope_up(face):
    point_lst_mouth_below_right = [52, 51, 50, 49]
    point_lst_mouth_below_left = [52, 53, 54, 55]
    avg_mouth_slope = abs(slope(face,point_lst_mouth_below_right)) \
                      + abs(slope(face,point_lst_mouth_below_left))
    avg_mouth_slope /= 2
    return avg_mouth_slope

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


def wait_esc():
    if cv2.waitKey(5) & 0xFF == 27:
        return True
    return False


# 横纵
def eye_aspect_ratio_hz(eye):
    eye_width, eye_height = get_one_eye_size(eye)
    return eye_width / eye_height
