import time

import cv2
import dlib
import os
import webcolors
from imutils import face_utils


# https://www.thinbug.com/q/45513886
# https://www.thinbug.com/q/45513886
# flag = 0
# detector = dlib.get_frontal_face_detector()
# model_path = "model/shape_predictor_68_face_landmarks.dat"
# predictor = dlib.shape_predictor(model_path)

# img_path="faces/YEzOe.jpg"
# img_path = r"G:\edgeDownload\闭眼图片_百度图片搜索_files\u=2263462917,3288796753&fm=117&fmt=auto&gp=0.jpg"
# img_path = r"G:\edgeDownload\闭眼图片_百度图片搜索_files\u=2135879452,833680737&fm=117&fmt=auto&gp=0.jpg"
# img_path = r"faces/close_eye.jpg"
# img_path = r"faces/close_eye2.jpg"
# img_path = r"faces/cle3.jpg"
# img_path = r"img/u=72080405,1785066593&fm=117&fmt=auto&gp=0.jpg"


# img_dir="img"
# img_dir_lst=os.listdir(img_dir)
# for i in img_dir_lst:
#     abs_path=os.path.join(img_dir,i)


def find_color(requested_colour):  # finds the color name from RGB values

    min_colours = {}
    for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = key
        closest_name = min_colours[min(min_colours.keys())]
    return closest_name


# 有问题 不要了
def get_dlib_faces(img_path):
    # img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print("img none")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    # cap = cv2.VideoCapture(0)             #turns on the webcam

    # (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # # points for left eye and right eye
    # (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # detect dlib face rectangles in the grayscale frame
    dlib_faces = detector(gray, 0)
    # print ("dlib_faces")
    # print (dlib_faces)
    return dlib_faces, img_rgb


def read_imgs():
    img_dir = "img"
    img_dir_lst = os.listdir(img_dir)
    can_detect_lst = []
    for i in img_dir_lst:
        abs_path = os.path.join(img_dir, i)
        # img = cv2.imread(img_path)
        dlib_faces = get_dlib_faces(abs_path)
        if len(dlib_faces) == 0:
            continue
        print("abs_path")
        print(abs_path)
        can_detect_lst.append(abs_path)
        #         img = cv2.imread(abs_path)
        #         if img is None:
        #             print ("img none")
        #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        #
        # # cap = cv2.VideoCapture(0)             #turns on the webcam
        #
        #         (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        #         # points for left eye and right eye
        #         (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        #         gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        #
        #         # detect dlib face rectangles in the grayscale frame
        #         dlib_faces = detector(gray, 0)
        print("dlib_faces")
        print(dlib_faces)
    print("can_detect_lst")
    print(can_detect_lst)


# def get_one_eye():

# def color_one_eye(eye, img_rgb, img, draw=False):
#     # flag += 1
#     left_side_eye = eye[0]  # left edge of eye
#     right_side_eye = eye[3]  # right edge of eye
#     top_side_eye = eye[1]  # top side of eye
#     bottom_side_eye = eye[4]  # bottom side of eye
#     # 可以当做点 画出来
#     # print ("left_side_eye")
#     # print (left_side_eye)
#     # print ("right_side_eye")
#     # print (right_side_eye)
#     # print ("top_side_eye")
#     # print (top_side_eye)
#     # print ("bottom_side_eye")
#     # print (bottom_side_eye)
#     # top  0 相减
#
#     point_size = 1
#     point_color = (0, 0, 255)  # BGR
#     bottom_color = (255, 0, 0)  # BGR
#     thickness = 4  # 可以为 0 、4、8
#
#     # 要画的点的坐标
#     # points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]
#     # left_side_eye
#
#     # cv2.circle(img_rgb, left_side_eye, point_size, point_color, thickness)
#     # cv2.circle(img_rgb, right_side_eye, point_size, point_color, thickness)
#     # cv2.circle(img_rgb, top_side_eye, point_size, bottom_color, thickness)
#     # cv2.circle(img_rgb, bottom_side_eye, point_size, bottom_color, thickness)
#     if draw:
#         cv2.circle(img, left_side_eye, point_size, point_color, thickness)
#         cv2.circle(img, right_side_eye, point_size, point_color, thickness)
#         cv2.circle(img, top_side_eye, point_size, bottom_color, thickness)
#         cv2.circle(img, bottom_side_eye, point_size, bottom_color, thickness)
#
#     # for point in points_list:
#     #     cv2.circle(img_rgb, point, point_size, point_color, thickness)
#     # ————————————————
#     # 版权声明：本文为CSDN博主「Igor Sun」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
#     # 原文链接：https://blog.csdn.net/deflypig/article/details/103081649
#
#     # top bottom_side_eye
#     #     roi_eye1 = img_rgb[top_side_eye:bottom_side_eye, left_side_eye:right_side_eye]
#     # desired EYE Region(RGB)
#     # cv2.imshow("roi_eye1", roi_eye1)
#
#     # calculate height and width of dlib eye keypoints
#     eye_width = right_side_eye[0] - left_side_eye[0]
#     eye_height = bottom_side_eye[1] - top_side_eye[1]
#     print("eye_width/eye_height")
#     print(eye_width / eye_height)
#     # log = Logger('all.log',level='debug')
#     log.logger.info("eye_width/eye_height")
#     log.logger.info(eye_width / eye_height)
#     bi = eye_width / eye_height
#     if bi < 5:
#         print("open")
#         # log.logger.info("eye_width/eye_height")
#         log.logger.info("open")
#     else:
#         print("close")
#         log.logger.info("close")
#         # log.logger.info(eye_width/eye_height)
#     # 这个很准确吧
#     #     eye_width/eye_height
#     # 7.0
#
#     # create bounding box with buffer around keypoints
#     eye_x1 = int(left_side_eye[0] - 0 * eye_width)
#     eye_x2 = int(right_side_eye[0] + 0 * eye_width)
#
#     eye_y1 = int(top_side_eye[1] - 1 * eye_height)
#     eye_y2 = int(bottom_side_eye[1] + 0.75 * eye_height)
#
#     # draw bounding box around eye roi
#
#     # cv2.rectangle(img_rgb,(eye_x1, eye_y1), (eye_x2, eye_y2),(0,255,0),2)
#
#     roi_eye = img_rgb[eye_y1:eye_y2, eye_x1:eye_x2]  # desired EYE Region(RGB)
#     # print ("roi_eye")
#     # print (roi_eye)
#
#     # if flag == 1:
#     #     print ("flag")
#     #     print (flag)
#     #     # 这里跳出 是为了在外面画图吧
#     #     break
#
#     x = roi_eye.shape
#     row = x[0]
#     col = x[1]
#     # this is the main part,
#     # where you pick RGB values from the area just below pupil
#     array1 = roi_eye[row // 2:(row // 2) + 1, int((col // 3) + 3):int((col // 3)) + 6]
#
#     try:
#         array1 = array1[0][2]
#         array1 = tuple(array1)  # store it in tuple and pass this tuple to "find_color" Funtion
#
#         print(find_color(array1))
#     except Exception:
#         pass
#     return roi_eye

# cv2.imshow("frame"+str(index), roi_eye)


def get_one_eye(eye, img_rgb, img, draw=False):
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
    if draw:
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        bottom_color = (255, 0, 0)  # BGR
        thickness = 4  # 可以为 0 、4、8
        cv2.circle(img, tuple(left_side_eye), point_size, point_color, thickness)
        cv2.circle(img, tuple(right_side_eye), point_size, point_color, thickness)
        cv2.circle(img,tuple( top_side_eye), point_size, bottom_color, thickness)
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
    print("eye_width/eye_height")
    print(eye_width / eye_height)
    # log = Logger('all.log',level='debug')
    log.logger.info("eye_width/eye_height")
    log.logger.info(eye_width / eye_height)
    bi = eye_width / eye_height
    if bi < 5:
        print("open")
        # log.logger.info("eye_width/eye_height")
        log.logger.info("open")
    else:
        print("close")
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


def get_eye_color(roi_eye):
    x = roi_eye.shape
    row = x[0]
    col = x[1]
    # this is the main part,
    # where you pick RGB values from the area just below pupil
    array1 = roi_eye[row // 2:(row // 2) + 1, int((col // 3) + 3):int((col // 3)) + 6]

    try:
        array1 = array1[0][2]
        array1 = tuple(array1)  # store it in tuple and pass this tuple to "find_color" Funtion

        print(find_color(array1))
    except Exception:
        pass


def get_eyes(face, img_rgb, predictor, gray):
    eyes = []  # store 2 eyes

    # convert dlib rect to a bounding box
    (x, y, w, h) = face_utils.rect_to_bb(face)
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)  # draws blue box over face

    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # points for left eye and right eye
    (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[left_Start:left_End]
    # indexes for left eye key points

    rightEye = shape[right_Start:right_End]

    eyes.append(leftEye)  # wrap in a list
    eyes.append(rightEye)
    # print ("eyes")
    # print (eyes)
    print("len(eyes)")
    print(len(eyes))
    # 两只眼睛
    return eyes


def eye_color_one_img(img, detector, predictor):
    # flag = 0
    # img = cv2.imread(img_path)
    # print (img)
    # cv2.imshow("img origin", img)
    if img is None:
        print("img none")
        return
    # cv2.imshow("img origin", img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    # cap = cv2.VideoCapture(0)             #turns on the webcam

    # (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # # points for left eye and right eye
    # (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # ret, frame=cap.read()
    # frame = cv2.flip(frame, 1)

    # cv2.imshow(winname='face',mat=frame)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # detect dlib face rectangles in the grayscale frame
    dlib_faces = detector(gray, 0)
    # print ("dlib_faces")
    # print (dlib_faces)
    # 注释结束
    for face in dlib_faces:
        # eyes = []  # store 2 eyes
        #
        # # convert dlib rect to a bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(face)
        # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)  # draws blue box over face
        #
        # shape = predictor(gray, face)
        # shape = face_utils.shape_to_np(shape)
        #
        # leftEye = shape[left_Start:left_End]
        # # indexes for left eye key points
        #
        # rightEye = shape[right_Start:right_End]
        #
        # eyes.append(leftEye)  # wrap in a list
        # eyes.append(rightEye)
        # # print ("eyes")
        # # print (eyes)
        # print ("len(eyes)")
        # print (len(eyes))
        # 两只眼睛
        eyes = get_eyes(face, img_rgb, predictor, gray)
        for index, eye in enumerate(eyes):
            # roi_eye = color_one_eye(eye, img_rgb)
            # roi_eye = color_one_eye(eye, img_rgb, img)
            roi_eye = get_one_eye(eye, img_rgb, img, draw=True)
            get_eye_color(roi_eye)
            # cv2.imshow("frame" + str(index), roi_eye)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def eye_color():
    # flag = 0
    detector = dlib.get_frontal_face_detector()
    model_path = "model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model_path)
    # 模型都是这个 为什么感觉这个很准确

    can_detect_lst = ['img\\u=1793907243,3840715764&fm=117&fmt=auto&gp=0.jpg',
                      'img\\u=2135879452,833680737&fm=117&fmt=auto&gp=0.jpg',
                      'img\\u=2750830245,1102268003&fm=117&fmt=auto&gp=0.jpg',
                      'img\\u=3172036868,1694666136&fm=117&fmt=auto&gp=0.jpg',
                      'img\\u=3565677290,3058826256&fm=117&fmt=auto&gp=0.jpg']
    # img_path = can_detect_lst[0]
    # # 失败  没有显示
    for i in can_detect_lst:
        img = cv2.imread(i)
        eye_color_one_img(img, detector, predictor)
    # img_path="faces/obama.jpg"
    # # 没有
    # img_path = "faces/YEzOe.jpg"
    # eye_color_one_img(img_path, detector, predictor)

    # dlib_faces, img_rgb = get_dlib_faces(img_path)
    # (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # # points for left eye and right eye
    # (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # 是因为中文读不出吗
    # img= cv2.imread('blue2.jpg')
    # 注释开始
    # img = cv2.imread(img_path)
    # # print (img)
    # # cv2.imshow("img origin", img)
    # if img is None:
    #     print ("img none")
    #     return
    # cv2.imshow("img origin", img)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    #
    # # cap = cv2.VideoCapture(0)             #turns on the webcam
    #
    # (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # # points for left eye and right eye
    # (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    #
    # # ret, frame=cap.read()
    # # frame = cv2.flip(frame, 1)
    #
    # # cv2.imshow(winname='face',mat=frame)
    #
    # gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    #
    # # detect dlib face rectangles in the grayscale frame
    # dlib_faces = detector(gray, 0)
    # print ("dlib_faces")
    # print (dlib_faces)
    # # 注释结束
    # for face in dlib_faces:
    #     eyes = []  # store 2 eyes
    #
    #     # convert dlib rect to a bounding box
    #     (x, y, w, h) = face_utils.rect_to_bb(face)
    #     cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)  # draws blue box over face
    #
    #     shape = predictor(gray, face)
    #     shape = face_utils.shape_to_np(shape)
    #
    #     leftEye = shape[left_Start:left_End]
    #     # indexes for left eye key points
    #
    #     rightEye = shape[right_Start:right_End]
    #
    #     eyes.append(leftEye)  # wrap in a list
    #     eyes.append(rightEye)
    #     # print ("eyes")
    #     # print (eyes)
    #     print ("len(eyes)")
    #     print (len(eyes))
    #     # 两只眼睛
    #     for index, eye in enumerate(eyes):
    #         flag += 1
    #         left_side_eye = eye[0]  # left edge of eye
    #         right_side_eye = eye[3]  # right edge of eye
    #         top_side_eye = eye[1]  # top side of eye
    #         bottom_side_eye = eye[4]  # bottom side of eye
    #
    #         # calculate height and width of dlib eye keypoints
    #         eye_width = right_side_eye[0] - left_side_eye[0]
    #         eye_height = bottom_side_eye[1] - top_side_eye[1]
    #
    #         # create bounding box with buffer around keypoints
    #         eye_x1 = int(left_side_eye[0] - 0 * eye_width)
    #         eye_x2 = int(right_side_eye[0] + 0 * eye_width)
    #
    #         eye_y1 = int(top_side_eye[1] - 1 * eye_height)
    #         eye_y2 = int(bottom_side_eye[1] + 0.75 * eye_height)
    #
    #         # draw bounding box around eye roi
    #
    #         # cv2.rectangle(img_rgb,(eye_x1, eye_y1), (eye_x2, eye_y2),(0,255,0),2)
    #
    #         roi_eye = img_rgb[eye_y1:eye_y2, eye_x1:eye_x2]  # desired EYE Region(RGB)
    #         # print ("roi_eye")
    #         # print (roi_eye)
    #
    #         if flag == 1:
    #             print ("flag")
    #             print (flag)
    #             break
    #
    #         x = roi_eye.shape
    #         row = x[0]
    #         col = x[1]
    #         # this is the main part,
    #         # where you pick RGB values from the area just below pupil
    #         array1 = roi_eye[row // 2:(row // 2) + 1, int((col // 3) + 3):int((col // 3)) + 6]
    #
    #         array1 = array1[0][2]
    #         array1 = tuple(array1)  # store it in tuple and pass this tuple to "find_color" Funtion
    #
    #         print(find_color(array1))
    #
    #         cv2.imshow("frame", roi_eye)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break


def video(FLAGS):
    # file_path=r"G:\录屏\睁开bandicam 2021-09-21 10-39-34-271.mp4"
    # file_path = r"E:\edgeDownload\玛 丽·萝 丝.flv"


    if FLAGS.video_file_path=="0":
        file_path = 0
    else:
        file_path = FLAGS.video_file_path

    # file_path = "G:\录屏\睁眼bandicam 2021-09-21 11-21-45-917.mp4"
    # file_path = r"G:\file\phone\VID_20210921_113737.mp4"
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(file_path)
    detector = dlib.get_frontal_face_detector()
    model_path = "model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model_path)
    # 模型都是这个 为什么感觉这个很准确
    # 戴不戴眼镜 相差太大
    # 录屏打开很卡 是因为分辨率太高吗
    # 5.6 close
    while (1):
        success, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
        # or not img
        if not success:
            # print ("danger")
            continue
        # img = cv2.imread(img_path)
        eye_color_one_img(img, detector, predictor)
        cv2.imshow("Frame", img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# read_imgs()
# eye_color()
from logger import Logger

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_file_path", required=True, type=str, help="Absolute path to video path.")
# "G:\FFOutput\手指肌腱断裂康复训练记录Day1&2 00_00_10-00_01_22.flv"

# parser.add_argument(
#     "--log", required=False, type=str, help="Absolute path to video path.",default="log/all.log")
FLAGS, unparsed = parser.parse_known_args()
# hand_detect(FLAGS)


# log_path = "log/mqp_vi.log"
# log_path = "log/mqp_vi2.log"
# log_path = "log/malilusi.log"
# log_path = "log/mqp3.log"
# log_path = "log/mqp4.log"
now_time_str=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
# log_path = FLAGS.log
# --log  log/mqp5.log
# log_path = "log/"+now_time_str+".log"
log_path = "log/%s.log"%now_time_str
# 笔记本摄像头 用 5 作为阈值是很不错的
# 因为不太清晰
# log = Logger('all.log', level='debug')
log = Logger(log_path, level='debug')
# log = Logger('all.log',level='info')
# log.logger.info("1")


video(FLAGS)
