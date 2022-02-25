# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:47:53 2018
@author: kuangyongjian
svm_train.py
"""
# @Time    : 2021/10/4 19:13
# @Author  : 喵奇葩
# @FileName: dlib_svm.py
# @Software: IntelliJ IDEA
# https://blog.csdn.net/yongjiankuang/article/details/79808346
import joblib
import numpy as np
import cv2
import dlib
# from scipy.spatial import distance
# import os
# from imutils import face_utils
# import time
# import torch

import commons


# import util
# # from blink_detect_svm import RIGHT_EYE_END
# from logger import Logger

# import numpy as np
# import cv2 as cv2

def svm_predict(svm, testData):
    res = svm.predict(testData)
    # res = svm.predict(np.array(testData,dtype='float32'))
    idx = res[0]
    return idx
    # print(commons.emo_str_lst[res[0]])
    # cv2.putText(img, commons.emo_str_lst[res[0]], (150, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def svm_predict_opencv_ver(svm, testData):
    (par1, par2) = svm.predict(np.array(testData, dtype='float32'))
    # (par1,par2) =
    # svm_save()
    # print("res")
    # print(res)
    # idx=res[0]
    # print(commons.emo_str_lst[res[0]])
    # cv2.putText(img, commons.emo_str_lst[res[0]], (150, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 不准确 换成了 opencv 的 svm 也是一样,参数也是按照那篇文章 设置了
    idx = int(par2[0])
    return idx

def test_one_face(shape_predictor, gray, face, img, coefficient, faces, svm):
    # print ("have pic")
    # print('-'*20)
    # landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(im_rd, faces[i]).parts()])
    shape = shape_predictor(gray, face)  # 返回68个人脸特征点的位置
    # 一张脸
    # 这里返回的是 shapes吗
    # points = face_utils.shape_to_np(shape)

    # testData[1][136]
    # testData = [[0 for i in range(136)]]
    testData = [[0 for i in range(136)]]
    # data_vector=[0 for i in range(136)]
    # coefficient = -(faces[0].top() - faces[0].bottom()) / 300.0
    # for i in range(68):
    #     cv2.circle(img, tuple(face[0].part(i).x()), 2, DeepPink, -1, 8)
    for i in range(68):
        # 每个特征点 到 他的 矩形 框出人脸的 左上角的 距离
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
        # testData[0][i * 2] = (shape.part(i).x() - faces[0].left()) / coefficient
        testData[0][i * 2] = (shape.part(i).x - faces[0].left()) / coefficient
        testData[0][i * 2 + 1] = (shape.part(i).y - faces[0].top()) / coefficient
        # data_vector[i*2]= (shape.part(i).x() - faces[0].left()) / coefficient
        # data_vector[i * 2 + 1]= (shape.part(i).x() - faces[0].left()) / coefficient
    # np.array(testData,dtype='float32')
    # res = svm.predict(testData)
    # res = svm.predict(np.array(testData,dtype='float32'))
    (par1, par2) = svm.predict(np.array(testData, dtype='float32'))
    # (par1,par2) =
    # svm_save()
    # print("res")
    # print(res)
    # idx=res[0]
    # print(commons.emo_str_lst[res[0]])
    # cv2.putText(img, commons.emo_str_lst[res[0]], (150, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 不准确 换成了 opencv 的 svm 也是一样,参数也是按照那篇文章 设置了
    idx = int(par2[0])
    print(commons.emo_str_lst[idx])
    cv2.putText(img, commons.emo_str_lst[idx], (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # if res==commons.disgust:
    # 这好像是 0 1 。。
    # joblib svm train


# https://blog.csdn.net/zmdsjtu/article/details/53667929
def test(config):
    # file_path = 0
    file_path = config.video_file_path
    if file_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Unable to connect to camera")
        return 1
    shape_detector_path = r"G:\project\pythonProj\wink-test\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()  # 人脸检测器
    shape_predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器
    # cv::ml; python
    #     cv2.ml.

    # svm_path = "SVM_DATA.xml"
    # svm1 = joblib.load(svm_path)

    # svm..predict(input_vector)
    svm1 = svm_load(commons.svm_filename)

    while (cv2.waitKey(30) != 27):
        ret, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
        # or not img
        if not ret:
            # print ("danger")
            # continue
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)  # 调用检测器人脸检测
        # coefficient = -(faces[0].top() - faces[0].bottom()) / 300.0
        if len(faces) >= 1:
            coefficient = -(faces[0].top() - faces[0].bottom()) / 300.0
            face = faces[0]
            test_one_face(shape_predictor, gray, face, img,
                          coefficient, faces, svm1)
        # for face in faces:
        #     test_one_face(shape_predictor, gray, face, img, coefficient, shapes, )
        # # print ("have pic")
        # # print('-'*20)
        # # landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(im_rd, faces[i]).parts()])
        # shape = shape_predictor(gray, face)  # 返回68个人脸特征点的位置
        # points = face_utils.shape_to_np(shape)
        # testData[1][136];
        # coefficient = -(faces[0].top() - faces[0].bottom()) / 300.0
        # # for i in range(68):
        # #     cv2.circle(img, tuple(face[0].part(i).x()), 2, DeepPink, -1, 8)
        # for i in range(68):
        #     cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
        #     testData[0][i * 2] = (shapes[0].part(i).x() - faces[0].left()) / coefficient;
        #     testData[0][i * 2 + 1] = (shapes[0].part(i).y() - faces[0].top()) / coefficient;
        cv2.imshow("test", img)
    cv2.destroyAllWindows()


# def writeTxt(name,content,fill):
def writeTxt(name, content):
    with open(name, "a+") as f:
        f.write(content + "\n")


def readData(matrix, start, dir_name):
    for i in range(50):
        with open(f"{dir_name}/{i + 1}.txt") as f:
            data = f.read()
            nums = data.split("\n")
            for j in range(136):
                # matrix[start+i][j]=f.read()
                matrix[start + i][j] = float(nums[j])
                # f.re


def push_dic(dic, key, val):
    if key in dic:
        dic[key].append(val)
    else:
        dic[key] = [val]


def write_lst_to_file(filename, lst, mode="w+"):
    data = ""
    for i in lst:
        data += f"{i}\n"
    # writeTxt(filename,data)
    with open(filename, mode) as f:
        f.write(data)


def train(config):
    dir_prefix = "train_txt"
    # emotion_prefix = "normal"
    # emotion_prefix = commons.EmoStr.happy
    # emotion_prefix = commons.EmoStr.disgust
    # emotion_prefix = commons.EmoStr.normal
    # emotion_prefix = commons.EmoStr.amazing
    # emotion_prefix = commons.EmoStr.angry
    # emotion_prefix = commons.EmoStr.fear
    # emotion_prefix = commons.EmoStr.sad
    emotion_prefix = commons.EmoStr.amazing
    # file_path = 0
    file_path = config.video_file_path
    if file_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Unable to connect to camera")
        return 1

    shape_detector_path = r"G:\project\pythonProj\wink-test\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()  # 人脸检测器
    shape_predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器
    filename = 0
    write_lst = []
    write_data_dic = {}
    print("start video")
    while (cv2.waitKey(30) != 27):
        ret, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
        # or not img
        if not ret:
            # print ("danger")
            # continue
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)  # 调用检测器人脸检测

        if len(faces) >= 1:
            # print("faces[0].top()")
            # print(faces[0].top())
            # print("faces[0]")
            # print(faces[0])
            coefficient = -(faces[0].top() - faces[0].bottom()) / 300.0
            face = faces[0]
            cv2.line(img, (face.left(), face.top()),
                     (faces[0].right(), faces[0].top()),
                     commons.DeepPink, commons.thickness)
            filename += 1
            # 是每一帧 都一个 txt
            # 每个txt 136 个数字  每个特征点 关于方框的左上角的一个关系
            shape = shape_predictor(gray, face)  # 返回68个人脸特征点的位置
            for i in range(68):
                # 每个特征点 到 他的 矩形 框出人脸的 左上角的 距离
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                cv2.putText(img, str(i), (150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # print("shape.part(i)")
                # print(shape.part(i))
                #                 shape.part(i)
                #               (215, 208)
                #                 print("type(shape.part(i))")
                #                 print(type(shape.part(i)))
                #                 print("shape.part(i).x()" )
                #                 print(shape.part(i).x() )
                #                 print("faces[0].top()")
                #                 print(faces[0].top())
                #                 print("faces[0].left()")
                #                 print(faces[0].left())
                # width=(shape.part(i).x() - faces[0].left()) / coefficient
                # print("shape.part(i)")
                # print(shape.part(i))
                # width=(shape.part(i)[0] - faces[0].left()) / coefficient
                # height=(shape.part(i)[1]- faces[0].top()) / coefficient
                width = (shape.part(i).x - faces[0].left()) / coefficient
                height = (shape.part(i).y - faces[0].top()) / coefficient
                # 不要 x()  而是 x
                #                 width=(shape.part(i)[0] - faces[0].left()) / coefficient
                # TypeError: 'dlib.point' object is not subscriptable
                # write_lst.append((filename, width))
                # write_lst.append((filename, height))
                push_dic(write_data_dic, filename, width)
                push_dic(write_data_dic, filename, height)
                # 根据 name ，他的数据 都 push
                # writeTxt(f"{filename}.txt", f"{width}")
                # writeTxt(f"{filename}.txt", f"{height}")
                # 写东西 有点卡  写内存吧
                # w+
                # https://blog.csdn.net/yinghuo110/article/details/79179165

                # testData[0][i * 2] = (shape.part(i).x() - faces[0].left()) / coefficient
                # testData[0][i * 2 + 1] = (shape.part(i).y() - faces[0].top()) / coefficient
                # data_vector[i*2]= (shape.part(i).x() - faces[0].left()) / coefficient
                # data_vector[i * 2 + 1]= (shape.part(i).x() - faces[0].left()) / coefficient

        # cv2.imshow("Dlib特征点", img)
        cv2.imshow("Dlib point", img)
        # for i in write_lst:
        #     writeTxt(f"{i[0]}.txt", i[1])

        # for key in write_data_dic:

        # write_lst_to_file(key,value)
        # 遍历 字典 python
        # with open(f"{i[0]}.txt","")
        # test_one_face(shape_predictor, gray, face, img,
        #               coefficient, faces,svm1)
        # line opencv
        # line opencv python
    cv2.destroyAllWindows()
    for key, value in write_data_dic.items():
        # write_data_dic[key] == list
        print("key", key)
        print("value", value)
        write_lst_to_file(f"{dir_prefix}/{emotion_prefix}/{key}.txt", value, "w+")


from sklearn import svm


def train_to_xml():
    # matrix = [[i for i in range(136)] for j in range(150)]
    # matrix = [[i for i in range(150)] for j in range(136)]
    # 150 行
    # print(matrix)
    # matrix = [[0]*136 for j in range(150)]
    # python 二维数组一开始 就有 大小
    # matrix = np.zeros((commons.batch_size*3, 136), dtype=float)
    # matrix = np.zeros((commons.batch_size * commons.type_cnt, 136), dtype=float)
    matrix = np.zeros((commons.batch_size * commons.type_cnt, 136), dtype='float32')
    # 写 float 不行,要写 float32
    # https://blog.csdn.net/red_ear/article/details/84940399
    train_dir_prefix = "train_txt"
    # readData(matrix, 0, "平静")
    # readData(matrix, 50, "高兴")
    # readData(matrix, 100, "厌恶")

    # readData(matrix, 0, f"{train_dir_prefix}/normal")
    # readData(matrix, 50, f"{train_dir_prefix}/happy")
    # readData(matrix, 100,f"{train_dir_prefix}/disgust")
    # # commons.EmoStr.normal
    # readData(matrix, 0, f"{train_dir_prefix}/normal")
    # readData(matrix, 50, f"{train_dir_prefix}/happy")
    # readData(matrix, 100,f"{train_dir_prefix}/disgust")

    for i in range(commons.type_cnt):
        # 50*(i+1)==50*1
        # 50-50
        # 100-50
        #     i*commons.batch_size
        #     readData(matrix, i*commons.batch_size, f"{train_dir_prefix}/normal")
        readData(matrix, i * commons.batch_size, f"{train_dir_prefix}/{commons.emo_str_lst[i]}")

    # faceLabel=[0 for i in range(50)]
    # faceLabel = [0] * 50
    # faceLabel = np.zeros((commons.batch_size*3), dtype=int)
    faceLabel = np.zeros((commons.batch_size * commons.type_cnt), dtype=int)
    # faceLabel = np.zeros((commons.batch_size * commons.type_cnt), dtype=float)
    # python 列表 ，初始化 长度
    # https://blog.csdn.net/u014470581/article/details/50937239
    for i in range(commons.batch_size):
        for j in range(commons.type_cnt):
            faceLabel[i + j * commons.batch_size] = j
            # 0 就是 normal
        # faceLabel[i] = commons.normal
        # faceLabel[i + 50] = commons.happy
        # faceLabel[i + 100] = commons.disgust

    # for i in range(50):
    #     # faceLabel[i] = 170
    #     # faceLabel[i + 50] = 250
    #     # faceLabel[i + 100] = 300
    #     faceLabel[i] = commons.normal
    #     faceLabel[i + 50] = commons.happy
    #     faceLabel[i + 100] = commons.disgust
    svm = svm_config()
    svm_train(svm, matrix, faceLabel)
    svm_save(svm, commons.svm_filename)

    # clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
    # # setTermCriteria
    # # svm.SVC TermCriteria
    # # clf.setTermCriteria((cv2.TermCriteria_MAX_ITER, 10000, 0))
    # # https://www.pythonheidong.com/blog/article/454441/5f12e4638490fc7eadb7/
    # clf.fit(matrix, faceLabel)
    # # svm_file = "train/ear_svm{}.m".format(time_str)
    # # svm_file = "ear_svm{}.m".format(now_time_str)
    # # joblib.dump(clf, "ear_svm.m")
    # svm_file = "SVM_DATA.xml"
    # joblib.dump(clf, svm_file)
    # print("dump at", svm_file)
    print("dump at", commons.svm_filename)


# svm参数配置
def svm_config():
    svm = cv2.ml.SVM_create()
    # svm.setCoef0(0)
    # svm.setCoef0(0.0)
    # svm.setDegree(3)
    # criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    svm.setTermCriteria(criteria)
    # svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setNu(0.5)
    # svm.setP(0.1)
    # svm.setC(0.01)
    # svm.setType(cv2.ml.SVM_EPS_SVR)
    svm.setType(cv2.ml.SVM_C_SVC)

    return svm


# svm训练
def svm_train(svm, features, labels):
    svm.train(np.array(features), cv2.ml.ROW_SAMPLE, np.array(labels))


# svm参数保存
def svm_save(svm, name):
    svm.save(name)


# svm加载参数
def svm_load(name):
    svm = cv2.ml.SVM_load(name)

    return svm


from config import Config

if __name__ == '__main__':
    config = Config()
    # train(config)
    # train_to_xml()
    # config=Config()
    test(config)
