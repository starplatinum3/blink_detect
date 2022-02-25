# author：郭昆仑
# coding=utf-8
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
import time


def eyebrow_aspect_ratio():
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


# 计算EAR
def eye_aspect_ratio(eye):
    # print(eye)计算阈值函数
    A = distance.euclidean(eye[1], eye[5])  # A,B是计算两组垂直眼睛标志的距离，而C是计算水平眼睛标志之间的距离
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def nose_aspect_ratio(nose):
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


def happy_aspect_ratio(mouth):
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


config = Config()
# 导入opnecv的级联分类器
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier(config.face_cascade_path)
# smile_cascade = cv2.CascadeClassifier('F:\\pycharm2017project\\pycharm project\\virtual env1\\Lib\\site-packages\\cv2\\data\\harrcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(config.eye_cascade_path)

pwd = os.getcwd()  # 获取当前路径
model_path = os.path.join(pwd, 'model')  # 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测模型路径

detector = dlib.get_frontal_face_detector()  # 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器

EYE_AR_THRESH = 0.3  # EAR判断阈值，默认0.3，如果大于0.3则认为眼睛是睁开的；小于0.3则认为眼睛是闭上的
EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，只有小于阈值的帧数超过了这个值时，才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。

# 嘴巴闭合阈值，大于0.5认为张开嘴巴
# 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
mCOUNT = 0
mTOTAL = 0
# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0
blink_counter = 0  # 眨眼计数
count = 0
# detect_danger = 0
detect_danger = 1
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
show_it=False

def draw_characteristic_point(img, shape):
    for i in range(68):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)

# class EmotionDetector:

print ("start")
while (1):
    ret, img = cap.read()  # 视频流读取并且返回ret是布尔值，而img是表示读取到的一帧图像
    if not ret:
        continue
    img = cv2.flip(img, 1)  # 水平翻转图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 把读取到Img转换为2进制灰度图像
    rects = detector(gray, 0)  # 调用检测器人脸检测
    # print ("detector")
    # print (detector)
    # 识别不出来 树莓派 dlib
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)  # 调用级联分类器对gray进行人脸识别并且返回一个矩形列表
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print ("set up")
    # 没有人脸
    # print ("rects")
    # print (rects)
    # print ("img")
    # print (img)
    # 注意摄像头不要拿倒过来了,摄像头是在上面的
    for k, d in enumerate(rects):
        # print("第", k + 1, "个人脸d的坐标：",
        # 	  "left:", d.left(),
        # 	  "right:", d.right(),
        # 	  "top:", d.top(),
        # 	  "bottom:", d.bottom())
        width = d.right() - d.left()
        heigth = d.bottom() - d.top()
    for rect in rects:
        # print ("have pic")
        # print('-'*20)
        # landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(im_rd, faces[i]).parts()])
        shape = predictor(gray, rect)  # 返回68个人脸特征点的位置
        points = face_utils.shape_to_np(shape)  # 将facial landmark (x, y)转换为NumPy 数组
        # points = shape.parts()
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼特征点
        mouth_points = points[48:68]  # 设置嘴巴特征点，其实也可以取出
        rmeimao_points = points[17:22]  # 设置右眉毛特征点，其实也可以取出
        lmeimao_points = points[22:27]  # 设置左眉毛特征点，其实也可以取出
        nose1_points = points[27:31]  # 设置鼻梁间距特征点，其实也可以取出
        nose2_points = points[31:36]  # 设置间距鼻孔特征点，其实也可以取出
        eye_points = points[36:48]
        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR阈值
        rightEAR = eye_aspect_ratio(rightEye)  # 计算左眼EAR阈值
        mouth_ear = mouth_aspect_ratio(mouth_points)  # 计算嘴巴EAR阈值
        eyebrow_ear = eyebrow_aspect_ratio()  # 眉毛
        nose_ear = nose_aspect_ratio(nose2_points)
        pain_ear1 = pain_aspect_ratio1(eye_points)
        pain_ear2 = pain_aspect_ratio2(mouth_points)
        happy_ear = happy_aspect_ratio(mouth_points)
        # print('leftEAR = {0}'.format(leftEAR))
        # print('rightEAR = {0}'.format(rightEAR))
        # print('openMouth = {0}'.format(mouth_ear))
        # print('eyeborwtilt = {0}'.format(eyebrow_ear))
        # print('happy_ear = {0}'.format(happy_ear))
        # print('noselength ={0}'.format(nose_ear))
        ear = (leftEAR + rightEAR) / 2.0
        # leftEyeHull = cv2.convexHull(leftEye) # 寻找左眼轮廓
        # rightEyeHull = cv2.convexHull(rightEye) # 寻找右眼轮廓
        # mouthHull = cv2.convexHull(mouth_points) # 寻找嘴巴轮廓

        lmeimaoHull = cv2.convexHull(lmeimao_points)  # 寻找左眉毛轮廓
        rmeimaoHull = cv2.convexHull(rmeimao_points)  # 寻找左眉毛轮廓
        nose1Hull = cv2.convexHull(nose1_points)  # 寻找鼻梁轮廓
        nose2Hull = cv2.convexHull(nose2_points)  # 寻找鼻孔轮廓

        # cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
        # cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓
        # cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)# 绘制嘴巴轮廓
        # cv2.drawContours(img, [lmeimaoHull], -1, (0, 255, 0), 2)  # 绘制眉毛轮廓
        # cv2.drawContours(img, [rmeimaoHull], -1, (0, 255, 0), 2)  # 绘制眉毛轮廓
        # cv2.drawContours(img, [nose1Hull], -1, (0, 255, 0), 2)  # 绘制鼻梁轮廓
        # cv2.drawContours(img, [nose2Hull], -1, (0, 255, 0), 2)  # 绘制鼻孔轮廓

        # draw_characteristic_point(img,shape)
        # for i in range(68):
        # 	cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
        count += 1
        # for pt in leftEye:
        # 	pt_pos = (pt[0], pt[1])
        # 	cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

        # for pt in rightEye:
        # 	pt_pos = (pt[0], pt[1])
        # 	cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        if mouth_ear > MAR_THRESH:
            mCOUNT += 1
        else:
            if mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
                mTOTAL += 1
            mCOUNT = 0
        # if blink_counter >= 10:
        # 	flag=1
        if detect_danger == 1:
            cv2.putText(img, "TIRED!!Warnning".format(ear), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print ( "TIRED!!Warnning")
        else:
            cv2.putText(img, "Blinks:{0}".format(blink_counter),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
            cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "MOUTH{0}".format(mTOTAL), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print ("Blinks:{0}".format(blink_counter))
            print ("EAR:{:.2f}".format(ear))
            print ( "MOUTH{0}".format(mTOTAL))


        # 嘴巴闭合阈值，大于0.5认为张开嘴巴
        # 当mar大于阈值0.5的时候，接连多少帧一定发生嘴巴张开动作，这里是3帧
        if mouth_ear > MAR_THRESH and mCOUNT >= MOUTH_AR_CONSEC_FRAMES:
            if ear > EYE_AR_THRESH:
                cv2.putText(img, "AMAZING", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print ("AMAZING")
        # 如果闭上嘴巴，可能是生气或者正常
        # 0.03 好像更加不准了
        # 0.02
        # 0.01 特别不准
        angry_limit = 0.02
        # if mouth_ear < MAR_THRESH and eyebrow_ear < 0.03:
        if mouth_ear < MAR_THRESH and eyebrow_ear < angry_limit:
            cv2.putText(img, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
            print ("angry")
        # else:
        # 	cv2.putText(img, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
        # nature 也打印出来的话 就挺烦的
        # 不过 angry 还是不准
        if happy_ear < 5:
            if ear < 0.18:
                if pain_ear2 > pain_ear1 * 3 / 4:
                    cv2.putText(img, "Pain", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print ("Pain")
        if happy_ear > 5:
            if ear > 0.2:
                if pain_ear2 > pain_ear1 * 4 / 5:
                    cv2.putText(img, "Happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                2)
                    print ("Happy")
    # cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 应该是跑起来了 ，但是没有显示屏
cap.release()
cv2.destroyAllWindows()
