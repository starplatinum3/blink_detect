# -*- coding: utf-8 -*-
# @Time    : 2021/9/21 16:44
# @Author  : 喵奇葩
# @FileName: commons.py
# @Software: IntelliJ IDEA

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
# EYE_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES = 1
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

blink_cnt_sleep = 4
fps_guess = 25
frame_cnt_blink_refresh = fps_guess * 3

# flip=True
flip = False

# train_type="train_eye_brow_angry"
# train_type="train_eye_amazing"
# train_type="train_mouth_sad"
train_type="mouth_up_sad"
# train_type="nose_wing_distance_disgust"
# train_type="nose_bridge_distance_disgust"
# train_type = "fear_eye"
# train_type = "close_eye"
# train_type = "disgust_nose_ar"
# train_type="train_eye_amazing"

# 鼻梁
# bridge
# train_open_path

close_eye_long_frame_cnt = 25
# close_eye_long_frame_cnt = 50

# warn_time = 1
warn_time = 3
# 每隔一秒 检查一下人在不在

VECTOR_SIZE = 3

DeepPink = (147, 255, 20)

thickness = 1

normal = 0
amazing = 1
disgust = 2
angry = 3
fear = 4
sad = 5
happy = 6
# - [x] 惊讶
# - [ ] 厌恶
# - [x] 愤怒、
# - [ ] 恐惧、
# - [x] 悲伤、
# - [x] 愉悦

batch_size = 50
# type_cnt = 4
type_cnt = 7


class EmoStr:
    normal = "normal"
    amazing = "amazing"
    disgust = "disgust"
    angry = "angry"
    fear = "fear"
    sad = "sad"
    happy = "happy"

train_txt_dir="train_txt"

shape_detector_path = r"G:\project\pythonProj\wink-test\shape_predictor_68_face_landmarks.dat"

emo_str_lst=["normal","amazing", "disgust","angry","fear","sad","happy"]

svm_filename= "SVM_DATA_opencv_ver.xml"