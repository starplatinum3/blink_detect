# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 19:57
# @Author  : https://blog.csdn.net/c2a2o2/article/details/88035635
# @FileName: face_align.py.py
# @Software: IntelliJ IDEA

import face_alignment
from skimage import io

# TypeError: __init__() got an unexpected keyword argument 'enable_cuda'
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)
img_path=r"G:\project\pythonProj\face-alignment\test\assets\aflw-test.jpg"
# input = io.imread('../test/assets/aflw-test.jpg')
input = io.imread(img_path)
preds = fa.get_landmarks(input)
