# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 16:00
# @Author  : 喵奇葩
# @FileName: detect_img.py.py
# @Software: IntelliJ IDEA


import dlib
import cv2

detector = dlib.get_frontal_face_detector() #获取人脸分类器
# https://blog.csdn.net/weixin_44086593/article/details/87510908
# img = cv2.imread('PIC/taonaimu5.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)

# 摘自官方文档：
# image is a numpy ndarray containing either an 8bit grayscale or RGB image.
# opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
b, g, r = cv2.split(img)    # 分离三个颜色通道
img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片

dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
# enumerate是一个Python的内置方法，用于遍历索引
# index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
# left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
show_it=False
for index, face in enumerate(dets):
    print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    if show_it:
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.namedWindow('taonaimu', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('taonaimu', img)
    # cv2.imsave()
    cv2.imwrite('lena2.jpg',img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()
