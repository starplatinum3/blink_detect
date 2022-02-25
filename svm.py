# -*- coding: utf-8 -*-
# @Time    : 2021/9/21 19:34
# @Author  : 喵奇葩
# @FileName: svm.py
# @Software: IntelliJ IDEA
import joblib

VECTOR_SIZE = 3


def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue


def blink_detect_svm(ear_vector, ear, clf):
    ret, ear_vector = queue_in(ear_vector, ear)
    if len(ear_vector) == VECTOR_SIZE:
        print(ear_vector)
        input_vector = [ear_vector]
        res = clf.predict(input_vector)
        print(res)
        return res
    return None


class Svm:
    VECTOR_SIZE = 3

    def __init__(self, svm_path="train/ear_svm2021_09_21_19_18_05.m"):
        self.ear_vector = []
        # svm_path = "train/ear_svm2021_09_21_19_18_05.m"
        self.clf = joblib.load(svm_path)

    def queue_in(self, queue, data):
        ret = None
        if len(queue) >= self.VECTOR_SIZE:
            ret = queue.pop(0)
        queue.append(data)
        return ret, queue

    # def blink_detect_svm(self, ear_vector, ear, clf):
    #     ret, ear_vector = queue_in(ear_vector, ear)
    #     if len(ear_vector) == self.VECTOR_SIZE:
    #         print(ear_vector)
    #         input_vector = [ear_vector]
    #         res = clf.predict(input_vector)
    #         print(res)
    #         return res
    #     return None

    # def blink_detect_svm(self, ear, clf):
    #     ret, self.ear_vector = queue_in(self.ear_vector, ear)
    #     if len(self.ear_vector) == self.VECTOR_SIZE:
    #         print(self.ear_vector)
    #         input_vector = [self.ear_vector]
    #         res = clf.predict(input_vector)
    #         print(res)
    #         return res
    #     return None

    def blink_detect_svm(self, ear):
        return  self.detect_svm(ear)
        # ret, self.ear_vector = queue_in(self.ear_vector, ear)
        # if len(self.ear_vector) == self.VECTOR_SIZE:
        #     print(self.ear_vector)
        #     input_vector = [self.ear_vector]
        #     res = self.clf.predict(input_vector)
        #     print(res)
        #     return res
        # return None

    def detect_svm(self, ear):
        ret, self.ear_vector = queue_in(self.ear_vector, ear)
        if len(self.ear_vector) == self.VECTOR_SIZE:
            print(self.ear_vector)
            input_vector = [self.ear_vector]
            # 输入需要是 二维
            res = self.clf.predict(input_vector)
            # svm 返回的 是 二分类
            # svm 是 二分类 吗
            # svm 多分类 joblib
            # svm 只会返回 0 1
            # opencv svm
            # https://blog.csdn.net/mervins/article/details/78860358
            print(res)
            return res
        return None