# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 10:27
# @Author  : 喵奇葩
# @FileName: Cnt.py
# @Software: IntelliJ IDEA

import time


class Cnt:
    HARD = "hard"
    NORMAL = "normal"

    def __init__(self, cnt=0, one_start=False):
        self.cnt = cnt
        self.one_start = one_start
        self.start_time=0
        # self.end=0
        self.pass_time=0

    def try_add_one(self, angle, mode="normal"):
        # if angle <= 120:
        #     self.one_start = True
        # elif angle >= 170:
        #     if self.one_start:
        #         self.one_start = False
        #         self.cnt += 1
        if mode == "normal":
            self.try_add_one_of_standard(angle, 120, 170)
        elif mode == "hard":
            self.try_add_one_of_standard(angle, 110, 175)
            # self.try_add_one_of_standard(angle, 110, 180)
            # 180 太难了

    # 腿部放下来 其实不用很多，其实90度也许就够了
    #     但是有时候会偏移。。 marks 那些
    #     这个动作容易把两条腿当作一条
    #     这几个测试数据好像都没办法合适 但是差不多也行
    #     他会分不清哪个是左腿 哪个是右腿
    # 这个时间是左腿的 下一秒也很可能是左腿，算法里不知道有没有这个。时间局部性
    # mediapipe 时间局部性
    def try_add_one_hug_leg(self, angle):
        # self.try_add_one_of_standard(angle, 60, 130)
        self.try_add_one_of_standard(angle, 60, 125)
        # self.try_add_one_of_standard(angle, 60, 90)
        # self.try_add_one_of_standard(angle, 60, 100)
        # self.try_add_one_of_standard(angle, 60, 110)

    # def try_add_one_hard(self, angle):

    def try_add_one_of_standard(self, angle, start_standard, end_standard):
        if angle <= start_standard:
            self.one_start = True
        elif angle >= end_standard:
            if self.one_start:
                self.one_start = False
                self.cnt += 1

    def try_cal_time_roll(self, angle):
        self.try_cal_time_of_standard(angle, 50, 60)

    # 不是在 超过标准的节点加的 应该是每秒加一次
    def try_cal_time_of_standard(self, angle, start_standard, end_standard):
        if angle <= start_standard:
            if self.one_start == False:
                self.start_time = time.time()

            self.one_start = True

        elif angle >= end_standard:
            if self.one_start:
                self.one_start = False
                # self.cnt += 1
                # self.start_time =time.time()
                # self.end_time = time.time()
        self.pass_time = time.time() - self.start_time
