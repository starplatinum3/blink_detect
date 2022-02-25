# -*- coding: utf-8 -*-
# @Time    : 2021/10/7 20:54
# @Author  : 喵奇葩
# @FileName: NoFaceWarnUtil.py
# @Software: IntelliJ IDEA
import time

# import cv2

import commons


# from blink_detect import beep_sec


class NoFaceWarnUtil:
    def __init__(self):
        self.start_time = time.time()
        self.should_warn = True

    def no_face_warn(self, rects):
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
        time_gap = now_time - self.start_time
        # print("time_gap",time_gap)

        if time_gap >= commons.warn_time:
            self.should_warn = True
            # check_danger(rects)
            # if len(rects) == 0:
            #     print("danger")
            #     cv2.putText(img, "TIRED!!Warnning", (150, 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     # if use_beep:
            #     #     beep.beep_of(1)
            #     beep_sec(1)
            # 不是这个时候 才能叫  是这个时候 改变了状态
            # print("start_time", self.start_time)
            self.start_time = now_time

        if len(rects) == 0:
            if self.should_warn:
                self.should_warn = False
                return True
                # print("danger")
                # cv2.putText(img, "TIRED!!Warnning", (150, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # # if use_beep:
                # #     beep.beep_of(1)
                # beep_sec(1)

        # return start_time, should_warn

    # def no_face_warn(self, start_time, rects, img, should_warn):
    #     now_time = time.time()
    #     # print ("now_time")
    #     # print (now_time)
    #     # print ("now_time-start_time")
    #     # print (now_time - start_time)
    #     # warn_time = 1
    #     # 每隔一秒 检查一下人在不在
    #     # if now_time - start_time >= warn_time:
    #     # 不是这个逻辑，应该是 每次过了 3s 就测试一次
    #     # print(start_time,start_time)
    #     # print(now_time,now_time)
    #     time_gap = now_time - start_time
    #     # print("time_gap",time_gap)
    #
    #     if time_gap >= commons.warn_time:
    #         should_warn = True
    #         # check_danger(rects)
    #         # if len(rects) == 0:
    #         #     print("danger")
    #         #     cv2.putText(img, "TIRED!!Warnning", (150, 30),
    #         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #         #     # if use_beep:
    #         #     #     beep.beep_of(1)
    #         #     beep_sec(1)
    #         # 不是这个时候 才能叫  是这个时候 改变了状态
    #         print("start_time", start_time)
    #         start_time = now_time
    #
    #     if len(rects) == 0:
    #         if should_warn:
    #             print("danger")
    #             cv2.putText(img, "TIRED!!Warnning", (150, 30),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #             # if use_beep:
    #             #     beep.beep_of(1)
    #             beep_sec(1)
    #             should_warn = False
    #     return start_time, should_warn
