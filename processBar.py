# -*- coding: utf-8 -*-
# @Time    : 2021/9/21 17:48
# @Author  : 喵奇葩
# @FileName: processBar.py
# @Software: IntelliJ IDEA
import cv2


def nothing(emp):
    pass


class ProcessBar:
    def __init__(self, cap, video_win_name):
        # cv2.namedWindow('video')
        # cv2.namedWindow(video_win_name)
        self.video_win_name = video_win_name
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap = cap
        self.loop_flag = 0
        self.pos = 0
        # cv2.createTrackbar('time', 'video', 0, self.frames, nothing)
        cv2.createTrackbar('time', video_win_name, 0, self.frames, nothing)

    def control(self):
        if self.loop_flag == self.pos:
            self.loop_flag = self.loop_flag + 1
            # cv2.setTrackbarPos('time', 'video', self.loop_flag)
            cv2.setTrackbarPos('time', self.video_win_name, self.loop_flag)
            # 这句话是让视频随着进度条动，进度条也随着数字动
        else:
            # self.pos = cv2.getTrackbarPos('time', 'video')
            self.pos = cv2.getTrackbarPos('time', self.video_win_name)
            self.loop_flag = self.pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
        # 这个不写的话 调整进度条就没用了
