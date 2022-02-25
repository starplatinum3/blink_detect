# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 16:28
# @Author  : https://www.jianshu.com/p/221ff6bf4f13
# @FileName: beep.py.py
# @Software: IntelliJ IDEA


import os


def beep_test1():
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))


# !/usr/bin/env python
import RPi.GPIO as GPIO
import time

# Buzzer = 11    # pin11
# Buzzer = 0    # pin11
Buzzer = 4  # pin11
on_level = GPIO.HIGH
off_level = GPIO.LOW


def setup(pin):
    global BuzzerPin
    BuzzerPin = pin
    # GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
    # GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(BuzzerPin, GPIO.OUT)
    # GPIO.output(BuzzerPin, GPIO.HIGH)
    GPIO.output(BuzzerPin,off_level)


def on():
    # GPIO.output(BuzzerPin, GPIO.LOW)
    GPIO.output(BuzzerPin,on_level)
    # 低电平是响
    # 我这个设备不是


def off():
    # GPIO.output(BuzzerPin, GPIO.HIGH)
    GPIO.output(BuzzerPin, off_level)
    # 高电平是停止响


def beep(x):  # 响3秒后停止3秒
    on()
    time.sleep(x)
    off()
    time.sleep(x)

def beep_of(x):  # 响3秒后停止3秒
    on()
    time.sleep(x)
    off()
    # time.sleep(x)


def loop():
    while True:
        beep(3)


def destroy():
    # GPIO.output(BuzzerPin, GPIO.HIGH)
    GPIO.output(BuzzerPin,off_level)
    GPIO.cleanup()  # Release resource


# 这个蜂鸣器是高电平唤起的
# https://www.basemu.com/raspberry-pi-4-gpio-pinout.html
# 树莓派 4 io 口
# 红色是 正极,橙色是负极
if __name__ == '__main__':  # Program start from here
    setup(Buzzer)
    try:
        loop()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
        destroy()
