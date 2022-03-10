没有检测出脸
close_eye.jpg
close_eye2.jpg
cle3.jpg

树莓派的摄像头比较清晰 他的视频，检测眼睛比较清楚
用5 作为阈值，识别 眼睛是睁开还是闭上比较合适（带眼镜也可以）
闭眼 6--8
睁眼 3.3 左右

纵横比 作为一个小于1的数字，很难调试。 横纵比就很好调， 闭眼和睁眼的
区别比较大
纵横比 基本就 0.25--0.3 左右

笔记本的摄像头不清楚


戴眼镜 树莓派，4.5 左右 闭眼
比如说5帧里面 有三帧 是闭眼，我就认为是闭眼

有表情的时候，睁眼也不是0。3 到不了0。3

这个蜂鸣器是高电平唤起的
https://www.basemu.com/raspberry-pi-4-gpio-pinout.html
树莓派 4 io 口
红色是 正极,橙色是负极

但是其实他识别出眼睛就是识别出人脸了，眼睛和人脸是一起的，就算
视频里有人脸，他识别不出来，他就会认为人不在，这样报警太多了吧
而且他是正放着的，不可能正放吧，这样挡着驾驶了

生气 貌似不会有当作睡觉的情况吧 不多

def eyebrow_aspect_ratio(shape, d):

文档：上面是 现在计算的斜率，.note
链接：http://note.youdao.com/noteshare?id=8c83b7996a7aaf7179feddf804210b4f&sub=11F0BE6D7DDC4A6B91ED3F89E1730D25

文档：伤心的 嘴 的 斜率 好像没什么意义.not...
链接：http://note.youdao.com/noteshare?id=824a590c16441c7931c0d3b0601740e1&sub=0F1DEFF24BC7485FAB5C62BD4E886755


https://blog.csdn.net/qq_21808961/article/details/83239631
md 代办
，分别为

- [x] 惊讶
- [ ] 厌恶
- [x] 愤怒、
- [ ] 恐惧、
- [x] 悲伤、
- [x] 愉悦

没有 疼痛表情吗

disgust
https://tieba.baidu.com/p/6257492572
微表情心理学：厌恶情绪的诱因与表情特征

data_collect.py
配置类型

https://tech.hexun.com/2019-03-18/196530858.html
特征点 图片

嘴巴 好像没有意义 因为一旦 你的 图片的方向不对,就是
下巴抬起来和不抬起来,他的区别很大的

我可以一个视频一直一个表情，然后看他预测出来的结果是什么 放在list 算 recall

依赖

D:\proj\private\wink-test-private\blink_detect>python -m pip install "D:\download\dlib-19.19.0-cp37-cp37m-win_amd64.whl"

D:\proj\private\wink-test-private\blink_detect>python -m pip install imutils

D:\proj\private\wink-test-private\blink_detect\blink_detect_func.py
2022年2月25日18:34:44
识别人脸表情和开车的眨眼
iot 电脑没有摄像头

git remote add origin https://gitee.com/starplatinum111/blink_detect.git

git add 文件夹

git 有些文件没有add ignore也没有

2022年2月25日21:52:36
disgust_res
amazing_res
angry_res
sad_res

2022年2月26日09:36:28
训练 表情 ，在 这里
G:\project\pythonProj\blink_detect\data_collect2.py
yes_video_path = r"G:\FFOutput\sad_VID_20211006_155448.mp4"
no_video_path = r"G:\FFOutput\normalVID_20211003_144749.mp4"
写上视频的路径
G:\project\pythonProj\blink_detect\commons.py

训练收集数据的时候要配置用的哪个脸部特征 比如说鼻子啊
G:\project\pythonProj\blink_detect\data_collect2.py
avg_mouth_slope_up = util.avg_mouth_slope_up(shape)


配置现在训练的是哪种表情
train_type="sad"
打印出来的 time_str 2022_02_26_09_37_27，就可以知道现在收集的数据txt是那个

复制到到这里
G:\project\pythonProj\blink_detect\train_svm.py
time_str= "2022_02_26_09_37_27"

yes_video_path = r"G:\FFOutput\sad_VID_20211006_155448.mp4"
使用这个视频训练的结果是这样的
sad 和 angry 很难分辨 很多时候 也不显示表情


svm_mouth_below_sad_path = "train/sad_2022_02_26_09_37_27.m"
    # 效果不好


    G:\project\pythonProj\blink_detect\blink_detect_func_right.py
    识别结果写入文件 做统计


这里配置 现在的表情
    G:\project\pythonProj\blink_detect\commons.py
    train_type="fear"

数据写入 这个文件夹
G:\project\pythonProj\blink_detect\res

video_path = r"G:\FFOutput\fear_VID_20211006_114409.mp4"
效果不好

happy disgust angry  amazing sad 都不行

## 准确性
2022年2月26日10:23:00
amazing 还是有点准确的 就是有时候又 disgust 

disgust 和andry 重合度太高 。。因为 都有皱眉头

video_path = r"G:\FFOutput\fear_VID_20211006_212800.mp4"
这个一点都不准确

video_path = r"G:\FFOutput\sad_VID_20211006_155448.mp4"
sad 的andry 甚至更多

happy 不行

不能只靠一个参数来训练吧

video_path = r"G:\emotion\angry_many_pos_VID_20211005_162002.mp4"
 angry 也全是disgust 和 sad 

 惊讶 还行
 厌恶 不太行
愤怒 有disgust 和 sad 
恐惧 完全不行
 悲伤  一般
 愉悦 不行

D:\proj\private\wink-test-private\blink_detect>git push origin master
Enumerating objects: 29, done.
Counting objects: 100% (29/29), done.
Delta compression using up to 16 threads
Compressing objects: 100% (26/26), done.
Writing objects: 100% (26/26), 69.01 MiB | 10.94 MiB/s, done.
Total 26 (delta 12), reused 0 (delta 0), pack-reused 0
remote: Powered by GITEE.COM [GNK-6.2]
remote: warning: This repository has 1 files larger than 50.00 MB
remote: The size information of several files as follows:
remote: File e0ec20d69ac9195f056214f071e12afcfe714199 72.31  MB
remote: Use command below to see the filename:
remote: git rev-list --objects --all | grep $oid
remote: HelpLink: https://gitee.com/help/articles/4232
To https://gitee.com/starplatinum111/blink_detect.git
   e7c0ef7..3d7556f  master -> master

大文件上传确实挺慢的 大文件要不要ignore呢
刚才试了一下 大文件通过阿里云盘传递的话 也可以，就是稍微麻烦了点 要建立文件夹啥的，但是也没有太麻烦，

大文件下载有点慢 要不还是 ignore ,但是又考虑到下载下来就能用。。

2022年3月1日19:48:43
show_win 
