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