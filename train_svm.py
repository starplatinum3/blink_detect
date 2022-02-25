import numpy as np
from sklearn import svm

import pandas as pd

# from sklearn.externals import joblib
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
import util

# time_str = "2021_09_21_19_09_06"
# tarin_str= "train_open2021_09_21_19_18_05.txt"
# time_str = "2021_09_21_19_18_05"
# time_str = "2021_09_21_22_01_12"
# time_str = "2021_09_21_22_01_12"
# time_str = "2021_09_22_08_48_21"
# time_str = "2021_09_22_08_53_41"
# time_str = "2021_09_22_09_17_24"
# time_str = "2021_10_01_22_27_18"
# time_str = "2021_10_02_10_44_20"
# time_str = "2021_10_02_10_49_42"
# time_str = "2021_10_02_11_12_46"
# time_str = "2021_10_03_11_06_37"
# time_str = "2021_10_03_11_20_49"
# time_str = "2021_10_03_11_31_39"
# time_str = "2021_10_03_13_13_59"
# time_str = "2021_10_03_15_22_24"
# time_str = "2021_10_03_17_01_07"
# time_str = "2021_10_05_17_31_42"
# time_str = "2021_10_05_17_48_49"
# time_str = "2021_10_05_20_16_42"
time_str = "2021_10_05_20_45_55"
# train_mouth_sad_yes_2021_10_01_22_27_18.txt
# now_time_str = util.get_now_time_str()
# train_yes_path = 'train/train_open{}.txt'.format(time_str)
# train_no_path = 'train/train_close{}.txt'.format(time_str)

# train_yes_path = 'train/train_open{}.txt'.format(time_str)
# train_no_path = 'train/train_close{}.txt'.format(time_str)
import commons
prefix="train/"
train_yes_path =prefix+ commons.train_type + '_yes_{}.txt'.format(time_str)
train_no_path =prefix+ commons.train_type + '_no_{}.txt'.format(time_str)
# svm_file = "train/ear_svm{}.m".format(time_str)
svm_file = "train/{}_{}.m".format(commons.train_type, time_str)

data_dic = {"yes": [], "no": []}
# https://www.gairuo.com/p/pandas-plot-scatter
# df = pd.DataFrame(data_dic)

# train_open_path='train_open.txt'
# train_close_path='train_close.txt'

# train_open_txt = open(train_yes_path, 'r')
# train_close_txt = open(train_no_path, 'r')

# train_open_txt = open('train_open.txt', 'rb')
# train_close_txt = open('train_close.txt', 'rb')

train = []
labels = []

# 在这里打了 标签 是 0 1
def read_file(train, labels, file_path,label=1):
    # train = []
    # labels = []
    train_txt = open(file_path, 'r')
    train_vars=[]

    # print('Reading train_open.txt...')
    line_ctr = 0
    for txt_str in train_txt.readlines():
        temp = []
        # print(txt_str)
        datas = txt_str.strip()
        datas = datas.replace('[', '')
        datas = datas.replace(']', '')
        datas = datas.split(',')
        print(datas)
        # 一行的 datas
        for data in datas:
            # print(data)
            data = float(data)
            temp.append(data)
        # print(temp)
        train.append(temp)
        train_vars.append(temp)
        # labels.append(0)
        labels.append(label)
        # 所有的数字作为训练
    train_txt.close()
    return train_vars


def push_train_vars(train, data_dic, key):
    vars = []
    for i in train:
        # avg_var=util.avg(i)
        vars.append(util.avg(i))

    data_dic[key] = vars
    # data_dic["yes"] = yes_vars


print('Reading train_open.txt...')
train_vars=read_file(train, labels, train_yes_path,1)

push_train_vars(train_vars, data_dic, "yes")

# data = float(txt_str)

# if line_ctr <= 12:
# 	line_ctr += 1
# 	temp.append(data)
# elif line_ctr == 13:
# 	# print(temp)
# 	# print(len(temp))
# 	train.append(temp)
# 	labels.append(0)
# 	temp = []
# 	line_ctr = 1
# 	temp.append(data)

print('Reading train_close.txt...')
train_vars=read_file(train, labels, train_no_path,0)
# line_ctr = 0
# temp = []
# for txt_str in train_close_txt.readlines():
#     temp = []
#     # print(txt_str)
#     datas = txt_str.strip()
#     datas = datas.replace('[', '')
#     datas = datas.replace(']', '')
#     datas = datas.split(',')
#     print(datas)
#     for data in datas:
#         # print(data)
#         data = float(data)
#         temp.append(data)
#         # 所有数字放进temp
#     # print(temp)
#     # train 就两个列表，不是的
#     train.append(temp)
#     labels.append(1)

# data = float(txt_str)

# if line_ctr <= 12:
# 	line_ctr += 1
# 	temp.append(data)
# elif line_ctr == 13:
# 	# print(temp)
# 	# print(len(stemp))
# 	train.append(temp)
# 	labels.append(1)
# 	temp = []
# 	line_ctr = 1
# 	temp.append(data)

push_train_vars(train_vars, data_dic, "no")
# 这里就有问题了
# 因为是引用 所以会变的
print("data_dic")
print(data_dic)

def cut_lst(lst1, lst2):
    len_1 = len(lst1)
    len_2 = len(lst2)
    if len_1 < len_2:
        # 3 取得 ：3 得到 0,1
        # 赋值  要返回
        lst2 = lst2[:len_1]
    elif len_1 > len_2:
        lst1 = lst1[:len_2]
    return lst1, lst2


data_dic["yes"], data_dic["no"] = cut_lst(data_dic["yes"], data_dic["no"])

import matplotlib.pyplot as plt
# https://blog.csdn.net/qq_27825451/article/details/83057541
def show(data_dic):
    df = pd.DataFrame(data_dic)
    print("df")
    print(df)
    # df.plot.scatter(x='')
    df.plot.bar()
    plt.title(svm_file)

    plt.show()

    # plt 名字
# ['0.123', ' 0.111', ' 0.098']
# ['0.155', ' 0.187', ' 0.187']

def start_train(train,labels):
    for i in range(len(labels)):
        print("{0} --> {1}".format(train[i], labels[i]))

    # train_close_txt.close()
    # train_open_txt.close()

    print(train)
    print(labels)
    clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
    clf.fit(train, labels)
    # svm_file = "train/ear_svm{}.m".format(time_str)
    # svm_file = "ear_svm{}.m".format(now_time_str)
    # joblib.dump(clf, "ear_svm.m")
    joblib.dump(clf, svm_file)
    print("dump at",svm_file)

    # print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]]')
    # res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]])
    # print(res)

    # print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]]')
    # res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]])
    # print(res)

    # print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]]')
    # res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]])
    # print(res)

    # print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]]')
    # res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]])
    # print(res)

    print('predicting [[0.34, 0.34, 0.31]]')
    res = clf.predict([[0.34, 0.34, 0.31]])
    print(res)

    print('predicting [[0.19, 0.18, 0.18]]')
    res = clf.predict([[0.19, 0.18, 0.18]])
    print(res)


# print('predicting [[0.34]]')
# res = clf.predict([[0.34]])
# print(res)

# print('predicting [[0.19]]')
# res = clf.predict([[0.19]])
# print(res)


def test_cut_lst():
    # [1, 1, 1, 1, 1] [1, 1, 1, 2, 2]
    lst1 = [1, 1, 1, 1, 1]
    lst2 = [1, 1, 1, 2, 2, 2, 2, 2]
    lst1, lst2 = cut_lst(lst1, lst2)
    print(lst1, lst2)

# 眼睛睁大了 框也没有增大 dlib
# show(data_dic)
start_train(train,labels)
# https://zhuanlan.zhihu.com/p/361835285
# 悲伤