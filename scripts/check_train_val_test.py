# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:49:45 2021

@author: Admin
"""
import tables
import pickle
from pathlib import Path
import os
import json
import numpy as np
import random
from tqdm import tqdm

config = {}
config["data_file"] = os.path.abspath("/data/liuaohan/co-training/more_brain_data_160_80.h5")
config["training_file"] = os.path.abspath("/data/liuaohan/co-training/more_refined_training_ids_160_80.pkl")
config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/more_refined_validation_ids_160_80.pkl")
config['label_file'] = 'more_refined_label_new1-nohand.h5'


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)
def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)

def resolve_subject_id(ID):
    """
    要从ID中分离出reportID和seriesID并返回
    ID是str
    """
    return [Path(ID).parent.parent.name, Path(ID).name]

def training_valid_cases():
    data_file = tables.open_file(config["data_file"], "r")
    al = []
    for i in range(len(data_file.root.subject_ids)):
        ID = data_file.root.subject_ids[i].decode('utf-8')
        al.append([i] + resolve_subject_id(ID))
    training_set = []
    training_indices = pickle_load(config["training_file"])
    #training_indices = random.sample(list(range(12253)), 5000)
    print(training_indices[:10])
    for index in training_indices:
        ID = data_file.root.subject_ids[index].decode('utf-8')
        training_set.append([index] + resolve_subject_id(ID))
        
    valid_set = []
    validation_indices = pickle_load(config["validation_file"])
    #validation_indices = list(set(range(12253))-set(training_indices))
    for index in validation_indices:
        ID = data_file.root.subject_ids[index].decode('utf-8')
        valid_set.append([index] + resolve_subject_id(ID))
    data_file.close()
    return training_set, valid_set, al

def operation_ratio(dataset):
    cnt_operation = 0
    for x in dataset:
        flag = 0
        for s in report_desc[x[1]]:
            if '术后' in s:
                flag = 1
        if flag:
            cnt_operation += 1
    print('术后', cnt_operation, cnt_operation/len(dataset))

training, valid, al = training_valid_cases()
print(len(training), len(valid))

with open('report_patient.json', 'r') as fp:
    report_patient = json.load(fp)
with open('report_desc.json', 'r') as fp:
    report_desc = json.load(fp)

operation_ratio(al)
##########################################################
filt_patient = {} #训练集出现的病人
for x in training:
    if x[1] in report_patient:
        filt_patient[report_patient[x[1]][0]] = True

inde = {}
for x in valid:
    if x[1] in report_patient and report_patient[x[1]][0] in filt_patient:
        continue
    inde[x[0]] = True
print(len(inde)) #372
label_file_opened = tables.open_file(config['label_file'], "r")

label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
cnt = 0
label_not_hand = np.zeros((20,4))
label_not_hand[13,3] = 1
for x in valid:
    if x[0] in inde:
        label = label_file_opened.root.label[x[0]]
        label_cnt += label
        label_cnt_ab += np.max(label-label_not_hand, 0)
        cnt += 1
#print(label_cnt/cnt)
print(label_cnt_ab, label_cnt_ab/cnt)

label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
cnt = 0
label_not_hand = np.zeros((20,4))
label_not_hand[13,3] = 1
for x in valid:
    if x[0] not in inde:
        label = label_file_opened.root.label[x[0]]
        label_cnt += label
        label_cnt_ab += np.max(label-label_not_hand, 0)
        cnt += 1
#print(label_cnt/cnt)
print(label_cnt_ab, label_cnt_ab/cnt)

cnt = 0
label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
for x in valid+training:
    label = label_file_opened.root.label[x[0]]
    label_cnt += label
    label_cnt_ab += np.max(label-label_not_hand, 0)
    cnt += 1
print(cnt, label_cnt_ab/cnt)

"""
真实：
372
inde [0.26344086 0.2983871  0.49193548 0.08064516]
not inde [0.45316159 0.41334895 0.66042155 0.09250585]
随机：
410
[0.25609756 0.26829268 0.52439024 0.05609756]
[0.47181373 0.45220588 0.64338235 0.09803922]
384
[0.25260417 0.2890625  0.51041667 0.08333333]
[0.4608076  0.46437055 0.65558195 0.09144893]
393
[0.23409669 0.24681934 0.50381679 0.04071247]
[0.44297719 0.43217287 0.65906363 0.10444178]

全部12253基本为（随机删了一个）：
[0.39536402 0.39177277 0.60741103 0.08602677]
设置训练集为1000而非11027：
[0.36086633 0.36713027 0.58891602 0.08238667]
[0.5697928  0.51035987 0.69193021 0.10687023]
乍一看，独立和不独立的异常比例都变高了，很奇怪。
假设inde上脑内高比例为a，inde占valid比例为b，在valid集上，脑内高比例为k，求not inde上脑内高比例x
则ab+x(1-b)=k，训练集变小，则b增加，a增加（按规律），k-ab减小，但是1-b也会减少，所以(k-ab)/(a-b)可能增加
"""

################################
#check目前手标580涉及的病人的所有序列的异常比例：
print()
in_valid = {}
cnt = 0
for x in valid:
    label = label_file_opened.root.label[x[0]]
    if label[13,3]==0:
        cnt += 1
        in_valid[report_patient[x[1]][0]] = 1
        
cnt = 0
label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
for x in valid+training:
    if report_patient[x[1]][0] in in_valid:
        label = label_file_opened.root.label[x[0]]
        label_cnt += label
        label_cnt_ab += np.max(label-label_not_hand, 0)
        cnt += 1
print(cnt) #1647
print(label_cnt_ab/cnt) #[0.52641166 0.37583485 0.66666667 0.07285974]

#####################################
print()
for i in range(5): #随机从12253中抽580个序列，把这580个序列涉及的病人的全部序列抽出来，统计异常比例
    in_valid = {}
    cnt = 0
    for x in random.sample(training+valid, 580):
        in_valid[report_patient[x[1]][0]] = in_valid.get(report_patient[x[1]][0], 0) + 1
    print(len(in_valid)) #平均550左右
    cnt = 0
    label_cnt = np.zeros((20,4))
    label_cnt_ab = np.zeros((4))
    for x in valid+training:
        if report_patient[x[1]][0] in in_valid:
            label = label_file_opened.root.label[x[0]]
            label_cnt += label
            label_cnt_ab += np.max(label-label_not_hand, 0)
            cnt += 1
    print(cnt) #1700-1800，平均每个病人三个多序列
    print(label_cnt_ab/cnt) #结论：比正常高，高密度能到0.5。平均抽1700个序列。[0.54658754 0.48130564 0.65697329 0.09198813]
    print(list(filter(lambda x:x>1, [in_valid[x] for x in in_valid])))
    #结论：随机抽取序列，然后抽取这些序列涉及的病人的全部序列，会导致分布变化，异常比例变高
    #原因：检查多的病人更有可能被抽中，而检查多的病人异常比例也高

################################
    #check是否序列多的病人异常也更多
patient = {}
for x in training+valid:
    patient[report_patient[x[1]][0]] = patient.get(report_patient[x[1]][0], [])+[x]
print('num patient', len(patient)) #6437，平均每个人1.9个序列
label_cnt = {}
label_cnt_ab = {}
cnt = {}
for x in patient:
    l = len(patient[x])
    for series in patient[x]:
        label = label_file_opened.root.label[series[0]]
        label_cnt[l] = label_cnt.get(l, np.zeros((20,4)))+label
        label_cnt_ab[l] = label_cnt_ab.get(l, np.zeros((4)))+np.max(label-label_not_hand, 0)
        cnt[l] = cnt.get(l, 0)+1
for num in range(7):
    if num in cnt:
        print(num, cnt[num]/num)
        print(label_cnt_ab[num]/cnt[num])
    """
    1 3453.0
    [0.23718506 0.28033594 0.46770924 0.05560382]
    2 1986.0
    [0.3255287  0.33257805 0.67270896 0.09340383]
    3 316.0
    [0.45675105 0.42088608 0.62130802 0.07911392]
    4 357.0
    [0.48109244 0.48879552 0.64705882 0.0987395 ]
    5 78.0
    [0.53846154 0.53076923 0.60512821 0.1       ]
    6 102.0
    [0.58333333 0.5620915  0.6127451  0.12254902]
    结论：做检查次数更多的人，除脑内低外异常比例也更大
    """
########################################
print()
patient = {}
for x in training+valid:
    patient[report_patient[x[1]][0]] = patient.get(report_patient[x[1]][0], [])+[x]

for i in range(5): #从所有病人中随机取550个，抽取涉及这些病人的全部序列
    in_valid = random.sample([x for x in patient], 550)
    cnt = 0
    label_cnt = np.zeros((20,4))
    label_cnt_ab = np.zeros((4))
    for x in valid+training:
        if report_patient[x[1]][0] in in_valid:
            label = label_file_opened.root.label[x[0]]
            label_cnt += label
            label_cnt_ab += np.max(label-label_not_hand, 0)
            cnt += 1
    print(cnt) #1000-1100左右，平均每个病人两个序列
    print(label_cnt_ab/cnt) #和正常比例相近，高密度0.38左右，但浮动较大
    #结论：直接随机抽取病人，然后抽取其涉及的全部序列，分布正常
########################################
print()
for i in range(5): #从12253中随机抽1647个序列，统计异常比例
    cnt = 0
    label_cnt = np.zeros((20,4))
    label_cnt_ab = np.zeros((4))
    temp_indices = random.sample(list(range(12253)), 1647)
    for x in valid+training:
        if x[0] in temp_indices:
            label = label_file_opened.root.label[x[0]]
            label_cnt += label
            label_cnt_ab += np.max(label-label_not_hand, 0)
            cnt += 1
    print(cnt, label_cnt_ab/cnt) #结论：符合正常比例，高密度0.39左右
    #结论：如果以序列为单位随机抽取，分布正常
###################################
def count_asymmetric(label):
    """
    label是29的list
    """
    for i in range(7):
        if label[i] and label[i+7]:
            label[i] = 0
            label[i+7] = 0
    return sum(label[:15])

print()
random.seed(9) #9
patient = {}

#接下来，移除数据集中的不准确低密度描述的序列

data_file = tables.open_file(config["data_file"], "r")

bad = {}

for i in tqdm(range(len(data_file.root.subject_ids))):
    ID = data_file.root.subject_ids[i].decode('utf-8')
    with open(ID+'/sentence_pred.json', 'r') as fp:
        temp = json.load(fp)
    sentence_pred = [(x, y) for x,y in zip(temp[0], temp[1])]
    
    for x in sentence_pred:
        if x[1][19] and sum(x[1][17:])==1: #只有低密度异常
            n_organs = sum(x[1][:17])
            asymmetric = count_asymmetric(x[1])
            if n_organs>=5 and asymmetric<2:
                bad[i] = True

print('bad', len(bad), len(bad)/len(data_file.root.subject_ids))
#random.shuffle(shuffled_all)
#print('not bad', len(shuffled_all), len(shuffled_all)/len(al))
shuffled_all = training+valid

for x in shuffled_all:
    patient[report_patient[x[1]][0]] = patient.get(report_patient[x[1]][0], [])+[x]
in_test = random.sample([x for x in patient], 500)
random.seed(9) #9
in_valid = random.sample(sorted(list(set(x for x in patient)-set(in_test))), 500) #set遍历顺序是随机的，random无法控制
print(len(in_test), len(in_valid), len(set(x for x in patient)-set(in_test)-set(in_valid)))
cnt = 0
label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
label_cnt2 = np.zeros((20,4))
label_cnt_ab2 = np.zeros((4))
label_cnt3 = np.zeros((20,4))
label_cnt_ab3 = np.zeros((4))

new_training = []
new_valid = []
new_test = []

for x in shuffled_all:
    label = label_file_opened.root.label[x[0]]
    if report_patient[x[1]][0] in in_test:
        if x[0] not in bad:
            new_test.append(x[0])
            label_cnt += label
            label_cnt_ab += np.max(label-label_not_hand, 0)
    elif report_patient[x[1]][0] in in_valid:
        if x[0] not in bad:
            new_valid.append(x[0])
            label_cnt2 += label
            label_cnt_ab2 += np.max(label-label_not_hand, 0)
    else:
        if x[0] not in bad:
            new_training.append(x[0])
            label_cnt3 += label
            label_cnt_ab3 += np.max(label-label_not_hand, 0)
print(len(new_test), len(new_valid), len(new_training))
print(label_cnt_ab/len(new_test))
print(label_cnt_ab2/len(new_valid))
print(label_cnt_ab3/len(new_training))

print(len(set([al[x][1] for x in new_test])), len(set([al[x][2] for x in new_test])))
print(len(set([al[x][1] for x in new_valid])), len(set([al[x][2] for x in new_valid])))
print(len(set([al[x][1] for x in new_training])), len(set([al[x][2] for x in new_training])))

#pickle_dump(new_test, '12253_test_500p.pkl')
#pickle_dump(new_valid, '12253_valid_500p.pkl')
#pickle_dump(new_training, '12253_training_5437p.pkl')

pickle_dump(new_test, '12253_test_500p2.pkl')
pickle_dump(new_valid, '12253_valid_500p2.pkl')
pickle_dump(new_training, '12253_training_5437p2.pkl')

"""
seed(9)
969 969 10315
[0.38183695 0.38390093 0.60165119 0.10732714]
[0.37048504 0.39112487 0.59855521 0.09184727]
[0.39893359 0.39253514 0.6088221  0.08347067]
报告数 序列数
665 920
681 924
7093 9718
"""
"""
bad 1434 0.11703256345384803
859 869 9091
[0.41327125 0.42142026 0.55064028 0.10826542]
[0.39930955 0.42347526 0.55235903 0.10241657]
[0.43075569 0.42899571 0.55615444 0.08832912]
598 812
611 826
6348 8518
"""
label_file_opened.close()