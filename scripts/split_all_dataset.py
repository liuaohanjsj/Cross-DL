# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:57:13 2021

@author: Admin
"""

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
config["data_file"] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80.h5")
config["training_file"] = os.path.abspath("/data/liuaohan/co-training/all_brain_training_ids_160_80.pkl")
config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/all_brain_validation_ids_160_80.pkl")
config['label_file'] = 'all_refined_label_new1.h5'


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

'''
print(len(training), len(valid))

with open('report_patient.json', 'r') as fp:
    report_patient = json.load(fp)
with open('report_desc.json', 'r') as fp:
    report_desc = json.load(fp)

operation_ratio(al)

label_file_opened = tables.open_file(config['label_file'], "r")

##########################################################


###################################
print()
random.seed(9) #9
patient = {}
for x in training+valid:
    patient[report_patient[x[1]][0]] = patient.get(report_patient[x[1]][0], [])+[x]
print(len(patient)) #应该是19877？
in_test = random.sample([x for x in patient], 1000)
random.seed(9) #9
in_valid = random.sample(sorted(list(set(x for x in patient)-set(in_test))), 1000) #set遍历顺序是随机的，random无法控制
print(len(in_test), len(in_valid), len(set(x for x in patient)-set(in_test)-set(in_valid)))
cnt = 0
label_cnt = np.zeros((20,4))
label_cnt_ab = np.zeros((4))
label_cnt2 = np.zeros((20,4))
label_cnt_ab2 = np.zeros((4))
label_cnt3 = np.zeros((20,4))
label_cnt_ab3 = np.zeros((4))
label_not_hand = np.zeros((20,4))
label_not_hand[0,3] = 1

new_training = []
new_valid = []
new_test = []

for x in valid+training:
    label = label_file_opened.root.label[x[0]]
    if report_patient[x[1]][0] in in_test:
        new_test.append(x[0])
        label_cnt += label
        label_cnt_ab += np.max(label-label_not_hand, 0)
    elif report_patient[x[1]][0] in in_valid:
        new_valid.append(x[0])
        label_cnt2 += label
        label_cnt_ab2 += np.max(label-label_not_hand, 0)
    else:
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
'''
#pickle_dump(new_test, '38713_test_1000p.pkl')
#pickle_dump(new_valid, '38713_valid_1000p.pkl')
#pickle_dump(new_training, '38713_training_17877p.pkl')


"""
19877
1000 1000 17877
2014 2094 34605
高低密度：
[0.30784508 0.61717974 0.         0.04816286]
[0.3252149  0.62082139 0.         0.05635148]
[0.31755527 0.60401676 0.         0.06054038]
4种异常：
[0.20357498 0.18321748 0.59334657 0.06852036]
[0.22445081 0.19054441 0.60315186 0.06733524]
[0.21095217 0.19835284 0.58297934 0.05762173]
不同报告数 不同序列数
1388 1957
1434 2007
24264 33489
"""
#label_file_opened.close()
def filter_dataset_age(data_file, pkl_file, f):
    def getage(seriesID):
        age = series_info[seriesID]['pAge']
        if 'Y' in age:
            return int(age[:3])
        return 0
    
    with open('all_report.json', 'r') as fp:
        report = json.load(fp)
    series_info = {}
    for reportID in report:
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                series_info[series['seriesID']] = series
        
    data = tables.open_file(data_file, "r")
    ret = []
    indices = pickle_load(pkl_file)
    print(len(indices))
    for index in indices:
        ID = data.root.subject_ids[index].decode('utf-8')
        [reportID, seriesID] = resolve_subject_id(ID)
        age = getage(seriesID)
        if f(age):
            ret.append(index)
        #training_set.append([index] + resolve_subject_id(ID))
    print(len(ret))
    return ret

def filter_dataset_bad(data_file, pkl_file, verbose = True):
    def standard_path(path):
        """
        path是一个序列路径，现在要知道路径的标准路径，只要最后三层的路径，
        而且如果path是在本机和服务器运行的，斜杠的用法也不同，现在转换为相同的。
        path为str，返回str
        """
        path = path.replace('\\', '/') #2021.7.20发现bug，linux应该用/
        reportID = Path(path).parent.parent.name
        studyID = Path(path).parent.name
        seriesID = Path(path).name
        return reportID+'/'+studyID+'/'+seriesID

    def is_good(series_path, bad):
        """
        series_path是原始序列路径，即里面可能没有CT而是用series_link
        1: 不是好的报告-序列对，如报告是眼眶报告、报告和序列不匹配
        2: 无CT.json，应该不存在此问题
        3: 序列过短
        4: 序列为骨窗重建（序列长度和正常一样，但噪声大）
        5: bad序列，即配准不好或者本身序列图像有问题的
        """
        #对good进行判断，用原始路径
        try:
            with open(series_path+'/good.txt') as fp:
                x = fp.read()
                if 'nearest within 1 day' not in x: #全部数据集
                    if verbose:
                        print('not good report', series_path)
                    return False, 1
        except:
            if verbose:
                print('cannnot find good', series_path)
            return False, 1
        #下面对CT本身进行判断，所以要读取series_link
        if Path(series_path+'/series_link.txt').exists():
            with open(series_path+'/series_link.txt', 'r') as fp:
                series_path = fp.readline().strip()
        if not Path(series_path+'/CT.json').exists():
            if verbose:
                print('no CT json', series_path)
            return False, 2
        with open(series_path+'/CT.json') as fp:
            temp = json.load(fp)
            if 'SliceTiming' not in temp: 
                #有的CT.json没有这个条目，但是还是有正常CT图像的
                with open(series_path+'/central_axis_correction.json', 'r') as fp:
                    ac = json.load(fp)
                if len(ac[0])<10:
                    if verbose:
                        print('series too short', series_path)
                    return False, 3
                #all_CT = list(Path(series_path).glob('*brain*.png')) #生成全部数据集
                #if len(all_CT)<10:
                #    print('series too short', series_path)
                #    return False, 3
            elif len(temp['SliceTiming'])<10: #序列长度过短的，去除
                if verbose:
                    print('series too short', series_path)
                return False, 3
            if 'SeriesNumber' in temp and temp['SeriesNumber']==202: #5.28.2021生成全部数据集时加的
                if verbose:
                    print('bone window', series_path) #噪声粒度比较大
                return False, 4
        if standard_path(series_path) in bad:
            if verbose:
                print('bad series', series_path)
            return False, 5
        return True, 0
    
    bad = {}
    for i in range(34):
        with open('/home/liuaohan/pm-stroke/aligned_registration/%s/registration_check_result.json'%str(i).zfill(2), 'r') as fp:
            registration_check = json.load(fp)
        for x in registration_check:
            if registration_check[x]>=3:
                bad[standard_path(x)] = True
    if type(data_file)==str: #如果是str，就load文件。否则就应该已经是series路径list
        data = tables.open_file(data_file, "r")
        series_paths = [x.decode('utf-8') for x in data.root.subject_ids]
    else:
        series_paths = data_file
    ret = [[] for i in range(10)]
    indices = pickle_load(pkl_file) if type(pkl_file)==str else pkl_file
    print(len(indices))
    cnt = np.zeros(10)
    for index in tqdm(indices):
        series_path = series_paths[index]
        [reportID, seriesID] = resolve_subject_id(series_path)
        temp = is_good(series_path, bad)
        cnt[temp[1]] += 1
        ret[temp[1]].append(index)
    print(len(ret), cnt)
    return ret

def filter_dataset_diagnose(data_file, pkl_file, f):
    
    with open('all_report.json', 'r') as fp:
        report = json.load(fp)
    series_info = {}
    for reportID in report:
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                series_info[series['seriesID']] = series
    with open('report_desc2.json', 'r') as fp:
        report_desc = json.load(fp)
    
    data = tables.open_file(data_file, "r")
    ret = []
    indices = pickle_load(pkl_file)
    print('ori', len(indices))
    for index in indices:
        ID = data.root.subject_ids[index].decode('utf-8')
        [reportID, seriesID] = resolve_subject_id(ID)
        diagnose = report_desc[reportID][1]
        if f(diagnose):
            ret.append(index)
        #training_set.append([index] + resolve_subject_id(ID))
    print('ret', len(ret))
    return ret
def filter_dataset_by(pkl_file, filtered_pkl_file):
    ret = []
    indices = pickle_load(pkl_file)
    filtered_indices = pickle_load(filtered_pkl_file)
    print(len(filtered_indices)) #34204
    for index in indices:
        if index in filtered_indices:
            ret.append(index)
    print(len(indices), len(ret))
    return ret
    """
    1887 1561
    1937 1594
    32170 27330
    """
    """
    1887 1607
    1937 1699
    32170 28472
    """
'''
for file_name in ['38713_test_1000p', '38713_valid_1000p', '38713_training_17877p']:
    indices = filter_dataset_age("/data/liuaohan/co-training/all_brain_data_160_80.h5", "/data/liuaohan/co-training/%s.pkl"%file_name,
                             lambda x:x>=60)
    pickle_dump(indices, '%s_old.pkl'%file_name)
    #>=60: 965 1051 16291

for file_name in ['38713_test_1000p', '38713_valid_1000p', '38713_training_17877p']:
    indices = filter_dataset_age("/data/liuaohan/co-training/all_brain_data_160_80.h5", "/data/liuaohan/co-training/%s.pkl"%file_name,
                             lambda x:x<60)
    pickle_dump(indices, '%s_young.pkl'%file_name)
    #<60: 1049 1043 18314
'''
'''
for file_name in ['38713_test_1000p', '38713_valid_1000p', '38713_training_17877p']:
    indices = filter_dataset_bad("/data/liuaohan/co-training/all_brain_data_160_80.h5", "/data/liuaohan/co-training/%s.pkl"%file_name)
    for i in range(10):
        if len(indices[i]):
            pickle_dump(indices[i], '%s_%d.pkl'%(file_name, i))
    """
    test 1887 57 0 12 26 32
    valid 1937 87 0 10 22 38
    training 32170 1116 0 258 376 685
    """
'''
'''
for file_name in ['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']:
    indices = filter_dataset_diagnose("/data/liuaohan/co-training/all_brain_data_160_80.h5", "/data/liuaohan/co-training/%s.pkl"%file_name,
                                      lambda x:('缺血' in x or '脑梗' in x or '软化' in x) and '老年' not in x)
    pickle_dump(indices, '%s_ischemia_young.pkl'%file_name)
    # 557 601 9253

for file_name in ['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']:
    indices = filter_dataset_diagnose("/data/liuaohan/co-training/all_brain_data_160_80.h5", "/data/liuaohan/co-training/%s.pkl"%file_name,
                                      lambda x:('缺血' in x or '脑梗' in x or '软化' in x) and '老年' in x)
    pickle_dump(indices, '%s_ischemia_old.pkl'%file_name)
    #311 276 4803
'''
'''
for file_name in ['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']:
    indices = filter_dataset_by("/data/liuaohan/co-training/%s.pkl"%file_name, 'all_filtered.pkl')
    pickle_dump(indices, '%s_not_too_many.pkl'%file_name)
'''
'''
for file_name in ['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']:
    indices = filter_dataset_by("/data/liuaohan/co-training/%s.pkl"%file_name, 'all_filtered2.pkl')
    #pickle_dump(indices, '%s_ischemia_false.pkl'%file_name)
'''

'''
#以下四个for在2022.2.27不小心多运行了几次，不知道有没有影响 。

#其实all_filtered2.pkl和all_filtered3、all_filtered4都是一样的
for i, file_name in enumerate(['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']):
    indices = filter_dataset_by("/data/liuaohan/co-training/%s.pkl"%file_name, 'all_filtered2.pkl')
    random.seed(10)
    #if i==1: #valid
    #    indices = random.sample(indices, 500)
    if i==2:
        indices = random.sample(indices, 2000)
    pickle_dump(indices, '%s_ischemia_false_small1.pkl'%file_name)


for i, file_name in enumerate(['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']):
    indices = filter_dataset_by("/data/liuaohan/co-training/%s.pkl"%file_name, 'all_filtered2.pkl')
    random.seed(10)
    #if i==1: #valid
    #    indices = random.sample(indices, 500)
    if i==2:
        indices = random.sample(indices, 5000)
    pickle_dump(indices, '%s_ischemia_false_small2.pkl'%file_name)

for i, file_name in enumerate(['38713_test_1000p_0', '38713_valid_1000p_0', '38713_training_17877p_0']):
    indices = filter_dataset_by("/data/liuaohan/co-training/%s.pkl"%file_name, 'all_filtered2.pkl')
    random.seed(10)
    #if i==1: #valid
    #    indices = random.sample(indices, 500)
    if i==2:
        indices = random.sample(indices, 10000)
    pickle_dump(indices, '%s_ischemia_false_small3.pkl'%file_name)
'''
