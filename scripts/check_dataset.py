# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:37:10 2021

@author: Admin
"""

import json
import tables
import pickle
from pathlib import Path
import glob
import numpy as np
import time
import random
import pydicom
from tqdm import tqdm
from split_all_dataset import filter_dataset_bad
import shutil
import os

def count_CT_dataset():
    with open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl", 'rb') as fp:
        train = pickle.load(fp)
    print(len(train))
    with open("/data/liuaohan/co-training/38713_valid_1000p_0_ischemia_false.pkl", 'rb') as fp:
        train = pickle.load(fp)
    print(len(train))
    with open("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl", 'rb') as fp:
        train = pickle.load(fp)
    print(len(train))

def count_retrospective():
    def update_sex(sex, now): #sex:'M'
        if now=='':
            return sex
        if sex=='':
            return now
        #if now!=sex:
        #    print('fuck', now, sex)
        return sex
    def update_age(age, now): #age:'063Y'
        if len(age):
            return age
        return now
    def int_age(age):
        if age[-1]=='M' or age[-1]=='D' or age[-1]=='W':
            return 0
        assert age[-1]=='Y', age
        return int(age[:-1])
    def find_series_path(patient_exam):
        #给每个series找到\data里的dcm路径，以便读取dcm的namufacturer信息
        series_path = {}
        for pID in patient_exam:
            for studyID in patient_exam[pID]:
                for series in patient_exam[pID][studyID]:
                    #可能同一series对应path不同的情况，但是看了一个是studyID和seriesID都没错，但是在不同文件夹下，就是说重复的
                    #assert series['seriesID'] not in series_path or series_path[series['seriesID']]==series['path'],\
                    #    (studyID, series, series_path[series['seriesID']])
                    series_path[series['seriesID']] = series['path']
        return series_path
    
    with open('report_patient.json', 'r') as fp: #怎么来的？
        report_patient = json.load(fp) #{reportID:[patientID, [sentence1, sentence2...]]}
    print(len(report_patient)) #251463个report
    patients = {}
    for reportID in report_patient:
        patients[report_patient[reportID][0]] = True
    print(len(patients)) #201383个patient
    
    #统计retrospective CT数据集：
    with open('/home/liuaohan/pm-stroke/project/EHR/json/exam_all.json', 'r') as fp:
        patient_exam = json.load(fp)
    print(len(patient_exam)) #49132
    series_path = find_series_path(patient_exam)
    
    data_file = tables.open_file("/data/liuaohan/co-training/all_brain_data_160_80.h5", "r")
    #report_ID = [] #report_ID[i]是data_file中第i个序列的报告ID
    subject_ids = []
    for x in data_file.root.subject_ids:
        subject_ids.append(x.decode('utf-8'))
        #report_ID.append(Path(x.decode('utf-8')).parent.parent.name)
    data_file.close()
    print('original CT series', len(subject_ids)) #38713
    
    sex_cnt, manu_cnt, model_cnt = {}, {}, {}
    series_info = {}
    cnt_weird = 0
    ages, nslices, pixelspacings, thicknesses = [],[],[],[]
    if Path('retrospective_info.json').exists():
        with open('retrospective_info.json', 'r') as fp:
            series_info = json.load(fp)
    for file in ['38713_training_17877p_0_ischemia_false', '38713_test_1000p_0_ischemia_false', '38713_valid_1000p_0_ischemia_false']:
    #for file in ['38713_training_17877p_0_ischemia_false']:
        with open("/data/liuaohan/co-training/%s.pkl"%file, 'rb') as fp:
            ids = pickle.load(fp)
        patients = {}
        for i,x in enumerate(tqdm(ids)):
            if str(x) in series_info:
                manu = series_info[str(x)]['manufacturer']
                pixelspacing = series_info[str(x)]['pixelspacing']
                slicethickness = series_info[str(x)]['slicethickness']
                model = series_info[str(x)]['model']
                nslice = series_info[str(x)]['nslice']
            else:
                seriesID = Path(subject_ids[x]).name
                paths = list(Path('/data/ctclass/RAW/'+series_path[seriesID]).glob('*.dcm'))
                assert len(paths)
                dcm = pydicom.read_file(str(paths[0]))
                manu = str(dcm.Manufacturer)
                pixelspacing = str(dcm.PixelSpacing)
                slicethickness = str(dcm.SliceThickness)
                model = str(dcm.ManufacturerModelName)
                nslice = len(paths)
                series_info[x] = {'manufacturer': manu, 'pixelspacing':pixelspacing, 'slicethickness':slicethickness,
                           'nslice': nslice, 'model':model}
                
            nslices.append(nslice)
            pixelspacings.append(json.loads(pixelspacing)[0])
            thicknesses.append(json.loads(slicethickness))
            manu_cnt[manu] = manu_cnt.get(manu, 0)+1
            model_cnt[model] = model_cnt.get(model, 0)+1
            
            pID = report_patient[Path(subject_ids[x]).parent.parent.name][0]
            assert pID in patient_exam, pID
            sex,age = '',''
            weird = 0
            for studyID in patient_exam[pID]:
                for series in patient_exam[pID][studyID]:
                    if len(sex) and update_sex(series['pSex'], sex) != sex:
                        weird  = 1 #真的nb
                    #有的是相同patientID对应了两个人，比如Y2761185，姓名年龄都不同
                    #有的应该是一个人。但是性别真的不同G079929
                    sex = update_sex(series['pSex'], sex)
                    age = update_age(series['pAge'], age)
            cnt_weird += weird
            sex_cnt[sex] = sex_cnt.get(sex, 0)+1
            if len(age):
                ages.append(int_age(age))
            patients[pID] = 1
            if i%1000==0:
                print(i)
                with open('retrospective_info_temp.json', 'w') as fp:
                    json.dump(series_info, fp)
        print(len(ids), len(patients))
        #1607 907
        #1699 902
        #28472 16316
        print(manu_cnt)
        #{'UIH': 243, 'SIEMENS': 941, 'GE MEDICAL SYSTEMS': 423}
        #{'UIH': 258, 'SIEMENS': 990, 'GE MEDICAL SYSTEMS': 451}
        #{'SIEMENS': 17113, 'GE MEDICAL SYSTEMS': 7146, 'UIH': 4213}
        
        print(manu_cnt) #{'SIEMENS': 19044, 'GE MEDICAL SYSTEMS': 8020, 'UIH':4714}
        print(model_cnt)
        print(sex_cnt, cnt_weird) #{'F': 13934, 'M': 17844} 84。0.264%的人出现了离奇性转现象
        ages.sort()
        print(len(ages), sum(ages)/len(ages), ages[len(ages)//4], ages[len(ages)*3//4]) #31778 54.342092013342565 41 69
        pixelspacings.sort()
        n = len(pixelspacings)
        print('pixel spacing', sum(pixelspacings)/n, pixelspacings[n//4], pixelspacings[n*3//4])
        nslices.sort()
        n = len(nslices)
        print('n slice', sum(nslices)/n, nslices[n//4], nslices[n*3//4])
        thicknesses.sort()
        n = len(thicknesses)
        print('thickness', sum(thicknesses)/n, thicknesses[n//4], thicknesses[n*3//4])
    with open('retrospective_info.json', 'w') as fp:
        json.dump(series_info, fp)
def count_NLP():
    data_file = tables.open_file("/data/liuaohan/co-training/all_brain_data_160_80.h5", "r")
    subject_ids = []
    for x in data_file.root.subject_ids:
        subject_ids.append(x.decode('utf-8'))
        #report_ID.append(Path(x.decode('utf-8')).parent.parent.name)
    data_file.close()
    
    with open('report_desc2.json', 'r') as fp:
        report_desc = json.load(fp)
    cnt_sentence = 0
    for file in ['38713_test_1000p_0_ischemia_false', '38713_valid_1000p_0_ischemia_false', '38713_training_17877p_0_ischemia_false']:
        with open("/data/liuaohan/co-training/%s.pkl"%file, 'rb') as fp:
            ids = pickle.load(fp)
        for x in ids:
            reportID = Path(subject_ids[x]).parent.parent.name
            cnt_sentence += len(report_desc[reportID][0])
    print(cnt_sentence) #125430

def count_prospective():
    data_file = tables.open_file("/data/liuaohan/co-training/prospective_160_80.h5", "r")
    with open('/data/liuaohan/co-training/prospective_good6_1500.pkl', 'rb') as fp:
        index = pickle.load(fp)
    patients = {}
    series = {}
    for i in index:
        patient = Path(data_file.root.subject_ids[i].decode('utf-8')).parent.parent.name
        s = Path(data_file.root.subject_ids[i].decode('utf-8')).name
        patients[patient] = True
        series[s] = True
    data_file.close()
    print(len(index), len(patients))
    
    with open('/home/liuaohan/pm-stroke/project/EHR/propective_series_info_more.json', 'r') as fp: #由EHR/check_series里生成
        series_info = json.load(fp)
    print('series_info', len(series_info))
    
    wl = {}
    manu = {}
    patient_info = {}
    for x in series_info:
        p = series_info[x]['patient']
        if p not in patients:
            continue
        if p in patient_info and patient_info[p]['sex']!=series_info[x]['sex']:
            print('daihen', patient_info[p], series_info[x])
        patient_info[p] = {'sex':series_info[x]['sex'], 'birth':series_info[x]['birth'],
                    'time': series_info[x]['time']} #怎么按人统计年龄？一个人做了多个检查怎么算检查时间？
    
    manu, model = {}, {}
    pixelspacing, thickness, n_slice = [], [], []
    for x in series:
        if x not in series_info:
            print('Error, cannot find series info', x)
        manu[series_info[x]['manufacturer']] = manu.get(series_info[x]['manufacturer'], 0)+1
        model[series_info[x]['model']] = model.get(series_info[x]['model'], 0)+1
        
        temp = json.loads(series_info[x]['wc'])
        if type(temp)==list:
            temp = temp[0]
        wl[temp] = wl.get(temp, 0)+1
        pixelspacing.append(json.loads(series_info[x]['pixelspacing'])[0])
        thickness.append(json.loads(series_info[x]['slicethickness']))
        n_slice.append(series_info[x]['nslice'])
    print(wl) #{40: 515, 30: 724, 35: 250, 800: 3, 45: 1, 300: 7}
    print(manu) #{'GE MEDICAL SYSTEMS': 583, 'UIH': 858, 'Philips': 59}
    print(model)
    pixelspacing.sort()
    n = len(pixelspacing)
    print('pixel spacing', sum(pixelspacing)/n, pixelspacing[n//4], pixelspacing[n*3//4])
    thickness.sort()
    n = len(thickness)
    print('slice thickness', sum(thickness)/n, thickness[n//4], thickness[n*3//4])
    n_slice.sort()
    n = len(n_slice)
    print('n slice', sum(n_slice)/n, n_slice[n//4], n_slice[n*3//4])
    
    sex = {}
    age = []
    for x in patient_info:
        sex[patient_info[x]['sex']] = sex.get(patient_info[x]['sex'], 0)+1
        birth = patient_info[x]['birth']
        if len(birth)==8:
            truncated = 0
            if int(birth[:4])<1971:
                truncated = 1971-int(birth[:4])
                birth = '1971'+birth[4:]
            timeArray = time.strptime(birth, "%Y%m%d")
            #print(series_info[x]['birth'], timeArray)
            birth = time.mktime(timeArray)#转换为时间戳
            age.append(patient_info[x]['time'] - (birth - truncated*86400*365))
    
    print(sex, sex['M']/sex['F']) #{'F': 511, 'M': 685, 'O': 1} 1.340508806262231
    age = [x/86400/365 for x in age] 
    age.sort()
    n = len(age)
    print(n, sum(age)/n, age[n//4], age[n*3//4]) #1168 55.35048851149895 41.351631690766105 69.98254356925418

'''
with open('/data/liuaohan/co-training/38713_test_1000p.pkl', 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp)) #2014
with open("/data/liuaohan/co-training/38713_test_1000p_0.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp)) #1887，去除了不好的CT
with open("/data/liuaohan/co-training/38713_valid_1000p_0.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp))
with open("/data/liuaohan/co-training/38713_training_17877p_0.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp))

with open("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp)) #1607，去除了描述同时有缺血和低密度、软化等
gt_label = tables.open_file('all_refined_label_ischemia_false_naoshi.h5', "r")
cnt = 0
for x in temp:
    if gt_label.root.label[x][12,3]==0 and (np.max(gt_label.root.label[x], 1))[7]==0:
        cnt += 1
print(cnt) #1403，去除非手标的、包含不明区域的
gt_label.close()
with open("/data/liuaohan/co-training/38713_valid_1000p_0_ischemia_false.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp))
with open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl", 'rb') as fp:
    temp = pickle.load(fp)
print(len(temp))
'''

def check_manual_filter():
    """
    配准效果不好的需要手工检查，现在看下大概占多少比例
    """
    cnt = 0
    for i in range(34):
        with open('Y:/aligned_registration/%s/registration_check_result.json'%str(i).zfill(2), 'r') as fp:
            check_result = json.load(fp)
        cnt += len(check_result)
    print(cnt) #4875


count_retrospective()
#count_prospective()
#count_NLP()
#check_manual_filter()
    
'''
with open('all_report.json', 'r') as fp:
    report = json.load(fp)
print(len(report), type(report)) #34708
cnt = 0
series_set = {}
for reportID in report:
    for studyID in report[reportID]:
        for series in report[reportID][studyID]:
            cnt += 1
            series_set[series['seriesID']] = 1
print(cnt) #53429
print(len(series_set)) #48265
'''
'''
for i in range(0, 34):
    series_paths = list(glob.glob("/home/liuaohan/pm-stroke/aligned_registration/%s/*/*/*"%str(i).zfill(2)))
    print(len(series_paths))
    filter_result = filter_dataset_bad(series_paths, list(range(len(series_paths))), verbose = False)
    with open('paper/data_%s.json'%str(i).zfill(2), 'w') as fp:
        json.dump([series_paths, filter_result], fp)
'''

'''
data = tables.open_file("/data/liuaohan/co-training/all_brain_data_160_80.h5", 'r')
subject_ids = [x.decode('utf-8') for x in data.root.subject_ids]
data.close()
cnt = 0
cnt_class = np.zeros(10)
for i in tqdm(range(34)):
    with open('paper/data_%s.json'%str(i).zfill(2), 'r') as fp:
        temp = json.load(fp)
        series_paths = temp[0]
        filter_result = temp[1]
    print(len(series_paths), len(filter_result[0]))
    cnt += len(series_paths)
    for j in range(6):
        temp = len(list(filter(lambda x:series_paths[x] in subject_ids, filter_result[j])))
        cnt_class[j] += temp
print(cnt) #43494
print(np.sum(cnt_class), cnt_class) #35994  1260 0  280  424  755。和为38713
#38713序列-报告对。脑部CT，报告EXAM_CLASS=CT，EXAM_SUB_CLASS= head and neck or brain
#去掉1260不好的匹配对（报告是眼眶报告等，或报告不是CT最近的报告）37453
#去掉280过短的序列 37173
#去掉424骨窗重建序列 36479
#去掉755手动检查配准有严重问题的，或图像有问题的 35994
#35994和进行低密度筛查前的数据集一致。32170+1937+1887=35994
#筛除低密度包含缺血且无法准确定位的。31778 '缺血' in diag and ('脑梗' in diag or '软化' in diag or '低密度' in diag)
'''
'''
cnt = 0
cnt1 = 0
for i in range(0,34):
    with open('/home/liuaohan/pm-stroke/project/EHR/all_report_ex%d_%d.json'%(i, i), 'r') as fp:
        report = json.load(fp)
    tot = 0
    for reportID in report:
        tot += 1
        if tot<i*1000 or tot>=(i+1)*1000:
            continue
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                cnt += 1
                if 'location' in series and series['location'][1]-series['location'][0]>100:
                    cnt1 += 1
    print(tot)
print(cnt, cnt1) #52386 43833
'''
def check_pair():
    with open('/home/liuaohan/pm-stroke/project/EHR/json/exam_all.json', 'r') as fp:
        temp = json.load(fp)
    with open('/home/liuaohan/pm-stroke/project/EHR/series_desc.json', 'r') as fp:
        series_desc = json.load(fp)
    cnt = 0
    series_set = {}
    patient_set = {}
    print(len(temp)) #49132个病人
    minn = 100000000
    maxn = 0
    for ID in temp:
        for study in temp[ID]:
            cnt += len(temp[ID][study])
            for series in temp[ID][study]:
                if series['seriesDesc'] not in series_desc:
                    continue
                if series_desc[series['seriesDesc']]!=1:
                    continue
                if series['modality']!='CT':
                    continue
                series_set[series['seriesID']] = 1
                patient_set[ID] = 1
                date = series['seriesDate']
                minn = min(minn, int(date))
                maxn = max(maxn, int(date))
                #timeArray = time.strptime(date, "%Y%m%d")
                #exam_time = time.mktime(timeArray)
    print(cnt) #541652
    print(len(patient_set)) #除去不是正确脑部CT的，剩27162个病人
    print(len(series_set)) #用series desc筛，剩56535个。如果只用modality=CT筛，竟然有541652个序列。其他的都是什么？
    print(minn, maxn) #20080922 20200323

'''
with open('/home/liuaohan/pm-stroke/project/EHR/series_desc.json', 'r') as fp:
    series_desc = json.load(fp)
for x in series_desc:
    if series_desc[x]==1:
        print(x)
'''
'''
#给医生标注异常区域分割生成数据：
data_file = tables.open_file("/data/liuaohan/co-training/all_head_data_160_80_ns.h5", "r")
label_file = tables.open_file("all_refined_label_ischemia_false_naoshi.h5", "r")
print(len(data_file.root.subject_ids))
print(len(label_file.root.label))
with open('/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl', 'rb') as fp:
    ids = pickle.load(fp)
print(len(ids))
Path('/home/liuaohan/pm-stroke/segmentation2').mkdir(exist_ok=True)
random.seed(10)
print(ids)
random.shuffle(ids)
print(ids)
use = ids[:100]
cnt = 0
#paths = list(Path('/home/liuaohan/pm-stroke/segmentation2').glob('*'))
#for x in paths:
#    os.removedirs(str(x))
for x in use:
    dir = data_file.root.subject_ids[x].decode('utf-8')
    paths = list(Path(dir).glob('*CT_check*.png'))
    if len(paths)>50:
        continue
    label = np.sum(label_file.root.label[x])
    print(label)
    if label==0:
        continue
    cnt += 1
    #continue
    new_dir = '/home/liuaohan/pm-stroke/segmentation2/'+str(Path(dir).parent.parent.parent.name)+'_'+\
        str(Path(dir).parent.parent.name)+'_'+str(Path(dir).parent.name)+'_'+str(Path(dir).name)
    Path(new_dir).mkdir(exist_ok=True)
    for path in paths:
        new_path = new_dir+'/'+path.name.replace('CT_check','CT')
        shutil.copy(str(path), str(new_path))
print(cnt)
data_file.close()
label_file.close()
'''
#config["data_file"] = os.path.abspath("/data/liuaohan/co-training/all_head_data_160_80_ns.h5") #ns是no ischemia
#    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl")
#    config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/38713_valid_1000p_0_ischemia_false.pkl")
#    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl")