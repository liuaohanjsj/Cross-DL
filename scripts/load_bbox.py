import json
import numpy as np
import xlrd
from tqdm import tqdm
import re
from pathlib import Path
import pydicom
import cv2
import os
import glob

def stw(img):
    img = np.minimum(np.maximum(img, -10), 80)
    img = (img+10)*254//90
    return img
def load_bboxes(bbox_path, scale1 = 1, scale2 = 1):
    bboxes = []
    with open(bbox_path,'r') as f:
        json_obj = json.load(f)
        
        if len(json_obj['shapes']) > 0:
            for box in json_obj['shapes']:
                label = box['label']
                box = box['points']
                box = np.array(box).astype(np.float)
                box = box * 512 / 200
                box = box.astype(np.int)
                if box[1][0] > box[0][0]:
                    left = box[0][0]
                    right = box[1][0]
                else:
                    left = box[1][0]
                    right = box[0][0]
                if box[1][1] > box[0][1]:
                    top = box[0][1]
                    bottom = box[1][1]
                else:
                    top = box[1][1]
                    bottom = box[0][1]
                new_box = [int(scale1*left), int(scale2*top), int(scale1*right), int(scale2*bottom)]
                bboxes.append((new_box, label))

    return bboxes

def split_sentence(s):
    seps = ['，左', '，右', '，两', '，双']
    ret = [s]
    for sep in seps:
        mem = ret.copy()
        ret = []
        for s in mem:
            temp = s.split(sep)
            for i in range(1,len(temp)):
                temp[i] = sep[1:]+temp[i]
            ret.extend(temp)
    return ret

def load(file, report, filt=False):
    """
    从原始xls文件中读取报告，生成{reportID:[desc1, desc2..]}的dict
    如果filt，则去除所有非头部CT平扫的报告
    """
    no_need = ['鼻窦', '颌面部', '眼眶', '腮腺', '颞骨 ','视神经', '鼻骨', 'CTP', 
               '颞骨', '颈椎', '颞颌关节', '脊椎']
    data = xlrd.open_workbook(file)
    sh = data.sheet_by_index(0)
    n = sh.nrows
    col10 = sh.col_values(10)
    col11 = sh.col_values(11)
    col1 = sh.col_values(1)
    col9 = sh.col_values(9)
    for i in tqdm(range(1, n)):
        #para = sh.col_values(9)[j]
        desc = col10[i]
        Id = col1[i]
        patientID = sh.cell_value(i, 2)
        if filt and any((x in col9[i]) for x in no_need):
            continue
        #按标点拆分成句子：
        sentences = re.split(r'。|；',desc.strip('。；'))
        #拆分长度过长句子：
        res = []
        for sentence in sentences: #按 “，+方位词”分词
            res.extend(split_sentence(sentence))
        if Id in report:
            #print('report ID already exists!', Id)
            if (res, col11[i])!=report[Id]:
                print('different report in 2 xls', report[Id], (res, col11[i]))
            continue
        #if (patientID, col11[i]) in report and report[(patientID, col11[i])]!=res:
        #    print('same key', (patientID, col11[i]), report[(patientID, col11[i])], res)

        report[Id] = (res, col11[i])
    pass

def arrange_yuwei_labeled_data():
    report_desc = dict()
    load('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/CTHead301_1.xlsx', report_desc)
    load('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/CTHead301_3.xlsx', report_desc)
    with open('all_report.json', 'r') as fp:
        report = json.load(fp)
    series_match = {} #series_match[seriesID] = [reportID1, reportID2, ..]
    for reportID in report:
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                if series['seriesID'] not in series_match:
                    series_match[series['seriesID']] = []
                series_match[series['seriesID']].append(reportID) 
    
    print(len(report))
    with open('valid.json', 'r') as fp:
        cases = json.load(fp)
    print('all series', len(cases))
    for case in tqdm(cases[:]):
        #print(case)
       # boxes = load_bboxes(cases[i]['bbox_paths'])
        #print(len(case['bbox_paths']), len(case['dcm_paths']))
        #print(case['report'])
        studyID = Path(case['dcm_paths'][0]).parent.parent.name
        seriesID = Path(case['dcm_paths'][0]).parent.name
        new_dir = 'yuwei_data/%s/%s'%(studyID, seriesID)
        assert len(set([Path(x).parent.name for x in case['dcm_paths']]))==1 
        if seriesID not in series_match:
            #print('cannot find series')
            continue
        if Path('yuwei_data/%s/%s'%(studyID, seriesID)).exists():
            continue
        Path('yuwei_data/%s/%s'%(studyID, seriesID)).mkdir(parents=True, exist_ok=True)
        
        #seriesID匹配的报告中，筛选疾病诊断和宇巍给的一样的：
        matched_reports = list(filter(lambda x:report_desc[x][1]==case['report'], [float(x) for x in series_match[seriesID]]))
        if len(matched_reports)==0: #极少
            continue
        #match_reports大部分长度都是1
        with open(new_dir+'/reportID.txt', 'w') as fp:
            fp.write(str(matched_reports[0]))
        
        with open(new_dir+'/report.json', 'w') as fp:
            json.dump(report_desc[matched_reports[0]][0], fp, ensure_ascii=False)
       # matched_reports[0][0]
        
        i = 0
        for box_path, dcm_path in zip(case['bbox_paths'], case['dcm_paths']):
            i += 1
            img = pydicom.dcmread(dcm_path)
            img = np.float64(img.pixel_array)
            img = np.uint8(stw(img-1024))
            img = np.stack((img,img,img), 2)
            if box_path is not None:
                with open(box_path, 'r') as f:
                    json_obj = json.load(f)
                with open(new_dir+'/%s.json'%str(i).zfill(3), 'w') as fp:
                    json.dump(json_obj, fp)
                boxes = load_bboxes(box_path)
                for box in boxes:
                    if box[1]=='1': #低密度
                        cv2.rectangle(img, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0,255,0))
                    else:
                        cv2.rectangle(img, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0,255,255))
            cv2.imwrite('yuwei_data/%s/%s/%s.png'%(studyID, seriesID, str(i).zfill(3)), img)

def count_asymmetric(label):
    """
    label是29的list
    """
    for i in range(7):
        if label[i] and label[i+7]:
            label[i] = 0
            label[i+7] = 0
    return sum(label[:15])

#arrange_yuwei_labeled_data()
all_paths = list(glob.glob(os.path.join('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/yuwei_data', "*", "*")))
cnt = 0
cnt_n = np.zeros(20)
for path in all_paths:
    if not Path(path+'/sentence_pred.json').exists():
        continue
    with open(path+'/sentence_pred.json', 'r') as fp:
        temp = json.load(fp)
    sentence_pred = [(x, y) for x,y in zip(temp[0], temp[1])]
    flag = False
    for x in sentence_pred:
        if x[1][19] and not x[1][16] and sum(x[1][17:])==1:
            n_organs = sum(x[1][:17])
            cnt_n[n_organs] += 1
            if n_organs>=5:
                asymmetric = count_asymmetric(x[1])
                if asymmetric>=2:
                    print('asymmetric', x[0])
                    continue
                flag = True
                print(x[0])
    if flag:
        cnt += 1
        print(Path(path).name)
print(cnt/len(all_paths))
print(cnt_n[:10])