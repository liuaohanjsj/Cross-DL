# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:15:37 2020

@author: Admin
"""
import cv2
import json
from pathlib import Path
import numpy as np
import nibabel as nib

def change_seg(img, pred, organs, d):
    """
    img是160*1760图片，pred是四维array，d是pred中的深度（调用前需要换算）
    """
    CT = img[:,:160]

    seg = np.max(np.stack([pred[organ, :, :, d] for organ in organs], 0), 0)
    seg = cv2.resize(seg, (160,160), interpolation=cv2.INTER_NEAREST)
    seg = np.transpose(seg, [1,0])
    seg = cv2.flip(seg, 0)
    heat_img = cv2.applyColorMap(np.uint8(seg*255), cv2.COLORMAP_JET)
    seg_add = cv2.addWeighted(CT, 0.8, heat_img, 0.2, 0)
    img[:, 960:1120] = seg_add
    
    for i in range(4):
        ab = pred[20+i, :, :, d]
        ab = cv2.resize(ab, (160,160), interpolation=cv2.INTER_NEAREST)
        ab = np.transpose(ab, [1,0])
        ab = cv2.flip(ab, 0)
        heat_img = cv2.applyColorMap(np.uint8(ab*seg*255), cv2.COLORMAP_JET)
        add = cv2.addWeighted(CT, 0.8, heat_img, 0.2, 0)
        img[:, 160+i*160:320+i*160] = add

def show_check_images(dir, organs, image_name, window_name):
    """
    dir是图片文件夹
    """
    print(dir)
    images = list(Path(dir).glob('*%s*.png'%image_name))
    img = cv2.imread(str(images[0]))
    img = img[-160:]
    paths = list(Path(dir).glob('prediction*.nii.gz'))
    pred = nib.load(str(paths[-1])).get_data()
    use_side_mask = True
    if use_side_mask:
        left_mask = np.zeros(pred[0].shape)
        left_mask[0:pred[0].shape[0]//2] = 1
        right_mask = np.zeros(pred[0].shape)
        right_mask[pred[0].shape[0]//2:] = 1
        print(right_mask.shape)
        for i in [0,2,4,8,10,15,18]:
            pred[i] = pred[i]*left_mask
        for i in [1,3,5,9,11,16,19]:
            pred[i] = pred[i]*right_mask
    #img是之前存储好的图片，里面的分割是所有脑区的分割。现在要根据x把对应脑区分割单独显示：
    change_seg(img, pred, organs, 0)
    cv2.imshow(window_name, img)
    now = 0
    while True:
        o = cv2.waitKey(0)
        if o==13: #enter
            cv2.destroyAllWindows()
            return -2;
        if o==27: #esc
            cv2.destroyAllWindows()
            return -1
        if o>=48 and o<=57:
            cv2.destroyAllWindows()
            return o-48
        if o==97:#a, 左
            now = max(now-1, 0)
        if o==113:#q,左5
            now = max(now-5, 0)
        if o==100:#d
            now = min(now+1, len(images)-1)
        if o==101:#e
            now = min(now+5, len(images)-1)
        img = cv2.imread(str(images[now]))
        img = img[-160:]
        change_seg(img, pred, organs, now*4//2) #根据图片显示间隔和模型输出分辨率计算
        cv2.imshow(window_name, img)
    pass

def check_cq500_class(dir, c):
    result = {}
    if Path(dir+'/manual_check_%d.json'%c).exists():
        with open(dir+'/manual_check_%d.json'%c, 'r') as fp:
            result = json.load(fp)

    if c<=1:
        organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19]
    elif c==2:
        organs = [0,2,4,8,10,15,17,18]
    elif c==3:
        organs = [1,3,5,9,11,16,17,19]
    c_desc = ['in', 'out', 'left', 'right']
    with open(dir+'/cq500_%d_max.txt'%c, 'r') as fp:
        cases = fp.readlines()
    for case in cases:
        gt = int(case.split(' ')[0])
        score = float(case.split(' ')[1])
        series = ' '.join(case.split(' ')[2:]).strip()
        if abs(gt-score)>0.5:
       # if gt==1:
            if series in result:
                print(result[series])
            o = show_check_images(dir+'/cq500_'+series, organs, 'resultabnormal', \
                                  c_desc[c]+' '+case)
            if o==-1:
                return
            if o==-2:
                continue
            """
            0：坏序列
            1: 假阳，小范围预测。例如将钙化、高密度血管影、脑边缘伪影预测高密度，或其他很小的高密度
            2：假阳，分割假阳
            3：假阳，就是预测假阳，如把低密度同时预测成高密度，或把脑内低同时预测成脑外低
            4：假阴，图像异常不明显
            5：假阴，异常在定义脑区之外
            6：假阴，就是没预测出来
            7：怀疑标注错误
            8：gt认为出血，但图像低密度；或低密度中有少量高密度，但gt不认为出血
            9：其他
            """
            result[series] = o
            with open(dir+'/manual_check_%d.json'%c, 'w') as fp:
                json.dump(result, fp)
            print(o)

def check_cq500(dir):
    for i in [1]:
        check_cq500_class(dir, i)

def analyse_check_result(file):
    with open(file, 'r') as fp:
        result = json.load(fp)
    cnt = np.zeros((10))
    for x in result:
        cnt[result[x]] += 1
    print(np.round(cnt/len(result), 3))

def check_false_case(dir, positive, ref_dir = None):
    """
    positive表示检查假阳性，否则检查假阴性
    ref_dir指参考。如果之前在某个模型的结果上做了检查，在同样数据集的新模型结果上
    做检查的时候，可以参考之前标注的结果，防止标注不一致。
    """
    with open(dir+'/scores_mem.txt', 'r') as fp:
        scores = fp.readlines()
    result = {}
    if positive:
        result_file = dir+'/prediction_check_result_pos.json'
    else:
        result_file = dir+'/prediction_check_result_neg.json'
    if Path(result_file).exists():
        with open(result_file, 'r') as fp:
            result = json.load(fp)
    if ref_dir is not None: #使用参考文件
        if positive:
            ref_file = ref_dir+'/prediction_check_result_pos.json'
        else:
            ref_file = ref_dir+'/prediction_check_result_neg.json'
        if Path(ref_file).exists():
            with open(ref_file, 'r') as fp:
                ref = json.load(fp)
    else:
        ref = {}
    print('ref loaded', len(ref))
    for case in scores:
        organ = case.split(' ')[0]
        gt = float(case.split(' ')[1])
        score = float(case.split(' ')[2])
        path = case.split(' ')[3].strip() #去掉行位回车
        if organ+'_'+path in result:
            continue
        if (positive and gt) or (not positive and not gt):
            continue
        if abs(gt-score)<0.5:
            continue
        with open(dir+'/'+path+'/'+'report.txt', 'r', encoding='utf-8') as fp:
            report = fp.read()
        print(report)
        if organ+'_'+path in ref:
            print('ref', ref[organ+'_'+path])
        else:
            print('not in ref')
        img = cv2.imread(dir+'/'+path+'/'+'resultabnormal_40.png')
        cv2.imshow(organ+' '+str(gt)+' '+str(score)+' '+path, img)
        now = 40
        while True:
            o = cv2.waitKey(0)
            if o==27:
                cv2.destroyAllWindows()
                return
            if o>=49 and o<=57:
                break
            if o==97:#a, 左
                now = max(now-1, 0)
            if o==113:#q,左5
                now = max(now-5, 0)
            if o==100:#d
                now = min(now+1, 79)
            if o==101:#e
                now = min(now+5, 79)
            
            img = cv2.imread(dir+'/'+path+'/'+'resultabnormal_%s.png'%str(now).zfill(2))
            cv2.imshow(organ+' '+str(gt)+' '+str(score)+' '+path, img)
        """
        1 可能由于头皮影响，无异常区域预测假阴性
        2 有高密度异常，脑区蹭到了
        3 伪影
        4 有低密度异常，可能认为高低混杂
        5 有高密度异常，但报告未描述
        6 报告描述有异常，但NLP错误
        7 序列错误，如短序列
        8 其他原因预测假阴性
        """
        cv2.destroyAllWindows()
        result[organ+'_'+path] = o-48
        with open(result_file, 'w') as fp:
            json.dump(result, fp)

def find_false_cases(dir, positive):
    with open(dir+'/scores_mem.txt', 'r') as fp:
        scores = fp.readlines()
    ret = set()
    for case in scores:
        if case.strip()=='':
            continue
        organ = case.split(' ')[0]
        gt = float(case.split(' ')[1])
        score = float(case.split(' ')[2])
        path = case.split(' ')[3].strip() #去掉行位回车
        if (positive and gt) or (not positive and not gt):
            continue
        if abs(gt-score)<0.5:
            continue
        ret.add(organ+'_'+path)
    return ret

def compare_false_case(dir1, dir2, positive):

    false1 = find_false_cases(dir1, positive)
    false2 = find_false_cases(dir2, positive)
    if positive:
        result_file = dir1+'/prediction_check_result_pos.json'
    else:
        result_file = dir1+'/prediction_check_result_neg.json'
    with open(result_file, 'r') as fp:
        check_result = json.load(fp)
    
    print(len(false1))
    print(len(false2))
    print(len(false1.difference(false2)))
    cnt = np.zeros((9))
    cnt_all = np.zeros((9))
    for x in false1.difference(false2):
        cnt[check_result[x]] += 1
    for x in check_result:
        cnt_all[check_result[x]] += 1
    print(cnt)
    print(cnt_all)
    print((cnt[1:]/cnt_all[1:]).round(2))
    print(len(false2.difference(false1)))

#check_false_case('Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multitask20', True)
#check_false_case('Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multitask20', True,
#                 'Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multitask18')

#compare_false_case('Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multitask18',
#                   'Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multiinput7', True)

'''
parts = ['左半卵圆中心', '右半卵圆中心', '左颞叶', '右颞叶', '左小脑', '右小脑', 
         '左岛叶', '右岛叶', '左额叶', '右额叶',  
         '左顶叶', '右顶叶', '脑桥', '左丘脑',   '右丘脑',
         '左基底节区', '右基底节区', '脑干', '左枕叶','右枕叶']
ID = {}
for i,x in enumerate(parts):
    ID[x] = i

with open('Z:\\project\\3DUnetCNN-master\\brats\\prediction\\valid_multitask18\\prediction_check_result_pos.json', 'r') as fp:
    result = json.load(fp)

#with open('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/prediction/valid_multitask18/prediction_check_result_pos.json', 'r') as fp:
#    result = json.load(fp)
output = []

cnt = np.zeros((10))
for x in result:
    cnt[result[x]] += 1
    if result[x]==5:
        print(x)
        with open('Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask18/'+x[3:]+'/report.txt', 'r', encoding='utf-8') as fp:
        #with open('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/prediction/valid_multitask18/'+x[3:]+'/report.txt', 'r', encoding='utf-8') as fp:
            report = fp.read()
            print(report)
        output.append((x,report,[]))
        """
        labels = np.zeros((20))
        while True:
            organ = input()
            print(organ)
            if organ[0]=='q':
                break
            if organ.strip() not in parts:
                continue
            
            labels[ID[organ.strip()]] = 1
        print(labels)
        """
print(cnt, len(result))
print((cnt/len(result)).round(2))
'''
#傻逼json输出tm到底是什么编码？？全tm是utf码转码也转不了，老子就想看个中文！！
#with open('samples_to_be_checked.json', 'w') as fp:
#    json.dump(output, fp, ensure_ascii=False)
'''
with open('samples_to_be_checked.json', 'r') as fp:
    result = json.load(fp)
with open('Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask18/scores.txt', 'r') as fp:
    scores = fp.readlines() #读这个文件只是想知道每个series在哪个文件夹（14,15...还是fold_1）
dirs = {}
for x in scores:
    if x.strip()=='':
        continue
    temp = x.strip().split(' ')[3]
    if len(temp.split('_'))==3:
        dirs[temp.replace(temp.split('_')[0]+'_', '')] = 'Z:/aligned_registration/%s/'%temp.split('_')[0]+temp.split('_')[-2]
    elif len(temp.split('_'))==4:
        dirs[temp.replace('fold_1_', '')] = 'Z:/registration/fold_1/'+temp.split('_')[-2]
        #print(temp.replace('fold_1_', ''))

for x in result:
    labels = np.zeros((20))
    for organ in x[2]:
        print(organ)
        if organ not in parts:
            print('no', organ)
            continue
        labels[ID[organ]] = 1
    print(dirs[x[0][3:]])
    all_paths = list(Path(dirs[x[0][3:]]).rglob('high_label.txt'))
    for path in all_paths:
        #np.savetxt(str(path.parent/'label_high_checked.txt'), labels)
        print('saved')
'''

#check_cq500('Z:/project/3DUnetCNN-master/brats/prediction/test_cq500_71')
#for i in range(1,4):
#    analyse_check_result('Z:/project/3DUnetCNN-master/brats/prediction/test_cq500_71/manual_check_%d.json'%i)

#check_cq500('Z:/project/3DUnetCNN-master/brats/prediction/test_cq500_74')
check_cq500('Z:/project/3DUnetCNN-master/brats/prediction/test_cq500_73')
