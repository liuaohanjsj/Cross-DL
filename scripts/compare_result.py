# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:41:36 2021

@author: Admin
"""

import cv2
import numpy as np
import json
import nibabel as nib
from pathlib import Path

def resize(img, shape, inter=cv2.INTER_NEAREST):
    """
    三维resize
    """
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=inter) #行列resize
    img = np.transpose(img, (0,2,1)) #160,5,160
    img = cv2.resize(img, (shape[2], shape[0]), interpolation=inter) #纵向resize
    img = np.transpose(img, (0,2,1))
    return img

def load_scores(file, scores):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    a = []
    for l in lines:
        if l=='':
            continue
        s = l.strip().split(' ')
        gt = float(s[2])
        pred = float(s[3])
        path = s[4]
        scores[s[1]+'_'+s[0]+'_'+path] = (gt,pred)
        a.append(pred)
    return np.mean(np.array(a)), np.std(np.array(a))
    


def compare(dir1, dir2, organs):
    def change_seg(img, pred, organ, i):
        CT = img[:,:160]

        seg = pred[organ, :, :, i//4]
        seg = cv2.resize(seg, (160,160), interpolation=cv2.INTER_NEAREST)
        seg = np.transpose(seg, [1,0])
        seg = cv2.flip(seg, 0)
        heat_img = cv2.applyColorMap(np.uint8(seg*255), cv2.COLORMAP_JET)
        seg_add = cv2.addWeighted(CT, 0.8, heat_img, 0.2, 0)
        img[:, 640:800] = seg_add
        
        high = pred[20, :, :, i//4]
        high = cv2.resize(high, (160,160), interpolation=cv2.INTER_NEAREST)
        high = np.transpose(high, [1,0])
        high = cv2.flip(high, 0)
        heat_img = cv2.applyColorMap(np.uint8(high*seg*255), cv2.COLORMAP_JET)
        add = cv2.addWeighted(CT, 0.8, heat_img, 0.2, 0)
        img[:, 160:320] = add
        
        low = pred[21, :, :, i//4]
        low = cv2.resize(low, (160,160), interpolation=cv2.INTER_NEAREST)
        low = np.transpose(low, [1,0])
        low = cv2.flip(low, 0)
        heat_img = cv2.applyColorMap(np.uint8(low*seg*255), cv2.COLORMAP_JET)
        add = cv2.addWeighted(CT, 0.8, heat_img, 0.2, 0)
        img[:, 320:480] = add
    
    def check_images(x, result):
        img1 = cv2.imread(dir1+'/'+x[5:]+'/resultabnormal_40.png')
        img2 = cv2.imread(dir2+'/'+x[5:]+'/resultabnormal_40.png')
        pred1 = nib.load(dir1+'/'+x[5:]+'/prediction.nii.gz').get_data()
        pred2 = nib.load(dir2+'/'+x[5:]+'/prediction.nii.gz').get_data()
        
        #img是之前存储好的图片，里面的分割是所有脑区的分割。现在要根据x把对应脑区分割单独显示：
        
        now = 40
        change_seg(img1, pred1, int(x[:2]), now)
        change_seg(img2, pred2, int(x[:2]), now)
        
        #name = old_labels[int(x[:2])]+'_'+('低' if int(x[3]) else '高')+x[4:]
        name = x
        img = np.concatenate((img1, img2), 0)
        cv2.imshow(name+' '+str(score1[x][0]), img)
        while True:
            o = cv2.waitKey(0)
            if o==27:
                cv2.destroyAllWindows()
                return False
            if o>=48 and o<=57:
                break
            if o==97:#a, 左
                now = max(now-1, 0)
            if o==113:#q,左5
                now = max(now-4, 0)
            if o==100:#d
                now = min(now+1, 79)
            if o==101:#e
                now = min(now+4, 79)
            
            img1 = cv2.imread(dir1+'/'+x[5:]+'/resultabnormal_%s.png'%str(now).zfill(2))
            img2 = cv2.imread(dir2+'/'+x[5:]+'/resultabnormal_%s.png'%str(now).zfill(2))
            change_seg(img1, pred1, int(x[:2]), now)
            change_seg(img2, pred2, int(x[:2]), now)
        
            img = np.concatenate((img1, img2), 0)
        
            cv2.imshow(name+' '+str(score1[x][0]), img)
        """
        0 分割导致
        1 异常检测导致
        2 其他（两种都有）
        3 略过（序列严重错误，无参考意义）
        4 一个模型分割受异常假阳性，另一个模型没有。（默认未分割假阳的那个模型效果好；若不是则归入5）
        5 第二个模型误打误撞弄对（比如本脑区异常，由于脑区分割错误，在其它脑区成功预测出来了）
        """
        cv2.destroyAllWindows()
        if o-48!=3:
            result[x] = o-48
        return True
    
    def load(dir, score):
        mean = np.zeros((20))
        std = np.zeros((20))
        for organ in organs:
            m,s = load_scores(dir+'/scores_%d.txt'%organ, score)
            mean[organ] = m
            std[organ] = s
        print(np.round(mean,2))
        print(np.round(std,2))
    
    old_labels = ['左半卵圆中心', '右半卵圆中心', '左颞叶', '右颞叶', '左小脑', '右小脑', 
                  '左岛叶', '右岛叶', '左额叶', '右额叶',  
                  '左顶叶', '右顶叶', '脑桥', '左丘脑',   '右丘脑',
                  '左基底节区', '右基底节区', '脑干', '左枕叶','右枕叶']
    score1 = {}
    score2 = {}
    result_good = {}
    result_bad = {}
    good_file = 'compare_result/result_good_%s_%s.json'%(dir1[-2:], dir2[-2:])
    bad_file = 'compare_result/result_bad_%s_%s.json'%(dir1[-2:], dir2[-2:])
    
    if Path(good_file).exists():
        with open(good_file, 'r') as fp:
            result_good = json.load(fp)
    if Path(bad_file).exists():
        with open(bad_file, 'r') as fp:
            result_bad = json.load(fp)
        
    load(dir1, score1)
    load(dir2, score2)

    '''
    for x in score1:
        if x in score2 and abs(score2[x][1]-score1[x][1])>0.5 and (score2[x][1]-score1[x][1])*(score1[x][0]-0.5)>0:
            print(x, score1[x][0], score1[x][1], score2[x][1])
            if not check_images(x, result_good):
                return
            with open(good_file, 'w') as fp:
                json.dump(result_good, fp)
    
    print()
    '''
    '''
    for x in score1:
        if x in score2 and abs(score2[x][1]-score1[x][1])>0.5 and (score2[x][1]-score1[x][1])*(score1[x][0]-0.5)<0:
            
            print(x, score1[x][0], score1[x][1], score2[x][1])
            if not check_images(x, result_bad):
                return
            with open(bad_file, 'w') as fp:
                json.dump(result_bad, fp)
    '''
    cnt = np.zeros((2+1,6+1), np.int32)
    for x in result_good:
        cnt[int(x[3]),result_good[x]] += 1
    cnt[:,3] = 0
    cnt[2] = np.sum(cnt, 0)
    cnt[:,6] = np.sum(cnt, 1)
    print(cnt)
    
    cnt = np.zeros((2+1,6+1), np.int32)
    for x in result_bad:
        cnt[int(x[3]),result_bad[x]] += 1
    cnt[:,3] = 0
    cnt[2] = np.sum(cnt, 0)
    cnt[:,6] = np.sum(cnt, 1)
    print(cnt)
    
    cnt = np.zeros((2+1,6+1), np.int32)
    for x in result_good:
        cnt[int(score1[x][0]),result_good[x]] += 1
    cnt[:,3] = 0
    cnt[2] = np.sum(cnt, 0)
    cnt[:,6] = np.sum(cnt, 1)
    print(cnt)
    
    cnt = np.zeros((2+1,6+1), np.int32)
    for x in result_bad:
        cnt[int(score1[x][0]),result_bad[x]] += 1
    cnt[:,3] = 0
    cnt[2] = np.sum(cnt, 0)
    cnt[:,6] = np.sum(cnt, 1)
    print(cnt)
    print()

'''
compare('Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask29', 
        'Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask36',
        [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19])

compare('Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask36', 
        'Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask35',
        [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19])
'''
compare('Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask36', 
        'Z:/project/3DUnetCNN-master/brats/prediction/valid_multitask39',
        [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19])