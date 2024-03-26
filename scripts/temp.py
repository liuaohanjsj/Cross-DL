# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:14:34 2020

@author: Admin
"""

import time
import shutil
import json
from pathlib import Path
import numpy as np
import cv2
import pydicom
import os
import nibabel as nib
#from nilearn.image import reorder_img, new_img_like
import tables
import pickle
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import csv
import pickle as pkl
import xlrd

'''
with open('D:\\data\\CTMR\\yuwei/train.json', 'r') as fp:
    index = json.load(fp)
print(type(index))
print(len(index))
for i in range(10):
    print(index[i])
'''
'''
with open('D:\\data\\CTMR\\301_study_reports_paths.json') as fp:
    paths = json.load(fp)
print(type(paths))
print(len(paths))
print(len(paths['EXAM_NO']))
for x in paths:
    print(x, type(paths[x]['0']), paths[x]['0'])
print()
for x in paths['EXAM_NO']:
    if paths['EXAM_NO'][x]==24740459:
        print('found', x)
    #if paths['EXAM_NO'][x]==paths['EXAM_NO'][str(int(x)+1)]:
    #    print('duplicate', x, paths['EXAM_NO'][x])

exam_no = set()
for x in paths['EXAM_NO']:
    exam_no.add(paths['EXAM_NO'][x])
print('exam no', len(exam_no))

with open('D:\\data\\CTMR\\301_series_path.json') as fp:
    paths = json.load(fp)
print(type(paths))
print(len(paths))
for x in paths:
    print(x, paths[x]['0'])
'''
'''
with open('D:\\data\\CTMR\\301\\df_studies_301_14.json') as fp:
    paths = json.load(fp)
for x in paths:
    print(x, type(x), paths[x]['0'])
'''
'''
all_paths = list(Path('shiyan').rglob('*result_mask*.png'))
for path in all_paths:
    img = cv2.imread(str(path))
    Id = path.stem[-2:]
    try:
        Id = int(Id)
    except:
        continue
    print(Id)
    cv2.imwrite(str(path).replace('.png', '_b_b.png'), np.uint8(np.greater(img, 1))*255)
    mask = np.uint8(np.greater(img, 10))*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.dilate(mask, kernel)
    output = np.concatenate((img, mask, np.uint8(np.greater(img, 1))*255), 1)
    cv2.imwrite(str(path).replace('.png', '_b.png'), output)
    

all_paths = list(Path('shiyan').rglob('*CT*.png'))
for path in all_paths:
    img = cv2.imread(str(path))
    Id = path.stem[-2:]
    try:
        Id = int(Id)
    except:
        continue
    print(Id)
    
    img = img[:,:,0]
    mask = np.uint8(np.greater(img, 0))*255
    mask[-10:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours是很多1*n*2的array组成的list，每个array是一个封闭轮廓
    #我不想对每个轮廓单独求凸包，而是所有点一起求凸包，所以把这些轮廓合并
    temp = []
    for c in contours:
        for x in c:
            temp.append(x)
    contours = [np.array(temp)]

    hull = []
    
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    mask = np.zeros(img.shape, np.uint8)

    for i in range(len(contours)):
        cv2.fillPoly(mask, hull, 255)
    output = np.concatenate((img, mask, np.minimum(img, mask)), 1)
    cv2.imwrite(str(path).replace('.png', '_brain.png'), output)
'''
'''
def nib2std(img):
    """
    将使用nib读取的图片转成对应的标准方位
    """
    if img.ndim==2:
        img = np.transpose(img, (1,0))
    else:
        img = np.transpose(img, (1,0,2))
    img = cv2.flip(img, 0)
    return img

img = nib.load('CT.nii.gz')
temp = img.get_data()
cv2.imwrite('shenme.png', temp[:,:,10])
cv2.imwrite('shenme2.png', nib2std(temp[:,:,10]))

img = reorder_img(img, resample = 'linear')
temp = img.get_data()
temp = np.transpose(temp, [1,0,2])
cv2.imwrite('shenme3.png', temp[:,:,10])
temp = cv2.flip(temp, 0)
cv2.imwrite('shenme4.png', temp[:,:,10])
'''
def top_k_mask(img):
    a = np.reshape(img, (-1)).tolist()
    a.sort(reverse = True)
    print(a)
    t = a[10]
    return np.uint8(np.greater(img, t))

import math

def distance_map(img):
    """
    img应该是二值图像
    """
    a = [(1,0),(-1,0),(0,1),(0,-1)]
    lin = []
    lout = []
    n,m = img.shape
    for i in range(n):
        for j in range(m):
            flag = 0
            for x,y in a:
                if i+x>=0 and i+x<n and j+y>=0 and j+y<m and img[i+x][j+y]!=img[i][j]:
                    flag = 1
            if flag:
                if img[i][j]:
                    lin.append((i,j))
                else:
                    lout.append((i,j))
            pass
    print(len(lin))
    print(len(lout))
    ret = np.zeros(img.shape)
    maxn = -10000
    for i in range(n):
        for j in range(m):
            minn = 100000
            if img[i][j]:
                for x,y in lout:
                    minn = min(minn, (i-x)**2+(j-y)**2)
                minn = math.sqrt(minn)-1
            else:
                for x,y in lin:
                    minn = min(minn, (i-x)**2+(j-y)**2)
                minn = -math.sqrt(minn)
            maxn = max(maxn, minn)
            ret[i,j] = minn
    print(maxn)
    return np.minimum(ret, 0)

def distance_map_3D(img):
    """
    img应该是三维矩阵
    """
    n,m,l = img.shape
    d1 = np.zeros(img.shape)
    for i in range(n):
        d1[i,:,:] = fast_distance_map(img[i,:,:])
    d2 = np.zeros(img.shape)
    for i in range(m):
        d2[:,i,:] = fast_distance_map(img[:,i,:])
    d3 = np.zeros(img.shape)
    for i in range(l):
        d3[:,:,i] = fast_distance_map(img[:,:,i])
    return (d1+d2+d3)/3

def fast_distance_map(img):
    """
    img应该是01矩阵
    """
    a1 = [(1,0),(-1,0),(0,1),(0,-1)]
    a2 = [(-1,-1),(-1,1),(1,-1),(1,1)]
    a3 = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    n,m = img.shape
    img = np.uint8(img)
    def single_distance_map(d, img):
        for i in range(0,n):
            for j in range(0,m):
                if img[i,j]:
                    for x,y in a1:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+1)
                        except:
                            pass
                    for x,y in a2:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+1.41)
                        except:
                            pass
                    for x,y in a3:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+2.24)
                        except:
                            pass
        
        for i in range(n-1,-1,-1):
            for j in range(m-1,-1,-1):
                if img[i,j]:
                    for x,y in a1:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+1)
                        except:
                            pass
                    for x,y in a2:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+1.41)
                        except:
                            pass
                    for x,y in a3:
                        try:
                            d[i,j] = min(d[i,j], d[i+x,j+y]+2.24)
                        except:
                            pass
        return d
    din = np.float32(img)*1e5
    #din = single_distance_map(din, img)
    din = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dout = np.float32(np.max(img)-img)*1e5
    #dout = single_distance_map(dout, np.max(img)-img)
    dout = cv2.distanceTransform(1-img, cv2.DIST_L2, 3)
    return din-dout

def exp_s(x, s):
    return np.exp(x/s)
def p_map(img):
    print(img.shape)
    n = img.shape[0]
    d = np.zeros(img.shape)
    for i in range(n):
        #d[i] = distance_map(img[i])
        d[i] = fast_distance_map(img[i])
        print(np.max(d[i]), np.min(d[i]))
        cv2.imwrite('dis%d.png'%i, normalize(d[i]))
        d[i] = sigmoid(d[i], 1)
        cv2.imwrite('p%d.png'%i, normalize(d[i]))
    #d = exp_s(d, 10)
    #d = sigmoid(d, 10)
    s = np.sum(d, 0)
    return d/s

def sigmoid(x, scale = 10):
    return 1/(1+np.exp(-x/scale))

def normalize(d):
    return (d-np.min(d))*255/(np.max(d)-np.min(d))
"""
img = cv2.imread('square.png')
img = img[:,:,0]
d = fast_distance_map(img)
cv2.imwrite('temp.png', d*5)
"""
'''
a = np.asarray([1,1])
b = np.asarray([0,1])
print(a/b)
img = np.uint8(np.zeros((5,5)))
d = fast_distance_map(img)
print(d)
print(d+d+d)
print((d+d+d)/3)

img = cv2.imread('refine_mask_15.png')
img = img[:,:,0]
for i in range(256):
    if i in img:
        print(i)
#img = img[:,:,0]
#img2 = cv2.imread('seg2.png')
#img2 = img2[:,:,0]
img = cv2.resize(img, (128,128), cv2.INTER_NEAREST)
images = []
kernel = np.ones((5,5),np.uint8)
for i in [30,40,90,100,190,200,0]:
    images.append(cv2.erode(np.uint8(np.equal(img, i)), kernel))
p = p_map(np.stack(images))
temp = normalize(p)
for i in range(6):
    cv2.imwrite('p_seg%d.png'%i, temp[i])
'''
'''
d = distance_map(img)
cv2.imwrite('distance.png', (d-np.min(d))*255/(np.max(d)-np.min(d)))

d = sigmoid(d)
cv2.imwrite('p.png', (d-np.min(d))*255/(np.max(d)-np.min(d)))
'''
'''
def max_component(img):
    def get(x, fa):
        if fa[x]==x:
            return x
        fa[x] = get(fa[x], fa)
        return fa[x]
    mask = np.uint8(np.greater(img, 0.2))
    print(img.shape)
    n = img.shape[0]
    tot = 1
    mem = [None,]
    area = np.zeros(255) #从1开始使用
    size = np.zeros(255)
    fa = np.arange(255)#并查集
    image = np.zeros(img.shape, np.uint8) #在这上面记录不同连通区域
    for i in range(n):
        contours,hierarchy = cv2.findContours(mask[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area[tot] = cv2.contourArea(c)
            if area[tot]==0:
                print(c)
            cv2.fillPoly(image[i], [c], tot)
            mem.append((i,c))
            tot += 1
        
        if i>0:
            for x in range(img.shape[1]):
                for y in range(img.shape[2]):
                    if image[i,x,y] and image[i-1,x,y]:
                        fa[get(image[i,x,y],fa)] = get(image[i-1,x,y], fa)
    for i in range(1,tot):
        size[get(i,fa)] += area[i]
        print(i, fa[i])
    print(size[np.nonzero(size)])
    max_id = np.argmax(size)
    print(max_id)
    for i in range(1,tot):
        if get(i,fa)!=max_id:
            print('change', mem[i][0], mem[i][1])
            before = np.sum(img[mem[i][0]])
            cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0)
            print(before, np.sum(img[mem[i][0]]))
    return
'''
def get_max_component(img):
    """
    img是三维array。去除img中除最大连通区域以外的区域
    """
    def get(x, fa):
        if fa[x]==x:
            return x
        fa[x] = get(fa[x], fa)
        return fa[x]
    mask = np.uint8(np.greater(img, 0.2))
    print(type(img), img.dtype, img.shape)
    #print(img.shape)
    n = img.shape[0]
    tot = 1
    mem = [None,]
    area = np.zeros(255) #从1开始使用
    size = np.zeros(255)
    fa = np.arange(255)#并查集
    image = np.zeros(img.shape, np.uint8) #在这上面记录不同连通区域
    for i in range(n):
        contours,hierarchy = cv2.findContours(mask[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area[tot] = cv2.contourArea(c)
            cv2.fillPoly(image[i], [c], tot)
            mem.append((i,c))
            tot += 1
        
        if i>0:
            for x in range(img.shape[1]):
                for y in range(img.shape[2]):
                    if image[i,x,y] and image[i-1,x,y]:
                        fa[get(image[i,x,y],fa)] = get(image[i-1,x,y], fa)
    for i in range(1,tot):
        size[get(i,fa)] += area[i]
    print(size[np.nonzero(size)])
    max_id = np.argmax(size)
    #print(max_id)
    for i in range(1,tot):
        if get(i,fa)!=max_id:
            print('change', mem[i][0], area[i], mem[i][1])
            before = np.sum(img[mem[i][0]])
            print(cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0))
            print(type(img[mem[i][0]]),
                  type(cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0)))
            print(img[mem[i][0]].dtype,
                  cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0).dtype)
            img[mem[i][0]] = cv2.fillPoly(img[mem[i][0]], mem[i][1], 0)
            print(np.sum(img[mem[i][0]]), before)
    return

'''
img = cv2.imread('seg.png')
img = img[:,:,0]
img2 = cv2.imread('seg2.png')
img = np.stack((img, img2[:,:,0], img*0), 0)
print(img.shape)
img = np.float32(np.greater(img, 1))*0.8
img = img[np.newaxis]
print(img.shape)
temp = img.copy()
get_max_component(img[0])
print(np.sum(temp-img))
'''

'''
contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("number of contours:%d" % len(contours))
#cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
#找到最大区域并填充
area = []
for i in range(len(contours)):
    area.append(cv2.contourArea(contours[i]))
print(area)
max_idx = np.argmax(area)
print(max_idx)

cv2.fillPoly(img, contours, 10)
cv2.imwrite('contours.png', img)
'''
def view_log():
    version = '80'
    with open('log/training_%s.log'%version, 'r') as fp:
        s = fp.readlines()
    patience = 2
    n_epoch_decay = None#16
    
    seg_weight_decay = 1/0.81
    tot = 0
    plt.clf()
    t = 0
    initial_lr = 5e-4
    loss_decay = 0.5
    delta = 1e-4#1e-4
    seg_losses = []
    train_losses = []
    test_losses = []
    minn = float('inf')
    
    for x in s[1:]:
        if int(x.split(',')[0])==0:
            if tot>0:
                print(tot, seg_losses)
                plt.plot(range(len(seg_losses)),seg_losses)
                plt.plot(range(len(seg_losses)),train_losses)
                plt.plot(range(len(seg_losses)),test_losses)
                plt.savefig('log/loss_%s_%d.png'%(version,tot))
            tot += 1
            lrs = []
            minn = float('inf')
            t = 0
            lr = initial_lr
            seg_losses = []
            train_losses = []
            test_losses = []
            plt.clf()
        
        n = int(x.split(',')[0])
        train_loss = float(x.split(',')[2])
        seg_loss = float(x.split(',')[3])*min(seg_weight_decay**n, 100)
        seg_losses.append(seg_loss)
        train_losses.append(train_loss)
        loss = float(x.split(',')[5])
        test_losses.append(loss)
        print(n, round(seg_loss, 4), round(train_loss,4), round(loss,4), lr)
        
        lrs.append(lr)
        t += 1
        if loss<minn-delta:
            minn = loss
            t = 0
        if (n_epoch_decay is None and t>patience) or (n_epoch_decay and (n+2)%n_epoch_decay==0):
            lr *= loss_decay
            t = 1
            #plt.axvline(n)
    plt.plot(range(len(seg_losses)), seg_losses, label='Train segmentation')
    plt.plot(range(len(seg_losses)), train_losses, label='Train abnormal MIL')
    plt.plot(range(len(seg_losses)), test_losses, label='Test abnormal MIL')
    plt.grid(axis='y', zorder=0)
    plt.legend(loc='upper right')
    plt.savefig('log/loss_%s_%d.png'%(version,tot))
    
#view_log()

'''
file_opened = tables.open_file("/data/liuaohan/co-training/more_brain_data_160_80.h5", "r")
subject_ids = file_opened.root.subject_ids
subject_ids = [x.decode('utf-8') for x in subject_ids]
file_opened.close()
print(len(subject_ids))
print(subject_ids[:2])
while True:
    s = input()
    if s=='quit':
        break
    for x in subject_ids:
        if s in x:
            print(x)
'''
'''
np.set_printoptions(suppress=True)
ratio = np.loadtxt('positive_ratio_1.1.txt')
print(np.round(ratio, 4))
n,m = ratio.shape
organ_ratio = np.sum(ratio, 1, keepdims = True)
abnormality_ratio = np.sum(ratio, 0, keepdims = True)
print(organ_ratio.shape)
print(abnormality_ratio)

result = np.dot(organ_ratio, abnormality_ratio)/np.sum(ratio)
print(np.round(result, 4))

print(np.sum(result, 0))
print(np.sum(ratio, 0))
print(np.sum(result, 1))
print(np.sum(ratio, 1))
np.savetxt('positive_ratio_1.1_regular.txt', result)
'''
'''
def standard_path(path):
    """
    path是一个序列路径，现在要知道路径的标准路径，只要最后三层的路径，
    而且如果path是在本机和服务器运行的，斜杠的用法也不同，现在转换为相同的。
    path为str，返回str
    """
    reportID = Path(path).parent.parent.name
    studyID = Path(path).parent.name
    seriesID = Path(path).name
    print(reportID, studyID, seriesID)
    return reportID+'/'+studyID+'/'+seriesID

bad = {}
with open('/home/liuaohan/pm-stroke/aligned_registration/01'+'/registration_check_result.json', 'r') as fp:
    registration_check = json.load(fp)
for x in registration_check:
    if registration_check[x]>=3:
        bad[standard_path(x)] = True
        #print('add bad', standard_path(x))
subject_dir = '/home/liuaohan/aligned_registration/01/25852832.0/1.2.840.113820.861002.42693.589793831187.2.1150034/1.2.156.112605.14038005400544.20161119063926.3.6040.5'
if Path(subject_dir+'/series_link.txt').exists():
    with open(subject_dir+'/series_link.txt', 'r') as fp:
        image_dir = fp.read().strip() #image_dir用于judge及bad判断此序列是否可用。去除文末换行
else:
    image_dir = subject_dir
print(standard_path(image_dir))
if standard_path(image_dir) in bad:
    print('bad series', image_dir)

s = 'Z:/123\\456'
print(Path(s), Path(s).parent.name)
s = 'Z:/123/456'
print(Path(s), Path(s).parent.name)
s = 'Z:/123\\456/789'
print(Path(s), Path(s).parent.name)
s = 'Z:\\123\\456/789'
print(Path(s), Path(s).parent.name)
'''
'''
file_opened = tables.open_file("all_refined_label_new1.h5", "r")
labels = file_opened.root.label
print(len(labels))
cnt = np.zeros((2,2))
cntd = np.zeros((2,2))
for label in labels:
    cnt[int(label[15,0]), int(label[16,0])] += 1
print(cnt/len(labels))

with open('/data/liuaohan/co-training/38713_test_1000p.pkl', 'rb') as fp:
    test = pickle.load(fp)
cnt = np.zeros((2,2))
for x in test:
    cnt[int(labels[x][15,0]), int(labels[x][16,0])] += 1
print(cnt/len(test))

with open('/data/liuaohan/co-training/38713_valid_1000p.pkl', 'rb') as fp:
    valid = pickle.load(fp)
cnt = np.zeros((2,2))
for x in valid:
    cnt[int(labels[x][15,0]), int(labels[x][16,0])] += 1
print(cnt/len(valid))

with open('/data/liuaohan/co-training/38713_training_17877p.pkl', 'rb') as fp:
    training = pickle.load(fp)
cnt = np.zeros((2,2))
for x in training:
    cnt[int(labels[x][15,0]), int(labels[x][16,0])] += 1
print(cnt/len(training))
'''
def check_training_labeled():
    """
    2023.3.5 为了测试在训练集使用一小部分手标label，训练模型效果如何，手标了一些label
    这些label应该是训练集中前面大概1000例CT的标注（因为数据index的pkl文件里是随机打乱的所以分布均匀）。
    现在检查一下，是不是真的是这样的
    """
    data = tables.open_file("/data/liuaohan/co-training/all_brain_data_160_80.h5", "r")
    indexes = pickle.load(open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl", 'rb'))
    #with open('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/report_desc.json', 'r') as fp: #brats里的report_desc和这个report_desc还不一样！有的分句没有nlp_2里的好。不知道brats里的report_desc和report_desc2报告和分句是否一致
    with open('report_desc2.json', 'r') as fp:
        #brats里的report_desc2和text_classification_yingqi/nlp_2/report_desc是报告分句是一致的。brats里的report_desc报告分句不一样。
        report_desc = json.load(fp)
    #载入手标label：
    sentence_label = {}
    n_organ = 19
    with open('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/label_result_training_.csv', 'r', encoding='gbk') as fp:
        reader = csv.reader(fp, delimiter='|')
        for row in tqdm(reader): #要在fp还没释放的时候对reader进行操作
            print(row[2], row[2][n_organ+13])
            if (row[2][n_organ+13]=='1' or row[2][n_organ+14]=='1'): #存疑，待拆分，不使用手工标注。不明区域暂时使用手工标注，在CT模型predict计算指标的时候筛除
                #bad += 1
                continue
            #手标label长为34或34的倍数，前31个和NLP生成的对应，后面多了无异常、存疑、待拆分
            sentence_label[row[1]] = [int(x) for x in row[2]] #nlp预测为true false，手标的是1 0
    print('labeled', len(sentence_label))
    for tot, i in enumerate(indexes):
        print(i)
        path = data.root.subject_ids[i].decode('utf-8')
        report_id = Path(path).parent.parent.name
        if Path(path+'/cor_report.json').exists():
            with open(path+'/cor_report.json', 'r') as fp:
                report_id = json.load(fp)[0]
        desc = [s.strip() for s in report_desc[report_id][0]]
        flag = True
        for s in desc:
            s = s.strip()
            if s not in sentence_label:
                print('not labeled', s)
                flag = False
        if not flag:
            break
    print(tot) #标了training集中前1170个，不错
    
def analyse_report(data_file, index_file, use_index_file = None, naoshi=False, soft = False, use_hand_labeled = True,
                   suffix='', label_file_name = 'label_temp',doctor_label = False, save_ratio = False, label_file=None):
    def filt(desc, sentence_pred):
        for s in desc:
            pred = sentence_pred[s]
            if pred[19] and sum(pred[17:])==1 and sum(pred[:17])>=5:
                return True
        return False
    def change_doctor_label(label):
        ret = ''
        i = 0
        while i*30<len(label): #可能有多个标注
            temp = label[i:i+30]
            ret += temp[:18]+'0'+temp[18:22]+'0'+temp[22:26]+'0'+temp[26:30]+'0'
            i += 30
        return ret
    labels = ['左额叶','左顶叶','左颞叶','左枕叶','左基底节区','左半卵圆中心','左小脑',
     '右额叶','右顶叶','右颞叶','右枕叶','右基底节区','右半卵圆中心','右小脑',
     '脑干', '其他区域', '不明区域', 
     '脑内高密度','脑内等密度','脑内低密度','脑内液体','脑内气体',
     '脑外高密度','脑外等密度','脑外低密度','脑外液体','脑外气体',
     '脑沟高','其他异常']
    if naoshi:
        labels.insert(4,'左侧脑室')
        labels.insert(12,'右侧脑室')
        print(labels)
    old_labels = ['左半卵圆中心', '右半卵圆中心', '左颞叶', '右颞叶', '左小脑', '右小脑', 
                  '左岛叶', '右岛叶', '左额叶', '右额叶',  
                  '左顶叶', '右顶叶', '脑桥', '左侧脑室', '右侧脑室',
                  '左基底节区', '右基底节区', '脑干', '左枕叶','右枕叶']
    n_organ = 19 if naoshi else 17
    ID = {}
    for i,x in enumerate(old_labels):
        ID[x] = i
    #5.11.2021将其他区域、不明区域替换左岛叶、右岛叶的位置
    ID['其他区域'] = 6
    ID['不明区域'] = 7
    
    data = tables.open_file(data_file, "r")
    print(len(data.root.subject_ids))
    if suffix=='_prospective':
        with open('/home/liuaohan/pm-stroke/project/EHR/report_desc_p.json','r') as fp:
            report_desc = json.load(fp)
    elif suffix=='_prospective2' or suffix=='_prospective3': #这个是对的
        with open('/home/liuaohan/pm-stroke/project/EHR/report_desc_p2.json','r') as fp:
            report_desc = json.load(fp)
    else:
        with open('report_desc2.json','r') as fp:
            report_desc = json.load(fp)
    nlp_suffix = suffix if suffix!='_prospective3' else '_prospective2'
    with open('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/sentence_pred%s%s%s.json'%\
              ('' if not naoshi else '_bert', nlp_suffix, '' if not soft else '_soft'), 'r') as fp:
        sentence_pred = json.load(fp)
    print(len(sentence_pred))
    sentence_pred = {x[0].strip():x[1] for x in sentence_pred} #NLP生成label长为31
    '''
    for s in sentence_pred:
        minn = min(sentence_pred[s])
        if minn<1e-10:
            print('minn', minn)
    '''
    print(len(sentence_pred))
    bad = 0
    if label_file is None:
        if not naoshi:
            label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/label_result_38713.csv'
        elif suffix=='_prospective2': #前瞻集
            label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/result_arranged_prospective.csv'
        elif suffix=='_prospective3': #前瞻集    
            label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/result_arranged_prospective_lah.csv'
            
        else: #回顾集
            if not doctor_label: #自己手标的
                label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/label_result_38713_arranged-naoshi.csv'
            else: #医生手标的
                label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/result_arranged_doctor.csv'
    thresholds = np.loadtxt('/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/threshold_bert.txt') #本来是不用的，但是对于soft预测情况，发现还是需要用一下，很尴尬
    
    if use_hand_labeled:
        with open(label_file, 'r', encoding='gbk') as fp:
            reader = csv.reader(fp, delimiter='|')
            for row in tqdm(reader): #要在fp还没释放的时候对reader进行操作
                if doctor_label: #更新后医生用的标注格式，没有不明区域、待拆分等label了，转换回去
                    row[2] = change_doctor_label(row[2])
                if not doctor_label and (row[2][n_organ+13]=='1' or row[2][n_organ+14]=='1'): #存疑，待拆分，不使用手工标注。不明区域暂时使用手工标注，在CT模型predict计算指标的时候筛除
                    bad += 1
                    continue
                #手标label长为34或34的倍数，前31个和NLP生成的对应，后面多了无异常、存疑、待拆分
                sentence_pred[row[1]] = [int(x) for x in row[2]] #nlp预测为true false，手标的是1 0
    print(len(sentence_pred))
    cnt_nlp = 0
    for x in sentence_pred:
        if len(sentence_pred[x])==31: #用label长度判断即可
            cnt_nlp += 1
    print('sentence nlp, hand:', cnt_nlp, len(sentence_pred)-cnt_nlp)
    print('bad', bad)
    #print('右侧额部前方可见气体影，基底节区可见散在片状最大截面约5.3x3.6cm高密度出血影',sentence_pred['右侧额部前方可见气体影，基底节区可见散在片状最大截面约5.3x3.6cm高密度出血影'])
    cnt_filt, cnt = 0,0
    
    result = []
    label_list = [np.zeros((20,4)) for x in data.root.subject_ids]
    cnt_hand, cnt_nlp = 0,0
    sum_label = np.zeros([20, 4])
    sum_type = np.zeros([4])
    sum_organ = np.zeros([20])
    sum_whole = np.zeros([1])
    
    use_index = range(len(data.root.subject_ids))
    if use_index_file:
        with open(use_index_file, 'rb') as fp:
            use_index = pickle.load(fp)
    for j,path in enumerate(tqdm(data.root.subject_ids[:])):
        if j not in use_index:
            continue
        path = path.decode('utf-8')
        
        report_id = Path(path).parent.parent.name
        if Path(path+'/cor_report.json').exists():
            with open(path+'/cor_report.json', 'r') as fp:
                report_id = json.load(fp)[0]
        desc = [s.strip() for s in report_desc[report_id][0]]
        diag = report_desc[report_id][1]
        
        if Path(path).parent.parent.name=='385741':
            print(j, report_id, desc)
        if Path(path).parent.parent.name=='B097939':
            print(j, report_id, desc, diag)
        if Path(path).parent.parent.name=='C869270':
            print(j, report_id, desc, diag)
        if Path(path).parent.parent.name=='K0625490':
            print('K0625490', j, report_id, desc, diag)
        if Path(path).parent.parent.name=='Y4480447':
            print('Y4480447', j, report_id, desc, diag)
        if Path(path).parent.parent.name=='Y4886296':
            print('Y4886296', j, report_id, desc, diag)
            
        #with open(path+'/sentence_pred.json', 'r') as fp:
        #    temp = json.load(fp)
        #sentence_pred = [(x, y) for x,y in zip(temp[0], temp[1])]
        #for x in sentence_pred:
        #    if x[1][19] and sum(x[1][17:])==1: #只有低密度异常
        #        ischemia.append(x[0])
        
        #if filt(desc, sentence_pred):
        #    continue

        flag = False
        if '缺血' in diag: 
            cnt += 1
            #5.30.2022加入and suffix!='_prospective2'。之前前瞻1500个里筛掉了25个这样的。但是只是continue了，即label为0，但测试依然使用了。现在删除试试
            if ('脑梗' in diag or '软化' in diag or '低密度' in diag) and suffix!='_prospective2' and suffix!='_prospective3':
                cnt_filt += 1
                continue
            flag = True
        result.append(j)
            
        label = np.zeros((20,4))
        hand_labeled = True
        for s in desc:
            if s=='右侧额部前方可见气体影，基底节区可见散在片状最大截面约5.3x3.6cm高密度出血影':
                print(j, report_id, path)
                print(sentence_pred[s])
            if Path(path).parent.parent.name=='A109891':
                print(s, sentence_pred[s])
            prediction = sentence_pred[s]
            #if pred[0] is True or pred[0] is False: #用is可以区分True和1。==会认为True==1。但是这个判断对于soft 预测无效
            #    hand_labeled = False #只要有一句话是nlp预测的（包括是手标的但是存疑或待拆分），就认为整个序列是nlp预测的
            #elif pred[0]>0 and pred[0]<1: #这个是针对soft情况的，简单地认为网络预测的值不会是精确的0或1。
            #    hand_labeled = False
            if len(prediction)==31: #只有nlp生成的长才为31
                hand_labeled = False
                preds = [prediction]
            else: #手标的，长为34的倍数，可能有多个标注
                preds = []
                assert len(prediction)%34==0
                while len(prediction):
                    if (prediction[n_organ+7] or prediction[n_organ+8]) and (prediction[4] or prediction[5]):
                        print('脑室低密度', s)
                        print(diag, path)
                    preds.append(prediction[:31]) #只用31就可以，34 也可以
                    prediction = prediction[34:]
                    
            for pred in preds:
                binary_pred = [pred[i]>thresholds[i] for i in range(31)] #正常情况pred就已经是binary的了，这里是为了针对soft pred的情况
                if flag and binary_pred[n_organ+2] and sum(binary_pred[n_organ:])==1: #诊断缺血灶，而且此句预测只有低密度，不认为阳性
                    #if random.random()<0.01:
                    #    print(s, diag.replace('\n', ' '))
                    #continue
                    pred[n_organ+2] = 0 #之前遇到这种情况直接忽略这句话中的异常，continue。现在对于soft情况，将低密度弄成0，其他密度预测还是保留一点点值（应该很小），为了auc曲线做出来不那么理想
                for i in range(n_organ):
                    o = ID[labels[i]]
                    if not soft:
                        if pred[i]:
                            if pred[n_organ+0]: #脑内高
                                label[o,0] = 1
                            if pred[n_organ+5] or pred[n_organ+10]: #脑外高
                                label[o,1] = 1
                            if pred[n_organ+2] or pred[n_organ+3]: #脑内低
                                label[o,2] = 1
                            if pred[n_organ+7] or pred[n_organ+8]: #脑外低
                                label[o,3] = 1
                    else:
                        label[o,0] = max(label[o][0], pred[i]*pred[19+0])
                        label[o,1] = max(label[o][1], pred[i]*max([pred[19+5], pred[19+10]]))
                        label[o,2] = max(label[o][2], pred[i]*max([pred[19+2], pred[19+3]]))
                        label[o,3] = max(label[o][3], pred[i]*max([pred[19+7], pred[19+8]]))
        #continue
        sum_label += label
        sum_type += np.max(label, 0)
        sum_organ += np.max(label, 1)
        sum_whole += np.max(label)
        
        if not hand_labeled:
            cnt_nlp += 1
            #np.savetxt(path+'/refined_label_ischemia_false.txt', label)
            #if naoshi:
            #    np.savetxt(path+'/refined_label_ischemia_false_naoshi%s.txt'%('' if not soft else '_soft'), label)
            label[12, 3] = 1 #为了输出到h5文件里能区分是不是nlp预测的
        else:
            cnt_hand += 1
            #print(desc, [sentence_pred[s] for s in desc])
            #np.savetxt(path+'/refined_label_ischemia_false_checked.txt', label)
            #if naoshi:
            #    np.savetxt(path+'/refined_label_ischemia_false_naoshi%s_checked.txt'%('' if not soft else '_soft'), label)
        label_list[j] = label
    print('nlp, hand:', cnt_nlp, cnt_hand)
    print('result', len(result)) #34204
    return

    if save_ratio:
        np.savetxt('paper/%s_positive_ratio.txt'%save_ratio, sum_label/len(result))
        np.savetxt('paper/%s_positive_ratio_organ.txt'%save_ratio, sum_organ/len(result))
        np.savetxt('paper/%s_positive_ratio_type.txt'%save_ratio, sum_type/len(result))
        np.savetxt('paper/%s_positive_ratio_whole.txt'%save_ratio, sum_whole/len(result))
        
    #一般是先在本函数中把label写到原始数据文件夹中，然后在train_new里make_training_set再读取再生成h5，太慢了
    #2021.12.15生成soft预测的时候就直接写到h5文件里了，可以少写一遍文件，少读一遍文件，非常快。而且这里存的类型不再是UInt8Atom而是Float32Atom
    hdf5_file = tables.open_file('%s.h5'%label_file_name, mode='w')
    datatype = tables.Float32Atom() if soft else tables.UInt8Atom()
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', datatype, shape=([0,20,4]),
                                           expectedrows=len(data.root.subject_ids))
    for label in label_list:
        label_storage.append(label[np.newaxis])
    hdf5_file.close()
    
    with open(index_file, "wb") as fp:
        pickle.dump(result, fp)
    print(len(result))
    print(len(data.root.subject_ids), cnt_filt)
    print(cnt)
    #print(ischemia[:100])

def load_cq500_label(file):
    def vote(row, col):
        cnt = 0
        for i in range(3):
            cnt += sh.cell_value(row, col+i*14)
        return 1 if cnt>=2 else 0
    
    data = xlrd.open_workbook(file)
    sh = data.sheet_by_index(0)
    n = sh.nrows
    ret = []
    for i in range(1,n):
        label = []
        label.append(vote(i,3)) #脑内高
        label.append(1 if vote(i, 5) or vote(i,6) or vote(i, 7) else 0) #脑外高
        label.append(vote(i, 8)) #左高
        label.append(vote(i, 9)) #右高
        ret.append((sh.cell_value(i, 0)[9:], label))
    return ret
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered.pkl') #去除所有一句话中脑内低大于等于5个部位的序列
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered2.pkl') #将缺血灶认为是脑内低的负样本。去除无法明确知道低密度是缺血还是软化等其他低密度异常的序列
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered3.pkl') #和2一样，区别在于有手工标注的都使用了手工标注
'''
with open("/data/liuaohan/co-training/cq500_training.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/cq500_test.pkl", 'wb') as fp:
    pkl.dump(list(range(462)), fp)
'''
'''
with open("/data/liuaohan/co-training/cq500_training_full.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/cq500_test_full.pkl", 'wb') as fp:
    pkl.dump(list(range(489)), fp)
'''
#label = load_cq500_label('/home/liuaohan/pm-stroke/public/cq500/label.xlsx')
#print(len(label), label[0])
def split_5_fold():
    n = 462
    fold = 5
    ori = list(range(n))
    random.seed(10)
    random.shuffle(ori)
    n_valid = (n+fold-1)//fold #取上整
    for i in range(fold):
        valid = ori[i*n_valid : min((i+1)*n_valid, n)]
        train = list(set(ori)-set(valid))
        print(len(train), len(valid))
        with open("cq500_training_f%d.pkl"%i, 'wb') as fp:
            pkl.dump(train, fp)
        with open("cq500_validation_f%d.pkl"%i, 'wb') as fp:
            pkl.dump(valid, fp)

def split_10_fold(seed = 10, n=462, fold = 10, dataset='cq500'):
    fold = 10
    ori = list(range(n))
    random.seed(seed)
    random.shuffle(ori)
    n_valid = (n+fold-1)//fold #取上整
    name = '' if seed==10 else '_%d'%seed
    for i in range(fold):
        test = ori[i*n_valid : min((i+1)*n_valid, n)]
        dev = list(set(ori)-set(test))
        valid = random.sample(dev, n//fold)
        train = list(set(dev)-set(valid))
        
        print(len(train), len(test))
        with open("%s_training_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(train, fp)
        with open("%s_validation_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(valid, fp)
        with open("%s_test_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(test, fp)

def split_rsna_fold(seed = 10, fold = 10, dataset='rsna', patient_file = '/home/liuaohan/pm-stroke/public/rsna/patients.json',
                    data_file = "/data/liuaohan/co-training/rsna_data80_0.h5"):
    with open(patient_file, 'r') as fp:
        patients = json.load(fp)
    data_file = tables.open_file(data_file, "r")
    ID_in_dataset = {}
    for i in range(len(data_file.root.subject_ids)):
        ID = data_file.root.subject_ids[i].decode('utf-8')
        ID = Path(ID).name.split('_')
        ID = ID[1]+'_'+str(int(ID[2]))
        ID_in_dataset[ID] = i
    print(len(data_file.root.subject_ids), len(ID_in_dataset))
    data_file.close()
    
    patients = [patients[x] for x in patients]
    n = len(patients)
    print('n', n)
    ori = list(range(n))
    random.seed(seed)
    random.shuffle(ori)
    n_valid = (n+fold-1)//fold #取上整
    name = '' if seed==10 else '_%d'%seed
    for i in range(fold):
        test = ori[i*n_valid : min((i+1)*n_valid, n)]
        dev = list(set(ori)-set(test))
        valid = random.sample(dev, n//fold)
        train = list(set(dev)-set(valid))
        #test是病人在patientis中的下标，要找到这个病人所有series在h5文件的下标，也就是test_IDs
        test_IDs = []
        valid_IDs = []
        train_IDs = []
        for j in test:
            for series in patients[j]:
                if series in ID_in_dataset: #有些series虽然创建了文件夹，但是由于配准等原因，没有加入实际数据集
                    test_IDs.append(ID_in_dataset[series])
                    break #每个病人用一个序列就行了。因为发现大多数不是一个病人有多个序列，而是一个序列重复了几次
        for j in valid:
            for series in patients[j]:
                if series in ID_in_dataset:
                    valid_IDs.append(ID_in_dataset[series])
                    break
        for j in train:
            for series in patients[j]:
                if series in ID_in_dataset:
                    train_IDs.append(ID_in_dataset[series])
                    break
        print(len(train), len(valid), len(test), len(train_IDs), len(valid_IDs), len(test_IDs))
        
        with open("%s_training_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(train_IDs, fp)
        with open("%s_validation_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(valid_IDs, fp)
        with open("%s_test_f10%s_%d.pkl"%(dataset,name,i), 'wb') as fp:
            pkl.dump(test_IDs, fp)
        print(test_IDs)
#split_10_fold()
#split_10_fold(0)
#split_10_fold(n=975, dataset='rsna')
#split_rsna_fold()
#split_rsna_fold(fold=5, dataset='rsna4', patient_file = '/home/liuaohan/pm-stroke/public/rsna/patients_full.json',
#                data_file = "/data/liuaohan/co-training/rsna4_data80_10.h5")
'''
temp = list(range(100))
train = random.sample(temp, 80)
valid = list(set(temp)-set(train))
with open("/data/liuaohan/co-training/38713_training_temp.pkl", 'wb') as fp:
    pkl.dump(train, fp)
with open("/data/liuaohan/co-training/38713_valid_temp.pkl", 'wb') as fp:
    pkl.dump(valid, fp)
'''
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', naoshi=True, 
#               use_index_file = "/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl", save_ratio = 'NLP')
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, soft = True, use_hand_labeled = False, label_file_name = 'NLP_baseline_soft')


#用下面这些生成数据训练CT模型。300因为训练是在太少了，不用手标数据。3000为了和baseline不差太多，还是用了手标数据
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_300', use_hand_labeled = False)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_1500', use_hand_labeled = False)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_99', use_hand_labeled = False)

#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_1000')
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_3000')

#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_2000')
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_4000')
#前瞻数据集label：
#analyse_report('/data/liuaohan/co-training/prospective_160_80.h5', 'prospective.pkl', 
#               naoshi = True, suffix = '_prospective' , label_file_name = 'all_refined_label_ischemia_false_naoshi_prospective') #前瞻性先用NLP标签测试看下效果如何
#analyse_report('/data/liuaohan/co-training/prospective_160_80.h5', 'prospective2.pkl', 
#               naoshi = True, suffix = '_prospective2' , label_file_name = 'all_refined_label_ischemia_false_naoshi_prospective2') #之前报告弄错了

#analyse_report('/data/liuaohan/co-training/prospective_160_80.h5', 'prospective3.pkl', 
#               naoshi = True, suffix = '_prospective2' , doctor_label = True,
#               label_file_name = 'all_refined_label_ischemia_false_naoshi_prospective3', 
#               use_index_file = '/data/liuaohan/co-training/prospective_good6_1500.pkl', save_ratio='prospective') #医生手工标注了1500个序列
#5.30.2022本来是3，为了检查prospective标注重新生成
#2023.3.5注释掉下面这个
#analyse_report('/data/liuaohan/co-training/prospective_160_80.h5', 'prospective3.pkl', 
#               naoshi = True, suffix = '_prospective3' , doctor_label = True,
#               label_file_name = 'all_refined_label_ischemia_false_naoshi_prospective5', 
#               use_index_file = '/data/liuaohan/co-training/prospective_good6_1500.pkl', save_ratio='prospective') #医生手工标注了1500个序列



'''

label_file = tables.open_file('all_refined_label_ischemia_false_naoshi_prospective4.h5', "r")
label_file2 = tables.open_file('all_refined_label_ischemia_false_naoshi_prospective3.h5', "r")
data_file = tables.open_file("/data/liuaohan/co-training/prospective_160_80.h5", "r")
print(len(label_file.root.label), len(data_file.root.subject_ids))
for i in range(len(data_file.root.subject_ids)):
    path = data_file.root.subject_ids[i].decode('utf-8')
    temp = Path(path).parent.parent.name
    if temp=='385741':
        print(i)
    if temp=='A109891':
        print(i)
    if temp=='C869270':
        print(i)
    if temp=='G118441':
        print(i)
    if temp=='K0625490':
        print('K0625490', i)
#print(label_file.root.label[73])
#print(label_file.root.label[608])
cnt = 0
for i in range(len(label_file.root.label)):
    if np.any(label_file.root.label[i]!=label_file2.root.label[i]):
        print('different', i)
    if label_file.root.label[i][12,3]:
        print(data_file.root.subject_ids[i].decode('utf-8'))
        cnt += 1
print(cnt)
label_file.close()
label_file2.close()
data_file.close()
'''
#用下面这些生成数据，测试NLP模型准确率
#for x in ['_300','_1000','_2000','_3000', '_1500']:
#for x in ['_99','_100','_101']:
#    analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
 #                 naoshi = True, suffix = x, soft = True, use_hand_labeled = False, label_file_name = 'label_soft%s'%x)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '', soft = True, use_hand_labeled = False, label_file_name = 'label_soft')

#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_1000', soft = True, use_hand_labeled = False)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_2000', soft = True, use_hand_labeled = False)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_300', soft = True, use_hand_labeled = False)
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, suffix = '_3000', soft = True, use_hand_labeled = False)

#生成医生标注测试集的label file：
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, doctor_label = True, label_file_name='all_refined_label_ischemia_false_naoshi_doctor')
#2023.3.5我忘了下面这个东西和上面多了两个参数是干嘛的了，应该是统计retrospective测试集的异常比例？
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, doctor_label = True, label_file_name='all_refined_label_ischemia_false_naoshi_doctor',
#               use_index_file = "/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl", save_ratio='retrospective')


'''
a = tables.open_file('all_refined_label_ischemia_false_naoshi_1000.h5', 'r')
b = tables.open_file('all_refined_label_ischemia_false_naoshi_2000.h5', 'r')
for i in range(100):
    print(np.sum(a.root.label[i]-b.root.label[i]))
a.close()
b.close()
'''
def make_regular_ratio(file):
    np.set_printoptions(suppress=True)
    ratio = np.loadtxt(file+'.txt')
    print(np.round(ratio, 4))
    n,m = ratio.shape
    organ_ratio = np.sum(ratio, 1, keepdims = True)
    abnormality_ratio = np.sum(ratio, 0, keepdims = True)
    print(organ_ratio.shape)
    print(abnormality_ratio)
    
    result = np.dot(organ_ratio, abnormality_ratio)/np.sum(ratio)
    print(np.round(result, 4))
    
    print(np.sum(result, 0))
    print(np.sum(ratio, 0))
    print(np.sum(result, 1))
    print(np.sum(ratio, 1))
    np.savetxt(file+'_regular.txt', result)
    
#make_regular_ratio('positive_ratio_naoshi')
'''
#80的测试集1403，比之前1448要少，主要是因为，80是在label_result_38713_unique基础上加标脑室得到的label，
#之前是用label_result_38713，这个是在生成label_result_38713_unique之后我当时又改标了一些，所以“不确定”的情况少。
data_file = tables.open_file('all_refined_label_ischemia_false.h5', "r")
data_file2 = tables.open_file('all_refined_label_ischemia_false_naoshi.h5', "r")

with open("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl", 'rb') as fp:
    ids = pickle.load(fp)
print(len(ids))
print(len(data_file.root.label))
label = data_file.root.label
label2 = data_file2.root.label

cnt = 0
cnt2 = 0
for i in ids:
    if label[i][13,3]!=label2[i][12,3]:
        print(i, label[i][13,3], label2[i][12,3])
    if np.max(label[i][7])!=np.max(label2[i][7]):
        print(i, label[i][7], label2[i][7])
    if label[i][13,3]==0 and np.max(label[i][7])==0:
        cnt += 1
    if label2[i][12,3]==0 and np.max(label2[i][7])==0:
        cnt2 += 1
    
print(cnt, cnt2)

all_paths = list(Path('prediction/test_multitask80').rglob('*label.txt'))
cnt = 0
for path in tqdm(all_paths):
    label2 = np.loadtxt(str(path))
    label = np.loadtxt(str(path).replace('80','76', 1))
    if np.max(label2[7])==0 and label2[12,3]==0:
        cnt += 1
    if label[13,3]!=label2[12,3]:
        with open(str(path).replace('label', 'report'), 'r') as fp:
            print(fp.readlines())
        print(path, label[13,3], label2[12,3])
print(cnt)

data_file.close()
data_file2.close()
'''
'''
with open("/data/liuaohan/co-training/rsna_training.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/rsna_test.pkl", 'wb') as fp:
    pkl.dump(list(range(975)), fp)
'''
'''
with open("/data/liuaohan/co-training/rsna2_training.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/rsna2_test.pkl", 'wb') as fp:
    pkl.dump(list(range(856)), fp)
'''
'''
with open("/data/liuaohan/co-training/rsna4_training.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/rsna4_test.pkl", 'wb') as fp:
    pkl.dump(list(range(4194)), fp)
'''

'''
#宇巍前瞻数据集
with open('/home2/heyuwei/edgetask/buffer/study_path_pro_20191024/prospecV2.json', 'r') as fp:
    reports = json.load(fp)
print([x for x in reports])
desc = {}
imp = {}
paths = {}
for x in reports['study_id']:
    if reports['study_id'][x]!='':
        if reports['study_id'][x] in desc:
            print(reports['study_id'][x])
            print('pre', desc[reports['study_id'][x]])
            print('new', reports['DESCRIPTION'][x])
        else:
            desc[reports['study_id'][x]] = reports['DESCRIPTION'][x]
            imp[reports['study_id'][x]] = reports['IMPRESSION'][x]
        #assert reports['study_path'][x] not in paths, reports['study_path'][x]
        #paths[reports['study_path'][x]] = 1

with open('/home/heyuwei/CT/all_sick_classification_pro_20211025/valid.json', 'r') as fp:
    sick = json.load(fp)
for x in tqdm(sick[:]):
    dcm = pydicom.read_file(x['dcm_paths'][0])
    assert dcm.StudyInstanceUID in desc
    assert x['report']==imp[dcm.StudyInstanceUID], (x,imp[dcm.StudyInstanceUID])
    #print(x['report'], imp[dcm.StudyInstanceUID])

print(sick[0])
print(len(sick))
'''
'''
with open("/data/liuaohan/co-training/prospective_training.pkl", 'wb') as fp:
    pkl.dump([], fp)
with open("/data/liuaohan/co-training/prospective_test.pkl", 'wb') as fp:
    pkl.dump(list(range(3846)), fp)
'''

#2023.3.6，生成训练集上全部手标的小规模标注数据（前1170例是手标的，其他的用nlp预测的也行，反正我训练也不用）
#check_training_labeled() #前1170个可用
#analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
#               naoshi = True, doctor_label = True, label_file_name='all_refined_label_ischemia_false_naoshi_smallsilver',
#               label_file = '/home/liuaohan/pm-stroke/project/text_classification_yingqi/nlp_2/label_result_training_.csv')
def make_small_hand_labeled_dataset():
    indexes = pickle.load(open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl", 'rb'))
    temp = indexes[:1169]
    random.seed(10)
    random.shuffle(temp)
    with open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small4_train.pkl", 'wb') as fp:
        pkl.dump(temp[:936], fp)
    with open("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small4_valid.pkl", 'wb') as fp:
        pkl.dump(temp[936:], fp)
#make_small_hand_labeled_dataset()
#查看一下侧脑室的低密度是什么病：
'''
analyse_report('/data/liuaohan/co-training/all_brain_data_160_80.h5', 'all_filtered4.pkl', 
       naoshi = True, doctor_label = True, label_file_name='temp',
       use_index_file = "/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl")
analyse_report('/data/liuaohan/co-training/prospective_160_80.h5', 'prospective3.pkl', 
               naoshi = True, suffix = '_prospective3' , doctor_label = True,
               label_file_name = 'temp', 
               use_index_file = '/data/liuaohan/co-training/prospective_good6_1500.pkl', save_ratio='prospective') #医生手工标注了1500个序列
'''
