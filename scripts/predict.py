import os
from pathlib import Path
import itertools
import cv2
import numpy as np
import json
from tqdm import tqdm
import math
import nibabel as nib
import prettytable as pt
import glob
import xlrd
import re
import sys
import copy
from keras.activations import softmax
from keras import backend as K
import tensorflow as tf
import random
#from train import config
#from train_isensee2017 import config
#from train_mil_upsample import config
#from train_mil import config
#from train_mil_feature import config



#from train_mil_multiinput import config
#from train_mil_multitask_new import config
#from train_mil_multitask_suc import config
project_path = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from unet3d.training import load_old_model
from unet3d.prediction import run_validation_cases, run_single_case
from unet3d.prediction import resolve_subject_id
from unet3d.utils.utils import maxpool
from unet3d.utils import pickle_load

import tables
from tqdm import tqdm


parts = ['左半卵圆中心', '右半卵圆中心', '左颞叶', '右颞叶', '左小脑', '右小脑', 
         '左岛叶', '右岛叶', '左额叶', '右额叶',  
         '左顶叶', '右顶叶', '脑桥', '左丘脑',   '右丘脑',
         '左基底节区', '右基底节区', '脑干', '左枕叶','右枕叶']

def resize(img, shape, inter=cv2.INTER_NEAREST):
    """
    三维resize
    """
    #print('resize', img.shape, shape)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=inter) #行列resize
    img = np.transpose(img, (0,2,1)) #160,5,160
    img = cv2.resize(img, (shape[2], shape[0]), interpolation=inter) #纵向resize
    img = np.transpose(img, (0,2,1))
    return img

def make_single_png(CT, pred, gt, path, refined, prefix = '', seg = None, abnormal = None, organs = None,
                    save = True):
    """
    pred是一个三维图片的list，abnormal也是
    seg一般是表示不同区域的分割结果，abnormal是表示整个图像的异常预测结果（不使用seg过滤）
    """
    print('make single png')
    CT = CT.copy()
    pred = pred.copy()
    gt = gt.copy()
    
    #print('CT', CT.shape)
    #print('pred', pred.shape)
    #print('gt', gt.shape)
    assert CT.shape==gt.shape
    #print(np.max(CT), np.min(CT))
    maxn = np.max(CT)
    minn = np.min(CT)
    CT = np.uint8((CT-minn)/(maxn-minn)*255)
    
    if seg is not None:
        seg = seg.copy()
        seg = resize(seg, CT.shape)
        seg = np.transpose(seg, [1,0,2])
        seg = cv2.flip(seg, 0) #尝试不加这个？？
        #seg = cv2.flip(seg, 1) #尝试不加这个？？
        
    if abnormal is not None:
        abnormal = abnormal.copy()
        for i in range(len(abnormal)):
            abnormal[i] = resize(abnormal[i], CT.shape)
            abnormal[i] = np.transpose(abnormal[i], [1,0,2])
            abnormal[i] = cv2.flip(abnormal[i], 0)
        
    CT = np.transpose(CT, [1,0,2])
    CT = cv2.flip(CT, 0)
    assert type(pred)==list
    for i in range(len(pred)):
        pred[i] = resize(pred[i], CT.shape)
        pred[i] = np.transpose(pred[i], [1,0,2])
        pred[i] = cv2.flip(pred[i], 0)
    
    
    print(len(pred), len(abnormal))
    #print('gt', gt.shape, gt.dtype)
    gt = np.transpose(gt, [1,0,2])
    if not refined:
        gt = cv2.flip(gt, 0)
    else:
        label = np.loadtxt(str(path)+'/label.txt')
        mask = np.zeros(CT.shape, np.uint8)
        #print('label', label.shape, label.dtype)
        for i in organs:
            temp = np.bitwise_and(np.right_shift(gt, i), 1)
            
            temp = np.uint8(temp)
            mask = np.maximum(mask, cv2.flip(temp, 0))
        #mask = maxpool(mask, 16)
        #print('mask', mask.shape)
        mask = resize(mask, CT.shape)
        #print('mask', mask.shape)
        gt = mask
        
    ret = []
    for i in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2)
        output = CT3.copy()
        for x in pred:
            heat_img = cv2.applyColorMap(np.uint8(x[:,:,i]*255), cv2.COLORMAP_JET)
            pred_add = cv2.addWeighted(CT3, 0.5, heat_img, 0.5, 0)
            output = np.concatenate((output, pred_add), 1)
            #print(output.shape)
        heat_img = cv2.applyColorMap(np.uint8(gt[:,:,i]*255), cv2.COLORMAP_JET)
        gt_add = cv2.addWeighted(CT3, 0.5, heat_img, 0.5, 0)
        output = np.concatenate((output, gt_add), 1)
        
        if seg is not None:
            heat_img = cv2.applyColorMap(np.uint8(seg[:,:,i]*255), cv2.COLORMAP_JET)
            seg_add = cv2.addWeighted(CT3, 0.5, heat_img, 0.5, 0)
            #cv2.imwrite(str(path)+'/organ_%s.png'%str(i).zfill(2), np.uint8(seg[:,:,i]*255))
            output = np.concatenate((output, seg_add), 1)
        if abnormal is not None:
            for x in abnormal:
                heat_img = cv2.applyColorMap(np.uint8(x[:,:,i]*255), cv2.COLORMAP_JET)
                abnormal_add = cv2.addWeighted(CT3, 0.5, heat_img, 0.5, 0)
                output = np.concatenate((output, abnormal_add), 1)
        if save:
            cv2.imwrite(str(path)+'/result%s_%s.png'%(prefix, str(i).zfill(2)), output)
        else:
            ret.append(output)
    return ret

def make_organ_seg(CT, pred, gt, path, prefix = '', organs = []):
    """
    为了输出organs这些脑区的预测分割和配准分割
    CT是三维图像
    pred是四维图像，就是网络的预测值，第一维是channel维，和make_single_png中不同
    gt是三维图像，数值是二进制的mask，和make_single_png相同
    """
    CT = CT.copy()
    pred = [pred[i] for i in range(pred.shape[0])]
    gt = gt.copy()
    
    maxn = np.max(CT)
    minn = np.min(CT)
    CT = np.uint8((CT-minn)/(maxn-minn)*255)
    #改为(二维高、二维宽、深度)的标准格式：
    CT = np.transpose(CT, [1,0,2])
    CT = cv2.flip(CT, 0)
    
    for i in organs:
        pred[i] = resize(pred[i], CT.shape)
        pred[i] = np.transpose(pred[i], [1,0,2])
        #print(pred[i].dtype)
        pred[i] = cv2.flip(pred[i], 0) #float32也可以翻转
    
    gt = np.transpose(gt, [1,0,2])
    #gt = cv2.flip(gt, 0)
    
    for j in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,j], CT[:,:,j], CT[:,:,j]), 2)
        output = np.zeros((CT3.shape[0]*2, 0, 3), np.uint8)
        for i in organs:
            heat_img = cv2.applyColorMap(np.uint8(pred[i][:,:,j]*255), cv2.COLORMAP_JET)
            pred_add = cv2.addWeighted(CT3, 0.7, heat_img, 0.3, 0)

            temp = np.bitwise_and(np.right_shift(gt[:,:,j], i), 1)
            temp = cv2.flip(np.uint8(temp), 0)
            heat_img = cv2.applyColorMap(np.uint8(temp*255), cv2.COLORMAP_JET)
            gt_add = cv2.addWeighted(CT3, 0.7, heat_img, 0.3, 0)
            
            img = np.concatenate((pred_add, gt_add), 0)
            output = np.concatenate((output, img), 1)
        cv2.imwrite(str(path)+'/seg_%s.png'%(str(j).zfill(2)), output)

def AP(scores):
    """
    scores是(label, score)的list
    """
    scores.sort(key = lambda x: x[1], reverse = True)
    #print(scores)
    all_pos = len(list(filter(lambda x:x[0], scores)))
    if all_pos==0:
        return 0
    print('all pos', all_pos)
    positive = 0
    recall = 0
    ret = 0
    for i, x in enumerate(scores):
        if x[0]:
            positive += 1
        precision = positive/(i+1)
        ret += (positive/all_pos-recall)*precision #没有用梯形计算面积，直接用的柱形，所以可能会低一些
        recall = positive/all_pos
    return ret

def AUC(scores):
    """
    scores: [(gt,pred),]
    """
    scores = copy.deepcopy(scores)
    scores.sort(key = lambda x: x[1], reverse = True)
    #print(scores)
    ap = sum([x[0] for x in scores]) #actual positive
    an = len(scores)-ap #actual negative
    if ap==0 or an==0:
        return -1
    x = [0]
    y = [0]
    tp = 0
    fp = 0
    ret = 0
    i = 0
    while i<len(scores):
        #print(i)
        memtp = tp
        memfp = fp
        now_score = scores[i][1]
        while i<len(scores) and scores[i][1]==now_score:
            if scores[i][0]:
                tp += 1
            else:
                fp += 1
            i += 1
        ret += (fp-memfp)/an*(memtp+tp)/2/ap #如果使用这种方法，就可能出现斜线，从而可以计算梯形面积
        #无论[(1,0.6), (0,0.6)]还是[(0,0.6), (1,0.6)]结果都是0.5
        x.append(fp/an)
        y.append(tp/ap)
    #plt.plot(x,y)
    #print(x)
    #print(y)
    return ret

def top_k_mask(img, k):
    a = np.reshape(img, (-1)).tolist()
    #print(a)
    a.sort(reverse = True)
    t = a[k]
    return np.uint8(np.greater_equal(img, t))

def top_values(img, pred):
    maxn = np.max(img)
    minn = np.min(img)
    img = np.uint8((img-minn)/(maxn-minn)*255)
    if img.shape!=pred.shape:
        #pred = resize(pred, img.shape)
        img = resize(img, pred.shape)
    mask = top_k_mask(pred, 5)
    temp = mask*img
    values = temp.ravel()[np.flatnonzero(temp)].tolist()
    values.sort(reverse=True)
    if len(values)>10:
        return []
    return values

def distance_map_3D(img, f):
    """
    img应该是三维array
    """
    n,m,l = img.shape
    d1 = np.zeros(img.shape)
    
    for i in range(n):
        d1[i,:,:] = f(fast_distance_map(img[i,:,:]))
    d2 = np.zeros(img.shape)
    for i in range(m):
        d2[:,i,:] = f(fast_distance_map(img[:,i,:]))
    d3 = np.zeros(img.shape)
    for i in range(l):
        d3[:,:,i] = f(fast_distance_map(img[:,:,i]))
    #如果不先做f，这里会溢出！当img全零时，d为最小，三个d加在一起就成了-inf。不过python对-inf做运算也是可以的，exp(-inf)=0
    return (d1+d2+d3)/3 


def distance_map_3D_improve(img, f):
    """
    img应该是uint三维01矩阵
    """
    def distance_map_3D_single(img):
        """
        single指只在第0个轴面上进行计算，并进行层面间传播
        """
        n = img.shape[0]
        din = np.zeros(img.shape)
        dout = np.zeros(img.shape)
        for i in range(n):
            din[i,:,:] = cv2.distanceTransform(img[i,:,:], cv2.DIST_L2, 3)
            dout[i,:,:] = cv2.distanceTransform(1-img[i,:,:], cv2.DIST_L2, 3)
        for i in range(1,n):
            din[i,:,:] = np.minimum(din[i-1,:,:]+1, din[i,:,:])
            dout[i,:,:] = np.minimum(dout[i-1,:,:]+1, dout[i,:,:])
        for i in range(n-2,-1,-1):
            din[i,:,:] = np.minimum(din[i+1,:,:]+1, din[i,:,:])
            dout[i,:,:] = np.minimum(dout[i+1,:,:]+1, dout[i,:,:])
        return din, dout
    
    n,m,l = img.shape
    
    din1,dout1 = distance_map_3D_single(img)
    din2,dout2 = distance_map_3D_single(np.transpose(img, [1,0,2]))
    din2 = np.transpose(din2, [1,0,2])
    dout2 = np.transpose(dout2, [1,0,2])
    din3,dout3 = distance_map_3D_single(np.transpose(img, [2,1,0]))
    din3 = np.transpose(din3, [2,1,0])
    dout3 = np.transpose(dout3, [2,1,0])
    
    
    din = np.minimum(np.minimum(din1, din2), din3)
    dout = np.minimum(np.minimum(dout1, dout2), dout3)
    return f(din-dout)
    

def fast_distance_map(img):
    """
    img应该是01矩阵
    """
    img = np.uint8(img)
    din = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1-img, cv2.DIST_L2, 3)
    #当img全0时，din=0, dout非常大
    return din-dout

def exp_s(x, s = 1): 
    return np.exp(x/s)
def sigmoid(x, scale = 10):#s越小mask越紧凑，越大越松散
    return 1/(1+np.exp(-x/scale))

def p_map(img):
    img = np.uint8(np.greater(img, 0.5))
    n = img.shape[0]
    d = np.zeros(img.shape)
    for i in range(n):
        #d[i] = distance_map_3D(img[i], f=lambda x:exp_s(x, 10))
        d[i] = distance_map_3D_improve(img[i], f=lambda x:exp_s(x, 1))
        if i in [6,7,12,13,14]:
            d[i] = 0
    s = np.sum(d, 0)+1e-15 #如果某层上无分割，则s=0，小心产生NaN
    ret = d/s
    return ret

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
    #print(np.max(img), np.min(img))
    #print(type(img), img.dtype, img.shape)
    #print(img.shape)
    n = img.shape[2]
    tot = 1
    mem = [None,]
    area = np.zeros(255) #从1开始使用
    size = np.zeros(255)
    fa = np.arange(255)#并查集
    image = np.zeros(img.shape, np.uint8) #在这上面记录不同连通区域
    for i in range(n):
        contours,hierarchy = cv2.findContours(mask[:,:,i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area[tot] = cv2.contourArea(c)
            #print(cv2.fillPoly(image[:,:,i], [c], tot))
            image[:,:,i] = cv2.fillPoly(image[:,:,i], [c], tot).get()
            mem.append((i,c))
            tot += 1
        
        if i>0:
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if image[x,y,i] and image[x,y,i-1]:
                        fa[get(image[x,y,i],fa)] = get(image[x,y,i-1], fa)
    for i in range(1,tot):
        size[get(i,fa)] += area[i]
        #print(i,get(i,fa))
    print(tot, size[np.nonzero(size)])
    max_id = np.argmax(size)
    #print(max_id)
    for i in range(1,tot):
        if get(i,fa)!=max_id:
            print('change', mem[i][0], area[i])
            before = np.sum(img[:,:,mem[i][0]])
            #print(img[mem[i][0]])
            #print(cv2.fillPoly(img[:,:,mem[i][0]], [mem[i][1]], 0))
            #print(type(img[mem[i][0]]),
            #      type(cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0)))
            #print(img[mem[i][0]].dtype)
            #print(cv2.fillPoly(img[mem[i][0]], [mem[i][1]], 0).dtype)
            
            img[:,:,mem[i][0]] = cv2.fillPoly(img[:,:,mem[i][0]], [mem[i][1]], 0).get()
            print(np.sum(img[:,:,mem[i][0]]), before)
    return

def remove_symmetry_scores(scores):
    """
    返回和scores基本相同，但是如果某个样本gt有异常，而且其对称脑区此样本也有异常，则忽略此样本
    """
    cids = [1,0,3,2,5,4,7,6,9,8,11,10,12,14,13,16,15,17,19,18]
    ret = {}
    for tag in scores:
        ret[tag] = []
        organ = tag[0]
        symmetry_organ = cids[organ]
        #print(tag, symmetry_organ, (symmetry_organ, tag[1]))
        if organ!=symmetry_organ:
            for i in range(len(scores[tag])):
                if scores[tag][i][0] and scores[(symmetry_organ, tag[1])][i][0]:
                    continue
                ret[tag].append(scores[tag][i])
    return ret

def load_cq500_label(file, naoshi=False):
    def vote(row, col):
        cnt = 0
        for i in range(3):
            cnt += sh.cell_value(row, col+i*14)
        return 1 if cnt>=2 else 0
    
    data = xlrd.open_workbook(file)
    sh = data.sheet_by_index(0)
    n = sh.nrows
    ret = {}
    for i in range(1,n):
        label = []
        label.append(vote(i,3)) #脑内高
        label.append(1 if (vote(i,4) if naoshi else False) or vote(i, 5) or vote(i,6) or vote(i, 7) else 0) #脑外高
        label.append(vote(i, 8)) #左高
        label.append(vote(i, 9)) #右高
        ret[sh.cell_value(i, 0)[9:]] = label
    return ret
def load_scores_single_region(file, scores, il = False):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    a = []
    for l in lines:
        if l=='':
            continue
        s = l.strip().split(' ')
        gt = float(s[2 if not il else 1])
        pred = float(s[3 if not il else 2])
        path = s[4 if not il else 3].strip()
        if not il:
            scores[s[1]+'_'+s[0]+'_'+path] = (gt,pred) #主要是去重，因为txt里可能有好几次的结果
        else:
            scores['-1'+'_'+s[0]+'_'+path] = (gt,pred) #image level把organ设成0，方便处理
        a.append(pred)
    return np.mean(np.array(a)), np.std(np.array(a))

def renew_label(dir, paths = None, flip = False):
    """
    更换医生标注之后，重新进行测试。因为调用make_png还要读nii有点慢，可以直接用之前预测值配合新的label进行更新
    当然之前有一些序列因为存在不明区域、不是手标等原因没有记录预测值，那一部分序列还是需要调用make_png预测一次。
    """
    if paths is None:
        all_paths = list(Path(dir).glob('*'))
    else:
        all_paths = [Path(x) for x in paths]
    organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
    label = {}
    for path in all_paths:
        label[path.name] = np.loadtxt(str(path)+'/label.txt') #更新label
        label[path.name][12,3] = 0 #把NLP生成标志去掉，这样计算脑外低IL AUC才是对的
    scores = {}
    for i in organs:
        for a in range(4):
            scores[(i,a)] = []
    scores_il = {a:[] for a in range(4)}
    for i in organs:
        sss = {}
        load_scores_single_region('%s/scores_%d.txt'%(dir,i), sss)
        with open('%s/scores_%d.txt'%(dir,i), 'a') as fp:
            for x in sss:
                pred = float(sss[x][1])
                a = int(x.split('_')[1])
                path = '_'.join(x.split('_')[2:])
                if not path in label: #可能之前记录的序列结果比较多，但是新用的label_file只有其中一部分
                    continue
                if (not config['public_test'] or config['public_test']=='prospective') and (np.max(label[path], 1))[7]:
                    continue #除去不明区域
                scores[(i,a)].append((label[path][i,a], pred))
                fp.write(str(a)+' '+str(i).zfill(2)+' '+str(label[path][i,a])+' '+str(pred)+' '+path+'\n')
    sss = {}
    load_scores_single_region('%s/scores_il.txt'%dir, sss, il = True)                  
    with open('%s/scores_il.txt'%dir, 'a') as fp:
        for x in sss:
            pred = float(sss[x][1])
            a = int(x.split('_')[1])
            path = '_'.join(x.split('_')[2:])
            if not path in label:
                continue
            gt_label = np.max(label[path], 0)[a]
            scores_il[a].append((gt_label, pred))
            fp.write(str(a)+' '+str(gt_label)+' '+str(pred)+' '+path+'\n')
    log_AUC(dir, scores, scores_il, organs, False, 4)

def log_AUC(dir, scores, scores_il, organs, soft_seg, n_abnormals):
    micro_scores = {a:[] for a in range(n_abnormals)}
    with open(dir+'/mAP_%d.txt'%config['version'], 'a') as fp:
        fp.write(str(len(scores[(organs[0],0)]))+'\n')
        fp.write('soft segmentation '+str(soft_seg)+'\n')
        mAP = 0
        overall_mAP = 0 #有异常的mAP，不管异常类别
        macro_auc = {}
        macro_pos = {} #用来帮助计算加权macro_auc
        macro_pos_d = {}
        macro_auc_d = {}
        for a in range(n_abnormals):
            macro_auc[a] = []
            macro_pos[a] = []
            macro_auc_d[a] = []
            macro_pos_d[a] = []
        tb = pt.PrettyTable(border = False) #记录AUC
        tb2 = pt.PrettyTable(border = False) #用于记录mAP
        tb3 = pt.PrettyTable(border = False)
        for i in organs:
            for a in range(n_abnormals):
                n_pos = len(list(filter(lambda x:x[0], scores[(i,a)])))
                if n_pos==0:
                    continue
                ap = AP(scores[(i,a)])
                auc = AUC(scores[(i,a)])
                #nc_auc = AUC(nc_scores[(i,a)])
                if auc>=0: #auc!=NaN
                    macro_auc[a].append(auc)
                    macro_pos[a].append(n_pos)
                print('AUC_%d_%d'%(i,a), auc)
                tb.add_column('%d_%d'%(i,a), [round(auc, 4), n_pos])
                tb2.add_column('%d_%d'%(i,a), [round(ap, 4), n_pos])
                mAP += ap
            #print('AP_%d'%i, ap)
            #fp.write('AP_%d'%i+' '+str(ap)+'\n')
        fp.write(str(tb)+'\n')
        fp.write(str(tb3)+'\n')
        fp.write(str(tb2)+'\n')
        
        tb = pt.PrettyTable(border = False)
        for a in range(n_abnormals):
            temp = np.mean(np.asarray(macro_auc[a]))
            tb.add_column('macro AUC%d'%a, [np.round(temp, 4)])
        for a in range(n_abnormals):
            for i in organs:
                micro_scores[a].extend(scores[(i,a)])
            auc = AUC(micro_scores[a])
            tb.add_column('micro AUC%d'%a, [round(auc, 4)])
        
        for a in range(n_abnormals):
            weighted = np.average(np.asarray(macro_auc[a]), weights=np.asarray(macro_pos[a]))
            tb.add_column('w macro%d'%a, [np.round(weighted, 4)])
        for a in range(n_abnormals):
            auc = AUC(scores_il[a])
            tb.add_column('image AUC%d'%a, [round(auc, 4)])
        fp.write(str(tb)+'\n')
        print(tb)
        mAP /= len(organs)*n_abnormals
        print('mAP', mAP)
        fp.write('mAP'+' '+str(mAP)+'\n')
        overall_mAP /= len(organs)
        fp.write('overall mAP'+' '+str(overall_mAP)+'\n')
        fp.write('\n')

def IOU(a, b):
    assert a.shape==b.shape
    I = np.sum(np.minimum(a,b))
    U = np.sum(np.maximum(a,b))
    print('I,U', I, U)
    return I/(U+1e-9)
def Dice(a, b):
    assert a.shape==b.shape
    I = np.sum(np.minimum(a,b))
    return 2*I/(np.sum(a)+np.sum(b)+1e-9)

def make_png(dir, report_desc, report_label, report, fa, paths = None, filt = None, refined = False, multitask=False,
             dynamic_mask = False, flip = False):
    if paths is None:
        all_paths = list(Path(dir).glob('*'))
    else:
        all_paths = [Path(x) for x in paths]
    print('paths loaded', len(all_paths))
    cnt = {}
    tot = 0
    #organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19]
    organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
    cids = [1,0,3,2,5,4,6,7,9,8,11,10,12,14,13,16,15,17,19,18]
    #organs = [4,5]
    #organs = [3]
    #organs = [0,2,4,8,10,15,17,18]
    #organs = [15]
    
    #organs = [8,9]
    n_abnormals = 4
    
    use_coarse_label = (True if config['version']==43 else False)
    #use_coarse_label = True
    
    print(use_coarse_label)
    if config['version']>=80:
        hand_label_i = 12
    elif config['version']>=44:
        hand_label_i = 13
    else:
        hand_label_i = 0
    
    
    save_image = False
    save_segmentation = False
    evaluate_seg = True
    
    soft_seg = False
    write_score = True
    remove_small_component = False
    ignore_vague = False
    use_sum_score = False
    use_side_mask = False
    cq500_use_low = False
    ignore_outside_organs = True #测试CQ500时，是使用所有organ中的异常最大值，还是用整个图像的异常最大值
    naoshi = True #测试CQ500时，gt的脑外出血是否加入脑室出血
    use_real_image_level = True #对于加入整个图像异常loss的模型，如85、82，可使用true，其他模型使用true效果会很差
    
    scores = {} #记录每个organ的每个series的label和预测值，用来计算mAP
    score_cq500 = {}
    score_il_cq500 = {}
    micro_scores = {}
    cls_scores = {} #每个organ的每个series的label，和用直接分类得到的预测值
    overall_scores = {}
    scores_d = {}
    scores_il = {} #scores_image_level，序列级别各种异常的预测值
    for i in organs:
        for a in range(n_abnormals):
            scores[(i,a)] = []
            cls_scores[(i,a)] = []
        overall_scores[i] = []
        for a in range(2):
            scores_d[(i,a)] = []
    for a in range(n_abnormals):
        micro_scores[a] = []
        scores_il[a] = []
    label_cnt = None #统计gt label各脑区的异常分布
    label_cnt2 = None #统计gt label序列级的异常分布
    all_dice = {(a,t):[] for a in range(4) for t in range(0, 10)}
    all_iou = {(a,t):[] for a in range(4) for t in range(0, 10)}
    path_i = 0
    for path in tqdm(all_paths):
        #print(path)
        #print(path)
        #path_i += 1
        if not path.is_dir():
            print('not path', path)
            continue
        tot += 1
        if len(path.name.split('_'))==2: #之前生成的先不管
            continue
        seriesID = path.name.split('_')[-1]
        reportID = path.name.split('_')[-2]
        #if int(path.name[:2])<14 or int(path.name[:2])>20:
        #    continue
        #print(reportID, seriesID)
        flag = False
        if filt is not None:
            for x in filt:
                if reportID==x[0]:
                    print('same report', reportID)
                    flag = True
                if seriesID==x[1]:
                    print('same series', seriesID)
                    flag = True
        if flag:
            print('in training set', seriesID)
            continue
        try:
            if evaluate_seg:
                if not Path(str(path)+'/ab_seg.nii.gz').exists():
                    continue
                ab_seg = nib.load(str(path)+'/ab_seg.nii.gz').get_data()[0] #[160,160,80]
            else:
                ab_seg = None
            try:
                CT = nib.load(str(path)+'/data_%s.nii.gz'%config["all_modalities"][0]).get_data()
            except:
                if config['version']==85: #85虽然用的是未去骨的CT，之前模态却叫CT_brain。生成85.111（其他脑区分割文件）时改为了CT。不重新运行predict了，这样打下补丁吧
                    CT = nib.load(str(path)+'/data_CT_brain.nii.gz').get_data()
            pred_paths = list(path.glob('prediction_*.nii.gz'))
            preds = []
            for p in pred_paths:
                preds.append(nib.load(str(p)).get_data())
            try:
                pred = nib.load(str(path)+'/prediction.nii.gz').get_data()
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
            except:
                pred = preds[-1]
            try: #一般情况下
                gt = nib.load(str(path)+'/truth.nii.gz').get_data()
            except: #prospective目前没有配准，没有分割truth
                gt = np.zeros(CT.shape, np.uint32)
        except: #正常是不会有这种情况的，但是有时候删文件夹就是删不干净！
            continue
        if save_segmentation:
            make_organ_seg(CT, pred, gt, path, prefix = '', organs = organs)

        #mask = np.zeros(gt.shape, np.uint8)
        if remove_small_component:
            assert dynamic_mask,'' #dynamic_mask=False还未修改
            for i in range(20): 
                temp = pred[i].copy()
                get_max_component(pred[i])
                print('diff', np.sum(pred[i]-temp))
        if soft_seg:
            mask = []
            brain_mask = np.uint8(np.not_equal(np.min(CT), CT))
            brain_mask = resize(brain_mask, pred[0].shape, inter=cv2.INTER_NEAREST)
            other_organ = np.zeros(pred[0].shape) #不是任何脑区的mask
            #print('label', label.shape, label.dtype)
            for i in range(20): 
                if not dynamic_mask:
                    temp = np.bitwise_and(np.right_shift(gt, i), 1)
                    temp = np.uint8(temp)
                    temp = resize(temp, pred[0].shape, inter=cv2.INTER_NEAREST)
                    other_organ = np.maximum(other_organ, temp)
                    mask.append(temp)
                else:
                    temp = np.uint8(np.greater(pred[i], 0.1))
                    other_organ = np.maximum(other_organ, temp)
                    mask.append(temp)
            other_organ = (1-other_organ)*brain_mask
            mask.append(other_organ) #将脑中不属于任何脑区的地方也作为一个脑区插入
            #print('mask', mask.shape)
            mask = np.stack(mask, 0)
            #print(mask.shape)
            #mask = resize(mask, pred.shape)
            
            p_seg = p_map(mask)
            for i in range(20):
                p_seg[i] *= brain_mask
                maxn = np.max(p_seg[i])
                if maxn>0: #归一化：
                    p_seg[i] = p_seg[i]/maxn
            #print('p seg', p_seg.shape)
        #print(CT.shape, pred.shape, gt.shape)
        if not config['public_test'] or config['public_test']=='prospective':
            with open(str(path)+'/report.txt', 'w') as fp:
                if reportID not in report_desc:
                    print('reportID not found', reportID)
                    continue
                '''
                for x in report_label[reportID]:
                    if x not in cnt:
                        cnt[x] = 0
                    cnt[x] += 1
                    fp.write(x+'\n')
                '''
                vague = False
                for x in report_desc[reportID]:
                    if '模糊' in x:
                        vague = True
                    fp.write(x+'\n')
                if vague and ignore_vague:
                    continue
        pass
        
        if not multitask:
            make_single_png(CT, [pred], gt, path, refined=refined)
        else: #multitask
            label = np.loadtxt(str(path)+'/label.txt')
            """####################################################
            计算分割DICE，IOU：
            """
            if ab_seg is not None: 
                temp = copy.deepcopy(label)
                temp[hand_label_i,3] = 0
                gt_label = np.max(temp, 0)
                print(path, gt_label)
                for a in range(4):
                    if not gt_label[a]:
                        continue
                    temp = np.bitwise_and(np.right_shift(ab_seg, a), 1) #[160,160,80]
                    if np.sum(temp)==0:
                        print('no ground truth seg', a)
                        continue
                    for t in range(10):
                        max_class = np.equal(np.argmax(pred[20:24], axis=0), a) #预测所有异常类型中a类的置信度最大的map
                        #乘max_class会稍微第一点
                        binary_pred = resize(np.uint8(np.greater(pred[20+a], 0.1*t)), temp.shape)
                        if t==1:
                            for i in range(80):
                                img = np.uint8(np.concatenate([temp[:,:,i], binary_pred[:,:,i]])*255)
                                print(img.shape)
                                cv2.imwrite(str(path)+'/seg%d_%d.png'%(a,i), img)
                        iou = IOU(binary_pred, temp)
                        dice = Dice(binary_pred, temp)
                        print('iou', a, iou, dice)
                        all_dice[(a,t)].append(dice)
                        all_iou[(a,t)].append(iou)
                
            if not dynamic_mask and save_image: 
                temp = np.zeros(gt.shape)
                seg = np.zeros(pred[0].shape)
                for i in organs: #全部应该是20
                    temp = np.maximum(temp, np.bitwise_and(np.right_shift(gt, i), 1))
                    if not soft_seg:
                        seg = np.maximum(seg, pred[i])
                    else:
                        seg = np.maximum(seg, p_seg[i])
                temp = resize(np.float32(temp), pred[20].shape)
                if n_abnormals==1:
                    make_single_png(CT, [pred[20]*seg] if soft_seg else [pred[20]*temp], gt, path, refined=refined, prefix='abnormal',
                                    seg = seg, abnormal = pred[20], organs = organs)
                else:
                    final_pred = []
                    abnormal = []
                    for a in range(n_abnormals):
                        if not soft_seg:
                            final_pred.append(temp*pred[20+a])
                        else:
                            final_pred.append(seg*pred[20+a])
                        abnormal.append(pred[20+a])
                    make_single_png(CT, final_pred, gt, path, refined=refined, 
                                prefix='abnormal', seg=seg, abnormal=abnormal, organs = organs)

            if dynamic_mask and save_image: #soft_seg需修改
                if len(preds)==0:
                    temp_preds = [pred]
                else:
                    temp_preds = preds
                output_images = []
                         
                for pred_ in temp_preds:
                    brain_mask = np.uint8(np.not_equal(np.min(CT), CT))
                    brain_mask = resize(brain_mask, pred_[0].shape, inter=cv2.INTER_NEAREST)
               
                    temp = np.zeros(pred_[0].shape)
                    for i in organs: 
                        temp = np.maximum(temp, p_seg[i] if soft_seg else pred_[i])
                    if n_abnormals==1:
                        make_single_png(CT, [temp*pred_[20]], gt, path, refined=refined, 
                                        prefix='abnormal', seg=temp, abnormal=pred_[20], organs = organs)
                    else:
                        final_pred = []
                        abnormal = []
                        for a in range(n_abnormals):
                            final_pred.append(temp*pred_[20+a])
                            abnormal.append(pred_[20+a]*brain_mask)
                        output_images.append(make_single_png(CT, final_pred, gt, path, refined=refined, 
                                                        prefix='abnormal', seg=temp, abnormal=abnormal, organs = organs,
                                                        save = False))
                for i in range(len(output_images[0])):
                    image = None
                    for j in range(len(temp_preds)):
                        image = output_images[j][i] if image is None else np.concatenate((image, output_images[j][i]), 0)
                    cv2.imwrite(str(path)+'/resultabnormal_%s.png'%str(i*4).zfill(2), image)
                    
            
            if (not config['public_test'] or config['public_test']=='prospective') and (np.max(label, 1))[7]: #7.2.2021因为在非手标数据上，有不明区域（手标数据筛去了不明区域），这里筛除
                continue
            if dynamic_mask:
                if not config['public_test'] and n_abnormals>1 and (label[hand_label_i,3]==0 or use_coarse_label):
                    label_cnt = label if label_cnt is None else label_cnt+label
                    label_cnt2 = np.max(label, 0) if label_cnt2 is None else label_cnt2+np.max(label, 0)
            
                for i in organs:
                    if n_abnormals==1 and label[12]==0:
                        if write_score:
                            with open(dir+'/scores_%d.txt'%i, 'a') as fp:
                                fp.write(str(i).zfill(2)+' '+str(label[i])+' '+str(np.max(pred[i]*pred[20]))+' '+path.name+'\n')
                        scores[(i,0)].append((label[i], np.max(pred[i]*pred[20])))
                    if n_abnormals>1 and (config['public_test'] or (label[hand_label_i,3]==0 or use_coarse_label)): #不是手工标注的不算
                        
                        max_score = 0 #本脑区上各种异常的最大预测值，表明此脑区有异常的概率
                        gt_label = 0
                        for a in range(n_abnormals):
                            o = (i if not flip else cids[i])
                            score = np.max(pred[o]*pred[20+a])
                            if use_sum_score:
                                score = np.sum(pred[o]*pred[20+a])
                            #top_value = top_values(CT, pred[i]*pred[20+a])
                            if soft_seg:
                                score = np.max(p_seg[o]*pred[20+a])
                            
                            if not config['public_test'] or config['public_test']=='prospective':
                                if write_score:
                                    with open(dir+'/scores_%d.txt'%i, 'a') as fp:
                                        fp.write(str(a)+' '+str(i).zfill(2)+' '+str(label[i,a])+' '+str(score)+' '+path.name+'\n')
                                scores[(i,a)].append((label[i,a], score))
                            else: #rsna或cq500
                                series = '_'.join(path.name.split('_')[1:])
                                if series not in score_cq500:
                                    score_cq500[series] = {}
                                score_cq500[series][(i,a)] = score
                            if n_abnormals==4 and (not config['public_test'] or config['public_test']=='prospective'):
                                if a==1:
                                    scores_d[(i,0)].append((max(scores[(i,0)][-1][0], scores[(i,1)][-1][0]), 
                                                            max(scores[(i,0)][-1][1], scores[(i,1)][-1][1])))
                                elif a==3:
                                    scores_d[(i,1)].append((max(scores[(i,2)][-1][0], scores[(i,3)][-1][0]), 
                                                            max(scores[(i,2)][-1][1], scores[(i,3)][-1][1])))
                                max_score = max(max_score, score)
                                gt_label = max(gt_label, label[i,a])
                        overall_scores[i].append((gt_label, max_score))
                if n_abnormals>1 and (config['public_test'] or label[hand_label_i,3]==0 or use_coarse_label): 
                    if not config['public_test'] or config['public_test']=='prospective':
                        brain_mask = np.uint8(np.not_equal(np.min(CT), CT))
                        brain_mask = resize(brain_mask, pred[0].shape, inter=cv2.INTER_NEAREST)
                        print(np.min(CT), np.max(CT))
                        for a in range(n_abnormals): #统计序列级别各异常类型预测准确率
                            if use_real_image_level: #真的是序列级别的检测效果，包括脑部所有区域，包括其他脑区
                                score = np.max(pred[20+a] * brain_mask)
                                temp = copy.deepcopy(label)
                                temp[hand_label_i,3] = 0 #5.31.2022。之前prospective把非手标的[12,3]都弄进scan level标注了，导致scan-level脑外低很低。
                                gt_label = np.max(temp, 0)[a]
                            else: #只把所有定义脑区当做全脑，忽略其他脑区
                                score = max([scores[(i,a)][-1][1] for i in organs])
                                gt_label = max([label[i,a] for i in organs])
                            scores_il[a].append((gt_label, score))
                            with open(dir+'/scores_il.txt', 'a') as fp:
                                fp.write(str(a)+' '+str(gt_label)+' '+str(score)+' '+path.name+'\n')
                    else: #rsna或cq500
                        series = '_'.join(path.name.split('_')[1:])
                        if series not in score_il_cq500:
                            score_il_cq500[series] = {}
                        for a in range(n_abnormals):
                            score_il_cq500[series][a] = np.max(pred[20+a])
            else: #no dynamic mask; n_abnormals需修改
                print('no dynamic mask')
                for i in organs:
                    if not(n_abnormals==1 and label[12]==0) and not(n_abnormals>1 and label[hand_label_i,3]==0 or use_coarse_label):
                        continue
                    for a in range(n_abnormals):
                        temp = np.bitwise_and(np.right_shift(gt, i), 1)
                        temp = resize(np.float32(temp), pred[20+a].shape)
                        if n_abnormals==1:
                            l = label[i]
                        else:
                            l = label[i,a]
                        score = np.max(pred[20+a]*temp)
                        if soft_seg:
                            score = np.max(p_seg[i]*pred[20+a])
                        if write_score:
                            with open(dir+'/scores_%d.txt'%i, 'a') as fp:
                                fp.write(str(a)+' '+str(i).zfill(2)+' '+str(l)+' '+str(score)+' '+path.name+'\n')
                        scores[(i,a)].append((l, score))
                if n_abnormals>1 and (config['public_test'] or label[hand_label_i,3]==0 or use_coarse_label):  #2023.3.6。为了在dynamic为false时也计算全图AUC，加了这个。不知道之前为什么用这么复杂的条件
                    if not config['public_test'] or config['public_test']=='prospective':
                        brain_mask = np.uint8(np.not_equal(np.min(CT), CT))
                        brain_mask = resize(brain_mask, pred[0].shape, inter=cv2.INTER_NEAREST)
                        print(np.min(CT), np.max(CT))
                        for a in range(n_abnormals): #统计序列级别各异常类型预测准确率
                            if use_real_image_level: #真的是序列级别的检测效果，包括脑部所有区域，包括其他脑区
                                score = np.max(pred[20+a] * brain_mask)
                                temp = copy.deepcopy(label)
                                temp[hand_label_i,3] = 0 #5.31.2022。之前prospective把非手标的[12,3]都弄进scan level标注了，导致scan-level脑外低很低。
                                gt_label = np.max(temp, 0)[a]
                            else: #只把所有定义脑区当做全脑，忽略其他脑区
                                score = max([scores[(i,a)][-1][1] for i in organs])
                                gt_label = max([label[i,a] for i in organs])
                            scores_il[a].append((gt_label, score))
                            with open(dir+'/scores_il.txt', 'a') as fp:
                                fp.write(str(a)+' '+str(gt_label)+' '+str(score)+' '+path.name+'\n')
                                
            #label = np.loadtxt(str(path)+'/predicted_label.txt')

            
            #print('label', label.shape, label.dtype)
            '''
            mask = None
            for i in organs:#range(20):
                if mask is None:
                    mask = pred[i]*label[i]
                else:
                    mask = np.maximum(mask, pred[i]*label[i])
            '''
            #make_single_png(CT, mask, gt, path, refined=refined, prefix='abnormal')
            
            '''
            label = np.loadtxt(str(path)+'/label.txt')
            predicted_label = np.loadtxt(str(path)+'/predicted_label.txt')
            for i in organs:
                cls_scores[i].append((label[i], predicted_label[i]))
            with open(dir+'/cls_scores.txt', 'a') as fp:
                for i in organs:
                    fp.write(str(label[i])+' '+str(predicted_label[i])+' '+path.name+'\n')
            
            label = [(label[i], parts[i]) for i in organs]
            predicted_label = [(predicted_label[i], parts[i]) for i in organs]
            label.sort(key = lambda x:x[0], reverse = True)
            predicted_label.sort(key = lambda x:x[0], reverse = True)
            with open(str(path)+'/result.txt', 'w') as fp:
                fp.write(str(label)+'\n')
                fp.write(str(predicted_label))
            '''
        #print(report_desc[fa[seriesID][0]])
    
    if 'cq500' in config['public_test']:
        labels = load_cq500_label('/home/liuaohan/pm-stroke/public/cq500/label.xlsx', naoshi=naoshi)
        auc_score = [[] for i in range(6)]
        left_organs = [0,2,4,8,10,15,17,18]
        right_organs = [1,3,5,9,11,16,17,19]
        #if naoshi:
        #    left_organs.append(13)
        #    right_organs.append(14)
        for series in score_cq500:
            seriesID = series.split('_')[0]
            score = score_cq500[series]
            #print(score)
            label = labels[seriesID]
            #print(label)
            if ignore_outside_organs:
                auc_score[0].append((label[0], max([score[(i,0)] for i in organs])))
            else:
                auc_score[0].append((label[0], score_il_cq500[series][0]))
            if cq500_use_low: #ignore_outside_organs需修改
                auc_score[1].append((label[1], max([max(score[(i,1)], score[(i,3)]) for i in organs])))
                auc_score[2].append((label[2] if not flip else label[3], max([max(score[(i,0)], score[(i,1)], score[(i,3)]) for i in left_organs])))
                auc_score[3].append((label[3] if not flip else label[2], max([max(score[(i,0)], score[(i,1)], score[(i,3)]) for i in right_organs])))
            else:
                if ignore_outside_organs:
                    auc_score[1].append((label[1], max([score[(i,1)] for i in organs])))
                else:
                    auc_score[1].append((label[1], score_il_cq500[series][1]))
                auc_score[2].append((label[2] if not flip else label[3], max([max(score[(i,0)], score[(i,1)]) for i in left_organs])))
                auc_score[3].append((label[3] if not flip else label[2], max([max(score[(i,0)], score[(i,1)]) for i in right_organs])))
            auc_score[4].append((max(label), max([max(score[(i,0)], score[(i,1)]) for i in organs])))
            auc_score[5].append((max(label), max([max([score[(i,x)] for x in range(4)]) for i in organs])))
            
        AUC_fp = open(dir+'/mAP_%d.txt'%config['version'], 'a')
        AUC_fp.write('%d  sum %d  side %d  use low %d  naoshi %d  ignore_outside_organs %d\n'%
                     (len(score_cq500), use_sum_score, use_side_mask, cq500_use_low, naoshi, ignore_outside_organs))
        for i in range(6):
            if write_score:
                with open(dir+'/cq500_%d_%s.txt'%(i, 'sum' if use_sum_score else 'max'), 'w') as fp:
                    tot = 0
                    for series in score_cq500:
                        fp.write(str(auc_score[i][tot][0])+' '+str(auc_score[i][tot][1])+' '+series+'\n')
                        tot += 1
            AUC_fp.write('%f '%AUC(auc_score[i]))
            print(AUC(auc_score[i]))
        AUC_fp.write('\n')
        AUC_fp.close()
        return
    if 'rsna' in config['public_test']:
        auc_score = [[] for i in range(10)]
        f = max
        if use_sum_score:
            f = sum
        for series in score_cq500:
            label = np.loadtxt(dir+'/rsna_'+series+'/label.txt')
            score = score_cq500[series]
            auc_score[0].append((np.max(label), f([score[(i,0)] for i in organs])))
            auc_score[1].append((label[1], f([score[(i,0)] for i in organs])))
            auc_score[2].append((max([label[i] for i in [0,2,3,4]]), f([score[(i,1)] for i in organs])))
            auc_score[3].append((np.max(label), f([max([score[(i,x)] for x in range(4)]) for i in organs])))
            auc_score[4].append((np.max(label), f([max([score[(i,x)] for x in range(2)]) for i in organs])))
            auc_score[5].append((np.max(label), f([score[(i,1)] for i in organs])))
            
        AUC_fp = open(dir+'/mAP_%d.txt'%config['version'], 'a')
        
        AUC_fp.write('use_sum_score %d\nduplicate '%use_sum_score)
        for i in range(6):
            with open(dir+'/score_%d.txt'%(i), 'w') as fp:
                for tot,series in enumerate(score_cq500):
                    fp.write(str(auc_score[i][tot][0])+' '+str(auc_score[i][tot][1])+' '+series+'\n')
            AUC_fp.write('%f '%AUC(auc_score[i]))
            print(AUC(auc_score[i]))
        AUC_fp.write('\nunique    ')
        for i in range(6):
            AUC_fp.write('%f '%AUC(list(set(auc_score[i]))))
            print(AUC(list(set(auc_score[i]))))
        AUC_fp.write('\n')
        AUC_fp.close()
        return
            
    #nc_scores = remove_symmetry_scores(scores) #not symmetric score
    
    for a in range(4):
        print('abnormality', a)
        for t in range(10):
            print('IOU ', t, len(all_iou[(a,t)]), np.mean(np.array(all_iou[(a,t)])))
        for t in range(10):
            print('dice ', t, len(all_dice[(a,t)]), np.mean(np.array(all_dice[(a,t)])))
            l, r, results = bootstrap(all_dice[(a,t)])
            print(l, r)
    with open(dir+'/mAP_%d.txt'%config['version'], 'a') as fp:
        fp.write(str(len(scores[(organs[0],0)]))+'\n')
        fp.write('soft segmentation '+str(soft_seg)+'\n')
        mAP = 0
        overall_mAP = 0 #有异常的mAP，不管异常类别
        macro_auc = {}
        macro_pos = {} #用来帮助计算加权macro_auc
        macro_pos_d = {}
        macro_auc_d = {}
        for a in range(n_abnormals):
            macro_auc[a] = []
            macro_pos[a] = []
            macro_auc_d[a] = []
            macro_pos_d[a] = []
        tb = pt.PrettyTable(border = False) #记录AUC
        tb2 = pt.PrettyTable(border = False) #用于记录mAP
        tb3 = pt.PrettyTable(border = False)
        for i in organs:
            for a in range(n_abnormals):
                n_pos = len(list(filter(lambda x:x[0], scores[(i,a)])))
                if n_pos==0:
                    continue
                ap = AP(scores[(i,a)])
                auc = AUC(scores[(i,a)])
                #nc_auc = AUC(nc_scores[(i,a)])
                if auc>=0: #auc!=NaN
                    macro_auc[a].append(auc)
                    macro_pos[a].append(n_pos)
                #print('AP_%d_%d'%(i,a), ap)
                #fp.write('AP_%d_%d'%(i,a)+' '+str(ap)+'\n')
                print('AUC_%d_%d'%(i,a), auc)
                tb.add_column('%d_%d'%(i,a), [round(auc, 4), n_pos])
                tb2.add_column('%d_%d'%(i,a), [round(ap, 4), n_pos])
                #tb3.add_column('%d_%d'%(i,a), [round(nc_auc, 4), len(list(filter(lambda x:x[0], nc_scores[(i,a)])))])
                
                #fp.write('AUC_%d_%d'%(i,a)+' '+str(auc)+' '+str(n_pos)+'\n')
                mAP += ap
            ap = AP(overall_scores[i])
            #print('AP_%d'%i, ap)
            #fp.write('AP_%d'%i+' '+str(ap)+'\n')
            overall_mAP += ap
            if n_abnormals==4:
                for a in range(2): #计算高低密度准确率
                    n_pos = len(list(filter(lambda x:x[0], scores_d[(i,a)])))
                    if n_pos==0:
                        continue
                    auc = AUC(scores_d[(i,a)])
                    if auc>=0: #auc!=NaN
                        macro_auc_d[a].append(auc)
                        macro_pos_d[a].append(n_pos)
                        
        fp.write(str(tb)+'\n')
        fp.write(str(tb3)+'\n')
        fp.write(str(tb2)+'\n')
        
        tb = pt.PrettyTable(border = False)
        for a in range(n_abnormals):
            temp = np.mean(np.asarray(macro_auc[a]))
            tb.add_column('macro AUC%d'%a, [np.round(temp, 4)])
        for a in range(n_abnormals):
            for i in organs:
                micro_scores[a].extend(scores[(i,a)])
            auc = AUC(micro_scores[a])
            tb.add_column('micro AUC%d'%a, [round(auc, 4)])
        
        for a in range(n_abnormals):
            weighted = np.average(np.asarray(macro_auc[a]), weights=np.asarray(macro_pos[a]))
            tb.add_column('w macro%d'%a, [np.round(weighted, 4)])
        if n_abnormals==4:
            for a in range(2):
                micro_scores[a] = []
                for i in organs:
                    micro_scores[a].extend(scores_d[(i,a)])
                auc = AUC(micro_scores[a])
                tb.add_column('micro_d%d'%a, [round(auc, 4)])
            #for a in range(2):
            #    weighted = np.average(np.asarray(macro_auc_d[a]), weights=np.asarray(macro_pos_d[a]))
             #   tb.add_column('w macro_d%d'%a, [np.round(weighted, 4)])
        for a in range(n_abnormals):
            auc = AUC(scores_il[a])
            tb.add_column('image AUC%d'%a, [round(auc, 4)])
        fp.write(str(tb)+'\n')
        print(tb)
        
        mAP /= len(organs)*n_abnormals
        print('mAP', mAP)
        fp.write('mAP'+' '+str(mAP)+'\n')
        overall_mAP /= len(organs)
        print('overall mAP', overall_mAP)
        fp.write('overall mAP'+' '+str(overall_mAP)+'\n')
        """
        mAP = 0
        for i in organs:
            for a in range(n_abnormals):
                ap = AP(cls_scores[(i,a)])
                print('cls AP_%d_%d'%(i,a), ap)
                fp.write('cls AP_%d_%d'%(i,a)+' '+str(ap)+'\n')
                mAP += ap
        mAP /= len(organs)*n_abnormals
        print('cls mAP', mAP)
        fp.write('cls mAP'+' '+str(mAP)+'\n')
        """
        fp.write('\n')
    print(cnt)
    print(len(scores[(organs[0],0)]))
    #print(label_cnt/len(scores[(organs[0],0)]))
    #print(label_cnt2/len(scores[(organs[0],0)]))
    pass

def bootstrap(scores, n = 1000, e = None, seed=10):
    if e is not None and e<=0:
        return 0,0
    scores = copy.deepcopy(scores)
    l = len(scores)
    random.seed(seed)
    while True:
        results = []
        for i in tqdm(range(n)):
            s = []
            for j in range(l):
                x = random.randint(0,l-1)
                s.append(scores[x])
            x = sum(s)/len(s)
            results.append(x)

        results = sorted(results)
        L,R = results[int(0.025*n)], results[n-1-int(0.025*n)]
        if e is None or (L<=e and R>=e):
            return L, R, results

def evaluate_classification(dir, report_desc, report_label, report, fa, paths = None, filt = None, refined = False, multitask=False,
                            dynamic_mask = False, flip = False):
    if paths is None:
        all_paths = list(Path(dir).glob('*'))
    else:
        all_paths = [Path(x) for x in paths]
    organs = np.array([0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19])
    hand_label_i = 12
    n_abnormals = 4
    evaluate_seg = True
    
    scores = {(i,a):[] for i in organs for a in range(n_abnormals)}
    micro_scores = {a:[] for a in range(n_abnormals)}
    scores_il = {a:[] for a in range(n_abnormals)}
    all_dice = {(a,t):[] for a in range(4) for t in range(0, 10)}
    all_iou = {(a,t):[] for a in range(4) for t in range(0, 10)}
    
    '''
    for i in organs:
        for a in range(n_abnormals):
            scores[(i,a)] = []
    for a in range(n_abnormals):
        micro_scores[a] = []
        scores_il[a] = []
    '''
    for path in tqdm(all_paths):
        print(path)
        #print(path)
        #path_i += 1
        if not path.is_dir():
            print('not path', path)
            continue
        #seriesID = path.name.split('_')[-1]
        #reportID = path.name.split('_')[-2]
        label = np.loadtxt(str(path)+'/label.txt')
        pred = np.loadtxt(str(path)+'/pred.txt')
        if not evaluate_seg:
            for i in organs:
                for a in range(n_abnormals):
                    with open(dir+'/scores_%d.txt'%i, 'a') as fp:
                        fp.write(str(a)+' '+str(i).zfill(2)+' '+str(label[i,a])+' '+str(pred[i,a])+' '+path.name+'\n')
                    scores[(i,a)].append((label[i,a], pred[i,a]))
                   
            image_label = np.max(label[organs], 0)
            for a in range(n_abnormals):
                scores_il[a].append((image_label[a], pred[12,a]))
                with open(dir+'/scores_il.txt', 'a') as fp:
                    fp.write(str(a)+' '+str(image_label[a])+' '+str(pred[12,a])+' '+path.name+'\n')
        
        if evaluate_seg:
            if not Path(str(path)+'/ab_seg.nii.gz').exists():
                continue
            ab_seg = nib.load(str(path)+'/ab_seg.nii.gz').get_data()[0] #[160,160,80]
            pred = nib.load(str(path)+'/GC.nii.gz').get_data()
            print(pred.shape, np.max(pred), np.min(pred))
            pred = np.stack([pred[i]/np.max(pred[i]) for i in range(4)], axis=0)
            
            temp = copy.deepcopy(label)
            temp[hand_label_i,3] = 0
            gt_label = np.max(temp, 0)
            print(path, gt_label)
            for a in range(4):
                if not gt_label[a]:
                    continue
                temp = np.bitwise_and(np.right_shift(ab_seg, a), 1) #[160,160,80]
                if np.sum(temp)==0:
                    print('no ground truth seg', a)
                    continue
                for t in range(10):
                    binary_pred = resize(np.uint8(np.greater(pred[a], 0.1*t)), temp.shape)
                    iou = IOU(binary_pred, temp)
                    dice = Dice(binary_pred, temp)
                    print('iou', a, iou, dice)
                    all_dice[(a,t)].append(dice)
                    all_iou[(a,t)].append(iou)
                            

            loc = nib.load(str(path)+'/GC.nii.gz').get_data()
            print(loc.shape, np.max(loc), np.min(loc))
            CT = nib.load(str(path)+'/data_%s.nii.gz'%config["all_modalities"][0]).get_data()
            CT = np.uint8((CT-np.min(CT))/(np.max(CT)-np.min(CT))*255)
            
            loc = np.stack([resize(loc[i], CT.shape, cv2.INTER_LINEAR)/np.max(loc[i]) for i in range(4)], axis=0)
            print(loc.shape)
            
            for i in range(0,80,4):
                img = np.uint8(np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2))
                result = np.uint8(np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2))
                for j in range(4):
                    heat_img = cv2.applyColorMap(np.uint8(loc[j,:,:,i]*255), cv2.COLORMAP_JET)
                    add = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
                    result = np.concatenate((result, add), 1)
                cv2.imwrite(str(path)+'/gc_%s.png'%(str(i).zfill(2)), result)

    for a in range(4):
        print('abnormality', a)
        for t in range(10):
            print('IOU ', t, len(all_iou[(a,t)]), np.mean(np.array(all_iou[(a,t)])))
        for t in range(10):
            print('dice ', t, len(all_dice[(a,t)]), np.mean(np.array(all_dice[(a,t)])))
            l, r, results = bootstrap(all_dice[(a,t)])
            print(l, r)
    with open(dir+'/mAP_%d.txt'%config['version'], 'a') as fp:
        fp.write(str(len(scores[(organs[0],0)]))+'\n')

        mAP = 0
        overall_mAP = 0 #有异常的mAP，不管异常类别
        macro_auc = {a:[] for a in range(n_abnormals)}
        macro_pos = {a:[] for a in range(n_abnormals)} #用来帮助计算加权macro_auc

        tb = pt.PrettyTable(border = False) #记录AUC
        tb2 = pt.PrettyTable(border = False) #用于记录mAP
        tb3 = pt.PrettyTable(border = False)
        for i in organs:
            for a in range(n_abnormals):
                n_pos = len(list(filter(lambda x:x[0], scores[(i,a)])))
                if n_pos==0:
                    continue
                ap = AP(scores[(i,a)])
                auc = AUC(scores[(i,a)])
                #nc_auc = AUC(nc_scores[(i,a)])
                if auc>=0: #auc!=NaN
                    macro_auc[a].append(auc)
                    macro_pos[a].append(n_pos)
                #print('AP_%d_%d'%(i,a), ap)
                #fp.write('AP_%d_%d'%(i,a)+' '+str(ap)+'\n')
                print('AUC_%d_%d'%(i,a), auc)
                tb.add_column('%d_%d'%(i,a), [round(auc, 4), n_pos])
                tb2.add_column('%d_%d'%(i,a), [round(ap, 4), n_pos])
                #tb3.add_column('%d_%d'%(i,a), [round(nc_auc, 4), len(list(filter(lambda x:x[0], nc_scores[(i,a)])))])
                
                #fp.write('AUC_%d_%d'%(i,a)+' '+str(auc)+' '+str(n_pos)+'\n')
                mAP += ap
                        
        fp.write(str(tb)+'\n')
        fp.write(str(tb3)+'\n')
        fp.write(str(tb2)+'\n')
        
        tb = pt.PrettyTable(border = False)
        for a in range(n_abnormals):
            temp = np.mean(np.asarray(macro_auc[a]))
            tb.add_column('macro AUC%d'%a, [np.round(temp, 4)])
        for a in range(n_abnormals):
            for i in organs:
                micro_scores[a].extend(scores[(i,a)])
            auc = AUC(micro_scores[a])
            tb.add_column('micro AUC%d'%a, [round(auc, 4)])
        
        for a in range(n_abnormals):
            weighted = np.average(np.asarray(macro_auc[a]), weights=np.asarray(macro_pos[a]))
            tb.add_column('w macro%d'%a, [np.round(weighted, 4)])
        for a in range(n_abnormals):
            auc = AUC(scores_il[a])
            tb.add_column('image AUC%d'%a, [round(auc, 4)])
        fp.write(str(tb)+'\n')
        print(tb)
        fp.write('\n')

def make_png_cq500(dir, paths = None, filt = None):
    if paths is None:
        all_paths = list(Path(dir).glob('*'))
    else:
        all_paths = [Path(x) for x in paths]
    print(len(all_paths))
    n_labels = config['n_labels']
    scores = {i:[] for i in range(n_labels+1)}
    for path in all_paths:
        label = np.loadtxt(str(path)+'/label.txt')
        pred = np.loadtxt(str(path)+'/prediction.txt')
        for i in range(n_labels):
            scores[i].append((float(label[i]), float(pred[i])))
        scores[n_labels].append((np.max(label), np.max(pred)))
        for i in range(n_labels+1):
            with open(dir+'/scores_%d.txt'%(i), 'w') as fp:
                fp.write(str(scores[i][-1][0])+' '+str(scores[i][-1][1])+' '+Path(path).name[6:]+'\n')
    tb = pt.PrettyTable(border = False)
    result = np.zeros(n_labels+1)
    for i in range(n_labels+1):
        auc = AUC(scores[i])
        result[i] = auc
        tb.add_column(str(i), [auc])
    print(tb)
    with open(dir+'/auc.txt', 'w') as fp:
        fp.write(str(tb))
    np.savetxt(dir+'/result.txt', result)
    with open(dir+'/scores.json', 'w') as fp:
        json.dump(scores, fp)

def preset():
    report = {}
    fa = {}
    with open('all_report.json', 'r') as fp:
        report = json.load(fp)
    for reportID in report:
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                if series['seriesID'] not in fa:
                    fa[series['seriesID']] = []
                fa[series['seriesID']].append((reportID, studyID))
    print('report loaded', len(report))
    
    if config['public_test']=='prospective':
        with open('/home/liuaohan/pm-stroke/project/EHR/report_desc_p.json', 'r') as fp:
            report_desc = json.load(fp) 
            report_desc = {x:report_desc[x][0] for x in report_desc} #原来是x:[[desc sentences], diagnose]
    else:
        with open('report_desc.json', 'r') as fp:
            report_desc = json.load(fp) #{reportID: description sentences}
    print('report desc loaded', len(report_desc))
    with open('report_labels.json', 'r') as fp:
        report_label = json.load(fp) #{reportID: labels}
    print('report labels loaded', len(report_label))
    return report, fa, report_desc, report_label


def training_valid_cases():
    data_file = tables.open_file(config["data_file"], "r")
    training_set = []
    training_indices = pickle_load(config["training_file"])
    for index in training_indices:
        ID = data_file.root.subject_ids[index].decode('utf-8')
        training_set.append(resolve_subject_id(ID))
    valid_set = []
    if 'test_file' in config:
        validation_indices = pickle_load(config["test_file"])
    else:
        validation_indices = pickle_load(config["validation_file"])
    for index in validation_indices:
        ID = data_file.root.subject_ids[index].decode('utf-8')
        valid_set.append(resolve_subject_id(ID))
    data_file.close()
    return training_set, valid_set


def remove_bad_paths(paths):
    """
    paths
    """
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

    paths = [Path(x) for x in paths]
    ret = []
    ret_bad = []
    ret_long = []
    base_dir = '/home/liuaohan/pm-stroke/aligned_registration'
    bad = {}
    for i in range(34):
        with open(base_dir+'/%s/registration_check_result.json'%str(i).zfill(2), 'r') as fp:
            registration_check = json.load(fp)
        for x in registration_check:
            if registration_check[x]>=3:
                bad[standard_path(x)] = True
    cnt = {}
    for path in paths:
        info = path.name.split('_')
        #print(info, info[-2])
        report_dir = '%s/%s/%s'%(base_dir, info[0], info[1])
        #series_paths = list(Path(report_dir).glob('%s/*/*'%report_dir))
        series_paths = list(glob.glob('%s/*/*'%report_dir))
        series_path = list(filter(lambda x:Path(x).name==info[2], series_paths))[0]
        if Path(series_path+'/series_link.txt').exists():
            with open(series_path+'/series_link.txt', 'r') as fp:
                series_path = fp.readline().strip()
        if standard_path(series_path) in bad:
            print('bad', series_path)
            cnt['bad'] = cnt.get('bad', 0)+1
            ret_bad.append(str(path))
            continue
        image_paths = list(Path(series_path).glob('*brain*.png'))
        if len(image_paths)<10:
            cnt['short'] = cnt.get('short', 0)+1
            print('series too short', len(image_paths), series_path)
            continue
        if len(image_paths)>40:
            cnt['long'] = cnt.get('long', 0)+1
            print('series too long', len(image_paths), series_path)
            ret_long.append(str(path))
            continue
        ret.append(str(path))
    print(cnt)
    print('remaining', len(ret))
    return [ret, ret_bad, ret_long]

def analyse_auc(scores):
    """
    scores:[(label,pred,ID)]
    """
    scores.sort(key = lambda x:x[1])
    np = int(sum([x[0] for x in scores])) #num positive
    nn = len(scores)-np
    error = {}
    nowp = 0
    nown = 0
    for i in range(len(scores)):
        if scores[i][0]:
            error[scores[i][2]] = nn-nown #有多少label为0的，预测值比它高
            nowp += 1
        else:
            error[scores[i][2]] = nowp #有多少label为1的，预测值比它低
            nown += 1
    return error

def compare_downstream(ori, ds, valid_file):
    """
    ori:原始CT模型预测的文件夹，如multitask76。即作为downstream输入的
    ds: 下游任务预测的文件夹
    """
    print(valid_file)
    validation_indices = pickle_load(valid_file)
    data_file = tables.open_file(config["data_file"], "r")
    path1 = []
    path2 = []
    ori_pred = [{} for i in range(4)]
    ds_pred = [{} for i in range(4)]
    for i in range(4):
        with open(ori+'/cq500_%d_max.txt'%i, 'r') as fp:
            temp = fp.readlines()
            for line in temp:
                s = line.strip().split(' ')
                ID = ' '.join(s[2:])
                ori_pred[i][ID] = (int(s[0]), float(s[1]), ID)
        with open(ds+'/cq500_%d.txt'%i, 'r') as fp:
            temp = fp.readlines()
            for line in temp:
                s = line.strip().split(' ')
                ID = ' '.join(s[2:])
                ds_pred[i][ID] = (float(s[0]), float(s[1]), ID)
    scores1 = [[] for i in range(4)]
    scores2 = [[] for i in range(4)]
    for index in tqdm(validation_indices):
        if 'subject_ids' in data_file.root:
            ID = data_file.root.subject_ids[index].decode('utf-8')
            ID = Path(ID).name[6:]
            for i in range(4):
                scores1[i].append(ori_pred[i][ID])
                scores2[i].append(ds_pred[i][ID])
                #if abs(ori_pred[i][ID][1]-ds_pred[i][ID][1]-0.3)>0.5:
                #    print(i, ori_pred[i][ID][0], round(ori_pred[i][ID][1], 3), round(ds_pred[i][ID][1],3), ID)
    print('ori prediction auc')
    for i in range(4):
        auc = AUC(scores1[i])
        print(round(auc, 8), end=' ')
    print('downstream prediction auc')
    for i in range(4):
        auc = AUC(scores2[i])
        print(round(auc, 8), end=' ')
    print()
    for i in range(4):
        error1 = analyse_auc(scores1[i])
        error2 = analyse_auc(scores2[i])
        print(i, sum([error1[x] for x in error1]), sum([error2[x] for x in error2]))
        for x in scores1[i]:
            if abs(error1[x[2]]-error2[x[2]])>=5:
                print(x[0], error1[x[2]], error2[x[2]], x[2])
        error_diff = [(error1[x[2]]-error2[x[2]], x[0]) for x in scores1[i]]
        error_diff.sort(key = lambda x:x[0])
        print(np.transpose(np.array(error_diff)))

def main():
    training_set, valid_set = training_valid_cases()
    training_set = [] #
    #with open('report_patient.json', 'r') as fp:
    #    report_patient = json.load(fp) #用来删掉测试集中在训练集出现过的病人的序列
    report_patient = {}

    multitask = config.get('multitask', False)
    dynamic_mask = config.get('dynamic_mask', False)
    organs = config.get('organs', None)
    truth_file = config.get('truth_file', None)
    label_file = config.get('label_file', None)
    print('public_test', config['public_test'])
    if 'test_label_file' in config:
        label_file = config['test_label_file']
    if config.get('public_test', False):
        label_file = None
    
    if 'test_file' in config:
        valid_file = config['test_file']
    else:
        valid_file = config["validation_file"]
    
    
    print(valid_file)
    if valid_file[-1]!='l':
        valid_file += '_-1.pkl'
    print(valid_file)
    
    
    if 'test_dir' in config:
        prediction_dir = os.path.abspath(config['test_dir'])
        print('use test set')
    else:
        prediction_dir = os.path.abspath(config['valid_dir'])
        print('use valid set')
    flip = False
    if flip:
        prediction_dir += '_flip'
    #base_dir = '/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/prediction/'
    #dirs = ['test_multitask53/']
    
    series = ['18_28375969.0_1.2.840.113619.2.289.3.296546961.849.1503839602.372.4',
              '19_19372860.0_1.2.840.113619.2.289.3.296546961.433.1410232318.194',
              '18_30373072.0_1.2.840.113619.2.289.3.296546961.847.1523145110.154.4',
              '19_31790721.0_1.2.840.113619.2.289.3.296546961.132.1534293976.735.4',
              '08_20266669.0_1.2.840.113619.2.289.3.296546961.469.1421031069.800',
              '02_28776928.0_1.3.12.2.1107.5.1.4.32219.30000017101601364356200000000',
              '03_30687919.0_1.3.12.2.1107.5.1.4.32219.30000018050314061196800054831',
              '04_18212598.0_1.3.12.2.1107.5.1.4.32219.30000014041606084537500016855',
              '04_18929612.0_1.3.12.2.1107.5.1.4.32219.30000014071400250309300018505',
              '04_19279722.0_1.3.12.2.1107.5.1.4.43501.30000014082623560560900009408',
              '05_19816733.0_1.2.840.113619.2.289.3.296546961.399.1415144326.132',
              '05_20496183.0_1.2.840.113619.2.289.3.296546961.227.1423712732.191',
              '05_24937312.0_1.2.156.112605.161339402051.20160803070120.3.10124.5',
              '05_25265715.0_1.3.12.2.1107.5.1.4.32219.30000016090814283835900000441',
              '05_25617795.0_1.2.840.113619.2.289.3.296546961.512.1476867427.240',
              '05_26560135.0_1.2.840.113619.2.289.3.296546961.798.1487225860.750',
              '05_26618191.0_1.2.840.113619.2.289.3.296546961.113.1487835111.94',
              '05_26958769.0_1.2.156.112605.14038005400544.20170330061315.3.5904.5',
              '06_27586941.0_1.2.156.112605.14038011836937.20170605024318.3.5932.5',
              '06_28198649.0_1.3.12.2.1107.5.1.4.32219.30000017080803263712500034701',
              '06_28542841.0_1.3.12.2.1107.5.1.4.32219.30000017091401304628100007401',
              '10_19508032.0_1.3.12.2.1107.5.1.4.43501.30000014092615111420300000140',
              '12_26327427.0_1.3.12.2.1107.5.1.4.32219.30000017011308042951500028864',
              '12_27020469.0_1.2.156.112605.14038011836937.20170413004020.3.6216.5',
              '12_27046687.0_1.2.156.112605.14038011836937.20170411000007.3.6216.5',
              
              ]
    series = ['14_30554318.0_1.2.156.112605.14038011815645.20180423081033.3.6220.6',
              '28_34237073.0_1.2.840.113619.2.289.3.296546961.868.1555399802.411.4',
              '00_14020163.0_1.3.12.2.1107.5.1.4.32219.30000012082723510976500000762']
    paths = []
    #for s in series:
    #    paths.append(prediction_dir+'/'+s)
    #for i in range(10):
    #    compare_downstream('prediction/test_cq500_82', 'prediction/test_cq500_downstream261_%d'%i, 'cq500_test_f10_0_%d.pkl'%i)
    #return
    
    paths = run_validation_cases(validation_keys_file=valid_file,
                         model_file=config["model_file"],
                         #model_file=config["model_file"][:-3]+'_pretrained.h5',
                         training_modalities=config.get("training_modalities", None),
                         labels=config.get("labels", None),
                         hdf5_file=config["data_file"],	
                         output_label_map=False,
                         output_dir=prediction_dir,
                         #num=slice(0,1000,10), #1300
                         #num=slice(0,10),
                         paths = paths,
                         refined = True,
                         
                         multitask = multitask,
                         multiinput = organs,
                         truth_file = truth_file,
                         label_file = label_file,
                         run = False, 
                         filt = training_set,
                         #filt = [], #
                         load_loss_weight = config.get('load_loss_weight', False),
                         report_patient = report_patient,
                         full_version = config.get('full_version', None),
                         flip = flip,
                         downstream = downstream,
                         rsna = (False if config['public_test'] is None else ('rsna' in config['public_test'])),
                         save_truth_file = (config['public_test']!='prospective'),
                         ab_seg_file = config.get('ab_seg_file', None))
    
    '''
    prediction_dir = os.path.abspath(config['training_dir'])
    paths = run_validation_cases(validation_keys_file=config["training_file"],
                         model_file=config["model_file"],
                         training_modalities=config.get("training_modalities", None),
                         labels=config.get("labels", None),
                         hdf5_file=config["data_file"],
                         output_label_map=False,
                         output_dir=prediction_dir,
                         num=slice(0,100),
                         
                         refined = True,
                         multitask = multitask,
                         multiinput = organs,
                         truth_file = truth_file,
                         label_file = label_file,
                         run = False,
                         load_loss_weight = config.get('load_loss_weight', False),
                         full_version = config.get('full_version', None),
                         downstream = downstream)
    '''
    
    print('paths', len(paths), len(set(paths)))

    paths.sort()
    #if config.get('public_test', False):
    if config.get('public_test', '')=='cq500':
        labels = load_cq500_label('/home/liuaohan/pm-stroke/public/cq500/label.xlsx', naoshi=True)
        for path in paths:
            seriesID = Path(path).name.split('_')[1]
            np.savetxt(path+'/real_label.txt', np.array(labels[seriesID]))

    #paths_list = remove_bad_paths(paths)
    #for paths in paths_list:
    if config.get('public_test', '')=='cq500_3' or config.get('public_test', '')=='prospective':
        flip = True
    if config.get('classify_only', False):
        report, fa, report_desc, report_label = preset()
        evaluate_classification(prediction_dir, report_desc, report_label, report, fa, paths = paths, filt = training_set, refined=True, multitask=multitask,
                 dynamic_mask = dynamic_mask, flip = flip)
    # elif not downstream:
    #     report, fa, report_desc, report_label = preset()
    #     make_png(prediction_dir, report_desc, report_label, report, fa, paths = paths, filt = training_set, refined=True, multitask=multitask,
    #              dynamic_mask = dynamic_mask, flip = flip)
    else:
        make_png_cq500(prediction_dir, paths = paths, filt = None)
    #make_png(prediction_dir, report_desc, report_label, report, fa, paths=paths, filt = None, refined=True, multitask=multitask,
    #       dynamic_mask = dynamic_mask)

def load(file, report, filt=False, now_id=None):
    """
    从原始xls文件中读取报告，生成{reportID:[desc1, desc2..]}的dict
    以下参数用于NLP大赛数据生成
    如果filt，则去除所有非头部CT平扫的报告
    如果now_id不为None，则生成的{reportID:(ID, [desc1, desc2..])}
    其中ID从now_id开始计数，代表此描述在xls文件中的行号
    之所以记录xls文件的行号（2021.1.15），是因为原始xls文件是按病的类型顺序排列的，
    我的CT模型只用了xls文件中能找到匹配dicom的部分数据，而我想标注的也是这些能找到对应dicom的描述。
    我不想标注一堆没有对应dicom的描述这样最后我没法直接利用它们。
    这样就出现一个问题，原始xls文件中的描述和我找到匹配的描述是同一分布吗？
    分布不同可能的原因是病种不同，而病种又和行号相关。
    """
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
        if now_id is not None:
            res = (now_id, res)
            now_id += 1
        report[Id] = (res, col11[i]) #7.5.2021为了给宇巍生成数据的疾病描述
    pass

def evaluate_prediction_result():
    def getage(seriesID):
        age = series_info[seriesID]['pAge']
        if 'Y' in age:
            return int(age[:3])
        return 0
    def get_label(series_path):
        return np.loadtxt(prediction_dir+'/'+series_path+'/label.txt')
    def low_density_under(record, threshold, organs):
        label = series_labels[record[4].strip()]
        #return np.sum(label, 0)[2]<=threshold or np.sum(label)-np.sum(label, 0)[2]>=3
        if np.sum(label, 0)[2]<=threshold:
            return True
    def asym_above(record, threshold, organs):
        label = series_labels[record[4].strip()]
        cid = [1,0,3,2,5,4,6,7,9,8,11,10,12,14,13,16,15,17,19,18]
        asym = 0
        for i in organs:
            if i<cid[i] and label[i,2]!=label[cid[i],2]:
                asym += 1
        return asym>=threshold
    def no_diagnose(record, keys):
        reportID = record[4].split('_')[1]
        for s in keys:
            if s in report_desc[reportID][1]:
                return False
        return True
    def get_scores(organs, n_abnormals, samples, f = lambda x:True):
        scores = {(i,j):[] for i,j in itertools.product(range(20), range(n_abnormals))}
        for organ in organs:
            with open(prediction_dir+'/scores_%d.txt'%organ, 'r') as fp:
                temp = fp.readlines()
            records = {} #因为scores txt里可能重复好几次，这一步去重
            for record in temp:
                record = record.split(' ')
                record[0] = int(record[0]) #type
                record[2] = int(float(record[2])) #gt
                record[3] = float(record[3]) #predict
                record[4] = record[4].strip()
                records[(record[4], record[0])] = record
            for x in tqdm(records):
                if records[x][4].strip() in samples and f(records[x]):
                    scores[(organ, records[x][0])].append((records[x][2], records[x][3]))
        return scores
    def get_crowded_scores(scores):
        crowded_scores = {(i,j):[] for i,j in itertools.product(range(20), range(n_abnormals))}
        for organ in organs:
            for i in range(len(scores[organs[0], 0])):
                flag = False
                for a in range(n_abnormals):
                    if scores[(organ,a)][i][0]:
                        flag = True
                if flag:
                    for a in range(n_abnormals):
                        crowded_scores[(organ, a)].append(scores[(organ, a)][i])
        return crowded_scores
    def get_samples():
        data = tables.open_file(config['data_file'], "r")
        indices = pickle_load(config['test_file'])
        print(len(indices))
        ret = []
        for index in tqdm(indices):
            series_path = data.root.subject_ids[index].decode('utf-8')
            reportID = Path(series_path).parent.parent.name
            fold = Path(series_path).parent.parent.parent.name
            seriesID = Path(series_path).name
            ret.append(fold+'_'+reportID+'_'+seriesID)
        print(ret[0])
        return ret
    
    if 'test_dir' in config:
        prediction_dir = os.path.abspath(config['test_dir'])
        print('use test set')
    else:
        prediction_dir = os.path.abspath(config['valid_dir'])
        print('use valid set')
    
    report_desc = {}
    with open('report_desc2.json', 'r') as fp:
        report_desc = json.load(fp)
    #load('CTHead301_1.xlsx', report_desc)
    #load('CTHead301_3.xlsx', report_desc)
    #with open('report_desc2.json', 'w') as fp:
    #    json.dump(report_desc, fp)
    
    
    with open('all_report.json', 'r') as fp:
        report = json.load(fp)
    series_labels = {}
    series_paths = Path(prediction_dir).glob('*')
    for series_path in tqdm(series_paths):
        if (series_path/'label.txt').exists():
            series_labels[series_path.name] = np.loadtxt(str(series_path)+'/label.txt')
    
    series_info = {}
    for reportID in report:
        for studyID in report[reportID]:
            for series in report[reportID][studyID]:
                series_info[series['seriesID']] = series
    
    
    
    samples = get_samples()
    organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19]
    n_abnormals = 4
    #scores = get_scores(organs, n_abnormals, samples,lambda x:int(x[4].split('_')[0])<9 or int(x[4].split('_')[0])>13)
    #scores = get_scores(organs, n_abnormals, samples,lambda x:getage(x[4].split('_')[2].strip())<=60)
    #scores = get_scores(organs, n_abnormals, samples,lambda x:getage(x[4].split('_')[2].strip())>60)
    #scores = get_scores(organs, n_abnormals, samples) #1694  0.9345    0.9339    0.8586    0.9436 
    #scores = get_scores(organs, n_abnormals, samples,lambda x:int(x[4].split('_')[0])<9 or int(x[4].split('_')[0])>14 or getage(x[4].split('_')[2].strip())>60)#8702
    #scores = get_scores(organs, n_abnormals, samples,lambda x:int(x[4].split('_')[0])<9 or int(x[4].split('_')[0])>14 or getage(x[4].split('_')[2].strip())<60)#8863
    #scores = get_scores(organs, n_abnormals, samples,lambda x:int(x[4].split('_')[0])<8 or int(x[4].split('_')[0])>14) # 0.9417    0.9274    0.9086    0.9406 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:int(x[4].split('_')[0])<8 or int(x[4].split('_')[0])>14 or low_density_under(x, 1, organs)) #2:8865 1:9004 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:int(x[4].split('_')[0])>=8 and int(x[4].split('_')[0])<=14) # 0.9526    0.9708    0.6634    0.9678 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:int(x[4].split('_')[0])>=8 and int(x[4].split('_')[0])<=14 and low_density_under(x, 6, organs))
    #<=2  182  1.0      0.9665    0.6371    0.9752 
    #<=3  257  1.0      0.9676    0.6413    0.9582 
    #<=4  284 0.9785    0.9738    0.6265    0.9617 
    #<=5  367 0.9433    0.9733    0.6356    0.9656 
    #<=6 381  0.9459    0.9729    0.6351    0.9657 
    # scores = get_crowded_scores(scores)#  0.8756    0.8499    0.6828    0.8698 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:int(x[4].split('_')[0])>=8 and int(x[4].split('_')[0])<=14 and asym_above(x, 1, organs))
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:int(x[4].split('_')[0])<8 or int(x[4].split('_')[0])>14 or not low_density_under(x, 2, organs))
    #<=4 1553  0.9344    0.9346    0.8737    0.9403 
    #<=3 1526  0.9347    0.9287    0.8802    0.9377 
    #<=2 1451  0.9336    0.927     0.889     0.9379  
    #<=1 1347  0.932     0.9212    0.9014    0.9329  
    #>4 1410  0.9305    0.9202    0.8886    0.9387 
    #>3 1437  0.9302    0.928     0.8827    0.9417 
    #>2 1512  0.9315    0.9296    0.8732    0.9411 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['脑梗','缺血'])) #927 0.9316    0.9137    0.9145    0.9413 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['脑梗','缺血']) or not low_density_under(x, 4, organs))
    #以下范围均指保存的数据，而非筛除的数据
    #<=3 1304  0.9399    0.929     0.8681    0.9475 
    #<=2  1169 0.9388    0.9251    0.872     0.9468 
    #<=1 1031  0.9358    0.9157    0.8922    0.9394 
    #>3 1317   0.9283    0.925     0.8978    0.9383 
    #>4 1261   0.9292    0.9156    0.9099    0.9347 
    
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['脑梗','缺血']) or not no_diagnose(x, ['老年'])) #缺血脑梗只保留老年脑 1215 0.9287    0.917     0.9166    0.939  
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['脑梗','缺血']) or no_diagnose(x, ['老年'])) #缺血脑梗只保留年轻脑1406  0.9386    0.9341    0.8589    0.9457 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血']) and no_diagnose(x, ['老年'])) #只有年轻的缺血脑梗479  0.9498    0.9702    0.6933    0.9531 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血', '软化']) and no_diagnose(x, ['老年'])) #只有年轻的缺血脑梗软化513   0.9486    0.9685    0.7017    0.9482 
    
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血']) and not no_diagnose(x, ['老年'])) #只有老年的缺血脑梗288   0.9083    0.9187    0.6933    0.9282 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血'])) #只有缺血脑梗767    0.9296    0.9601    0.7031    0.9458
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血','软化'])) #只有缺血脑梗软化808    0.9305    0.9574    0.7087    0.9405 
    
    
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗','缺血'])) #767   0.9296    0.9601    0.7031    0.9458 
    
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['脑梗','缺血']) or asym_above(x, 1, organs)) 
    #>=2 987  0.9302    0.9161    0.915     0.9393 
    #>=1 1159 0.9349    0.9286    0.8926    0.9491 
    
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:no_diagnose(x, ['缺血'])) #1003   0.932     0.9137    0.9101    0.9433 
    #scores = get_scores(organs, n_abnormals, samples,
    #                    lambda x:not no_diagnose(x, ['脑梗'])) #91  0.9645    0.9596    0.7405    0.9713 
    scores = get_scores(organs, n_abnormals, samples,
                        lambda x:not no_diagnose(x, ['缺血'])) #391  0.9269    0.9649    0.6987    0.9446 
    
    print(len(scores[(organs[0],0)]))
    with open(prediction_dir+'/AUC_%d.txt'%config['version'], 'a') as fp:
        #fp.write(str(len(scores[(organs[0],0)]))+'\n')
        
        macro_auc = {i:[] for i in range(n_abnormals)}
        macro_pos = {i:[] for i in range(n_abnormals)} #用来帮助计算加权macro_auc
        #macro_pos_d = {i:[] for i in range(n_abnormals)}
        #macro_auc_d = {i:[] for i in range(n_abnormals)}
        micro_scores = {i:[] for i in range(n_abnormals)}
        
        tb = pt.PrettyTable(border = False) #记录AUC
        for i in organs:
            for a in range(n_abnormals):
                n_pos = len(list(filter(lambda x:x[0], scores[(i,a)])))
                if n_pos==0:
                    continue
                auc = AUC(scores[(i,a)])
                #nc_auc = AUC(nc_scores[(i,a)])
                if auc>=0 and n_pos>2: #auc!=NaN
                    macro_auc[a].append(auc)
                    macro_pos[a].append(n_pos)
        
                    tb.add_column('%d_%d'%(i,a), [round(auc, 4), n_pos])
                #tb3.add_column('%d_%d'%(i,a), [round(nc_auc, 4), len(list(filter(lambda x:x[0], nc_scores[(i,a)])))])
                
                #fp.write('AUC_%d_%d'%(i,a)+' '+str(auc)+' '+str(n_pos)+'\n')
        print(tb)
        #fp.write(str(tb)+'\n')
        
        
        tb = pt.PrettyTable(border = False)
        for a in range(n_abnormals):
            temp = np.mean(np.asarray(macro_auc[a]))
            tb.add_column('macro AUC%d'%a, [np.round(temp, 4)])
        for a in range(n_abnormals):
            for i in organs:
                micro_scores[a].extend(scores[(i,a)])
            auc = AUC(micro_scores[a])
            tb.add_column('micro AUC%d'%a, [round(auc, 4)])
        
        for a in range(n_abnormals):
            if sum(macro_pos[a]):
                weighted = np.average(np.asarray(macro_auc[a]), weights=np.asarray(macro_pos[a]))
                tb.add_column('w macro%d'%a, [np.round(weighted, 4)])
        
        #fp.write(str(tb)+'\n')
        print(tb)
        fp.write('\n')
    

    organs = [8]
    for organ in organs:
        print(organ)
        with open(prediction_dir+'/scores_%d.txt'%organ, 'r') as fp:
            temp = fp.readlines()
        scores = {} #因为scores txt里可能重复好几次，这一步去重
        for score in temp:
            score = score.split(' ')
            score[0] = int(score[0]) #type
            score[2] = int(float(score[2])) #gt
            score[3] = float(score[3]) #predict
            scores[(score[4], score[0])] = score
        score_list = {}
        
        tb = pt.PrettyTable(border = False)
        interval = 60
        for j in range((100+interval-1)//interval):
            for i in range(4):
                score_list[i] = []
            for x in scores:
                age = getage(scores[x][4].split('_')[2].strip())
                if j*interval<=age and age<(j+1)*interval:
                    score_list[x[1]].append((scores[x][2], scores[x][3]))
            row = []
            for i in range(4):
                try:
                    row.append(round(AUC(score_list[i]), 4))
                except:
                    row.append('')
            for i in range(4):
                row.append(sum(x[0] for x in score_list[i]))
            tb.add_row([j, len(score_list[0])]+row)
        print(tb)
        
        tb = pt.PrettyTable(border = False)
        interval = 3
        for j in range((34+interval-1)//interval):
            for i in range(4):
                score_list[i] = []
            for x in scores:
                fold = int(scores[x][4].split('_')[0])
                if j*interval<=fold and fold<(j+1)*interval:
                    score_list[x[1]].append((scores[x][2], scores[x][3]))
            row = []
            for i in range(4):
                try:
                    row.append(round(AUC(score_list[i]), 4))
                except:
                    row.append('')
            for i in range(4):
                row.append(sum(x[0] for x in score_list[i]))
            tb.add_row([j, len(score_list[0])]+row)
        print(tb)
        
        a = []
        cnt_age = np.zeros((20))
        for x in scores:
            if x[1]==2 and scores[(x[0], 0)][2]==0 and scores[(x[0], 1)][2]==0 and scores[(x[0], 3)][2]==0:
                seriesID = scores[x][4].split('_')[2].strip()
                age = getage(seriesID)
                cnt_age[age//10] += 1
                a.append((scores[x][2], scores[x][3], scores[x][4], age))
        print('age', cnt_age)
        a.sort(key = lambda x:x[1], reverse=True)
        a1 = list(filter(lambda x:x[0]==1, a))
        a0 = list(filter(lambda x:x[0]==0, a))
        
        print(len(temp),len(scores))
        #print([x[0] for x in a])
        print(len(a1), len(a0))

        m = np.asarray([[-i for i in range(len(a1))], [x[3] for x in a1]])
        print(np.corrcoef(m)[0,1])
        
        m = np.asarray([[-i for i in range(len(a0))], [x[3] for x in a0]])
        print(np.corrcoef(m)[0,1])
        
        m = np.asarray([[-i for i in range(len(a))], [x[3] for x in a]])
        print(np.corrcoef(m)[0,1])
        #有无异常和年龄的关系，为正值很正常，年龄越大的人，有异常概率越大
        m = np.asarray([[x[0] for x in a], [x[3] for x in a]])
        print(np.corrcoef(m)[0,1])
        
        
        """
        左额叶低密度正样本预测
        预测值最高：1、片状低密度，明显 2、除了点片状不明显描述外，多有 ”脑室略扩张，脑池及脑沟增深增宽“类似的描述，预计年龄较大
        预测值最低：有点片状不明显描述，但除此之外其他异常较少，基本很少有脑室扩张、脑池脑沟增宽等特征，预计年龄较小
        """
        cnt = 0
        for x in a0[len(a0)*9//10:]:
            cnt += x[3]
        print(cnt/(len(a0)//10))
        
        cnt = 0
        for x in a0[:len(a0)//10]:
            cnt += x[3]
            #print(age, x[2], report_desc[reportID])
        print(cnt/(len(a0)//10))
        
        cnt = 0
        for x in a1[len(a1)*9//10:]:
            cnt += x[3]
            #print(age, x[2], report_desc[reportID])
        print(cnt/(len(a1)//10))
        cnt = 0
        for x in a1[:len(a1)//10]:
            cnt += x[3]
            #print(age, x[2], report_desc[reportID])
        print(cnt/(len(a1)//10))

def make_segmentation():
    """
    此函数应该用multitask3跑
    """
    report, fa, report_desc, report_label = preset()
    prediction_dir = os.path.abspath(config['valid_dir'])
    try:
        multitask = config['multitask']
    except:
        multitask = False
    try:
        dynamic_mask = config['dynamic_mask']
    except:
        dynamic_mask = False
    
    i = 0
    model = load_old_model(config["model_file"])
    data_file = tables.open_file(config["data_file"], "r")

    truth_file_name = config["data_file"][:-3]+'_truth.h5' #新的truth的h5文件名。
    if not config['overwrite'] and Path(truth_file_name).exists():
        print('truth file already exists')
        return
    hdf5_file = tables.open_file(truth_file_name, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    
    truth_shape = (0, 1, 160, 160, 80)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt32Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=len(data_file.root.truth))
    n = len(data_file.root.truth)
    print('number of samples', n)
    for i in tqdm(range(n)):
        segmentation = run_single_case(data_file, i, model)
        segmentation = np.uint8(np.greater(segmentation, 0.1))
        #print(segmentation.shape)
        truth = np.zeros((160,160,80), dtype = np.uint32)

        for j in range(20):
            temp = segmentation[j]
            print('float', np.max(temp), np.min(temp))
            if temp.shape!=truth.shape:
                temp = resize(temp, truth.shape)
            temp = np.uint32(temp)
            print('uint', np.max(temp), np.min(temp))
            truth = np.bitwise_or(truth, np.left_shift(temp, j))
        #print(segmentation.shape)
        #truth = np.asarray([data_file.root.truth[i]])
        #print(truth.shape)
        #print(type(data_file.root.truth[i]), data_file.root.truth[i].shape)

        truth_storage.append(truth[np.newaxis][np.newaxis])
        
        #下面输出只是为了检验新的truth的正确性，直接把uint32转为uint8问题也不大
        #cv2.imwrite('ori_truth.png', np.uint8(np.log((data_file.root.truth[i])[0,:,:,50])))
        #cv2.imwrite('new_truth.png', np.uint8(np.log((hdf5_file.root.truth[i])[0,:,:,50])))
        
        #print(hdf5_file.root.truth[i].shape)
        
    hdf5_file.close() #新写的truth文件
    data_file.close() #原来的完整文件，包括data、truth、affine
    

if __name__ == "__main__":
    
    
    #if config["training_file"][-1]!='l':
    #    config["training_file"] += '_-1.pkl'
    #if config["validation_file"][-1]!='l':
    #    config["validation_file"] += '_-1.pkl'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", help='downstream', action = 'store_true')
    parser.add_argument('-f', type=int)
    parser.add_argument('-v', type=int)
    args = parser.parse_args()
    print(args)
    
    if args.ds:
        from train_cls import config, make_config
        downstream = True
        if args.f>=0:
            make_config(args)
            print(config)
            main()
        else:
            folds = -args.f
            for i in range(0, folds):
                args.f = i
                make_config(args)
                print(config)
                main()
    else:
        from train_mil_multitask import config
        #config['dynamic_mask'] = False
        downstream = False
        main()
    
    #evaluate_prediction_result()
    #make_segmentation()
