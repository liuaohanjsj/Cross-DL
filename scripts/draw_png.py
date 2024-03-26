# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:33:28 2021

@author: Admin
"""

import cv2
import numpy as np
import matplotlib.colors as mcolors
import nibabel as nib
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils import load_cq500_original_label
import json
import pydicom
import nrrd

def plot_colortable(colors, title, sort_colors=True, emptycols=0, ncols=5):

    cell_width = 145 #170
    cell_height = 22
    swatch_width = 18 #48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = ncols - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    #ax.set_title(title, fontsize=16, loc="left", pad=10)

    for i, name in enumerate(names):
        #row = i % nrows
        #col = i // nrows
        row = i // ncols
        col = i % ncols
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], linewidth = 0, edgecolor='0.0')
        )
    fig.savefig('region label.svg', bbox_inches='tight')
    return fig

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
def extract_brain_plus(img, seed_mask = None, location = 0, k_size = 13, cal_center = False):
    """
    来自EHR/extract_brain
    img是软组织窗0-255的二维单通道uint图像
    以下参数均用于原分辨率（512*512）图像
    location是0到1的float，表示从下到上处于序列中哪个位置
    k_size是kernel的大小，默认9。越大则筛除颅外组织可能性越高。
    cal_center表示是否计算图像中心（扩散起点）。一般不需要，但是在fold_1这种未经中轴线矫正的序列中，需要重新计算图像中心
    """
    def find_center(mask):
        (n,m) = mask.shape
        cnth = np.sum(np.arange(n)[:,np.newaxis]*mask)
        cntl = np.sum(np.arange(m)[np.newaxis]*mask)
        cnt = np.sum(mask)
        if cnt==0:
            return n//2, m//2
        return int(cnth/cnt), int(cntl/cnt)
    mask = np.greater(img, 0)*np.less(img, 200) #会无法区分颅内积气。不过即使扩散到部分颅外组织，也能删除掉较薄的颅外组织
    #mask = np.less(img, 255) #如果不慎扩散到一点颅外组织，就会扩散到外界空气，从而使得全部颅外组织都被扩散到
    
    mask = np.uint8(mask) #软组织区域mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size+5, k_size+5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    
    ff_mask = cv2.erode(mask, kernel) #腐蚀后的软组织mask，用于floodfill
    n = img.shape[0]
    m = img.shape[1]
    region = np.zeros((n+2, m+2), np.uint8)
    if seed_mask is None or location>0.7:
        h = n//2
        l = m//2
        if cal_center:
            h,l = find_center(np.greater(img, 0))
        #print(h,l)
        if ff_mask[h,l]: #用location判断是防止头顶部分配准mask不全，用这种方法可以保证是全的
            cv2.floodFill(ff_mask, region, (l, h), 1) #cv2坐标
    if seed_mask is not None:
        temp_mask = cv2.erode(seed_mask, kernel3)
        for i in range(0, n, 4): #用大步长增加效率
            for j in range(0, m, 4):
                if temp_mask[i,j] and region[i+1,j+1]==0 and ff_mask[i,j]: #这里必须用经过腐蚀的ff_mask判断而不是img
                    cv2.floodFill(ff_mask, region, (j, i), 1)
                    
    extended_result = cv2.dilate(region[1:-1, 1:-1], kernel2)
    return extended_result*mask*img, extended_result*mask

def actual_shape(shape, affine):
    """
    根据nii的shape (x,y,z)和affine，算出空间中正确的shape (x,y,newz)
    z是竖直方向
    """
    origin = reference_location(0,0,0,affine)
    x = np.abs(reference_location(1,0,0,affine) - origin)
    y = np.abs(reference_location(0,1,0,affine) - origin)
    z = np.abs(reference_location(0,0,1,affine) - origin)
    if x[1] or x[2] or y[0] or y[2] or z[0] or z[1]:
        print('fuck', x, y, z)
        return (-1,-1,-1)
    
    if abs((shape[0]*y[1])/(shape[1]*x[0])-1)>0.01:
        print('fuck', x, y, shape)
        return (-1,-1,-1)
    return (shape[0], shape[1], int(shape[2]*z[2]/x[0]))

def translate(img, x):
    #为了调整二维图像的上下间距。。。img:[h,w,3]
    return np.concatenate((img[x:], img[:x]), 0)

def output_manual_segmentation():
    CT = nib.load('temp/result.nii').get_data()
    CT = cv2.flip(np.transpose(CT, [1,0,2]), 0)
    print(CT.shape, CT.dtype, np.max(CT), np.min(CT))
    CT = np.minimum(np.maximum(CT, -5), 75)
    CT = (CT+5)*255/80
    for i in range(CT.shape[2]):
        CT3 = np.stack((CT[:,:,i],CT[:,:,i],CT[:,:,i]), 2)
        cv2.imwrite('temp/reference_%s.png'%(str(i).zfill(2)), CT3)
        print(CT3.dtype)
        seg = np.zeros(CT3.shape, np.float64)
        label_file = 'C:\\Users\\Admin\\Desktop\\averageCT\\新建文件夹(2)\\新建文件夹-utf8-脑室/FinalAverage%s.json'%(str(i).zfill(2))
        label_id = {'左半卵圆中心':0,'右半卵圆中心':1, '左颞叶': 2,'右颞叶': 3,
                    '左小脑': 4,'右小脑': 5,'左岛叶': 6,'右岛叶': 7,
                    '左额叶': 8,'右额叶':  9,'左顶叶': 10,'右顶叶': 11,
                    '脑桥': 12,'左脑室': 13,  '右脑室':14,'左基底节区': 15,'右基底节区':16 ,
                    '脑干': 17,'左枕叶':18,'右枕叶':19,}
        if Path(label_file).exists():
            with open(label_file, 'r', encoding='utf-8') as fp:
                label = json.load(fp)
            for shape in label['shapes']:
                if shape['label'] not in label_id:
                    continue
                c = mcolors.to_rgb(region_color_list[label_id[shape['label']]+1])
                c = np.array((c[2],c[1],c[0]))*255
                cv2.fillPoly(seg, [np.array(shape['points']).astype(np.int32)], c)  #fillPoly里点坐标需要是整数
            add = cv2.addWeighted(CT3, 0.5, seg, 0.5, 0)
            cv2.imwrite('temp/reference_segmentation_%s.png'%(str(i).zfill(2)), add)


def draw_segmentation(CT, gt, seg, path, prefix = '', organs = [],
                    save = True, color = []):
    def change(color, organ): #rgb和bgr之类的反一下
        ret = np.array([color[2], color[1], color[0]])
        if organ in [1,3,5,9,11,14,16,19]: #左侧脑区，序号加了1，因为0是无脑区
            ret *= 1.0
        return ret

    """
    CT: (160,160,80)
    gt: (160,160,80)，二进制编码
    seg: (20,80,80,40)
    """
    print('CT', CT.shape)
    print('gt', gt.shape)
    print('seg', seg.shape)
    assert CT.shape==gt.shape
    #print(np.max(CT), np.min(CT))
    maxn = np.max(CT)
    minn = np.min(CT)
    CT = np.uint8((CT-minn)/(maxn-minn)*255)
    
    
    temp = np.zeros([20]+list(CT.shape))
    for i in range(20):
        temp[i] = resize(seg[i], CT.shape)
        temp[i] = cv2.flip(np.transpose(temp[i], [1,0,2]), 0)
    seg = temp
    for i in range(0, CT.shape[2], 4):
        for j in range(20):
            cv2.imwrite(str(path)+'/segmentation%s_%s.png'%(str(i).zfill(2), str(j).zfill(2)), seg[j,:,:,i]*255)
    
    CT = np.transpose(CT, [1,0,2])
    CT = cv2.flip(CT, 0)

    for i in [6,7,12]: #排除这几个channel的干扰
        seg[i] = 0
    seg[6] = 0.5 #把6号channel值设为0.5，这样如果argmax之后结果是6的话，就说明所有脑区预测概率都小于0.5，就认为不属于任何脑区了
    organ_index = np.argmax(seg, axis=0) #pred是[channel, l1, l2, l3]的array，在0做argmax得到的就是[l1,l2,l3]大小的索引数组
    has_organ = np.float32(np.not_equal(organ_index, 6))
    seg = has_organ*(np.float32(organ_index)+1)
    #seg = cv2.flip(np.transpose(seg, [1,0,2]), 0)
    seg = seg.astype(np.uint8)
    print(seg.shape)
    
    #print('gt', gt.shape, gt.dtype)
    
    gt = np.transpose(gt, [1,0,2])
    mask = np.zeros(CT.shape, np.uint8)
    for i in organs:
        temp = np.bitwise_and(np.right_shift(gt, i), 1)
        temp = np.uint8(temp)
        mask = np.maximum(mask, cv2.flip(temp, 0)*(i+1)) #organ i在mask里用i+1标志
    #print('mask', mask.shape)
    gt = resize(mask, CT.shape)
    
    
    print(len(color), color)
    ret = []
    for i in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2).astype(np.float64)
        output = CT3.copy()
        
        color_img = np.zeros(CT3.shape)
        for j in range(gt.shape[0]):
            for k in range(gt.shape[1]):
                color_img[j,k] = change(color[gt[j,k,i]], gt[j,k,i])
        gt_add = cv2.addWeighted(CT3, 0.5, color_img*255, 0.5, 0)
        cv2.imwrite(str(path)+'/pseudo segmentation_%s.png'%(str(i).zfill(2)), gt_add)
        
        output = np.concatenate((output, gt_add), 1)
        
        color_img = np.zeros(CT3.shape)
        for j in range(seg.shape[0]):
            for k in range(seg.shape[1]):
                color_img[j,k] = change(color[seg[j,k,i]], seg[j,k,i])
        seg_add = cv2.addWeighted(CT3, 0.5, color_img*255, 0.5, 0)
        cv2.imwrite(str(path)+'/predicted segmentation_%s.png'%(str(i).zfill(2)), seg_add)
        
        output = np.concatenate((output, seg_add), 1)
        if save:
            cv2.imwrite(str(path)+'/result%s_%s.png'%(prefix, str(i).zfill(2)), output)
        else:
            ret.append(output)
    return ret

def draw_abnormality(CT, pred, path, organs = [], save = True,
                     use_brain_mask = False, use_manual_seg = False, label_path = '',
                     view = 'axial', affine = None, add_sagittal=None):
    """
    CT: (160,160,80)
    pred: (4,80,80,40)
    """
    def output_abnormality(CT, pred, save = True, image_add = None, slice_location = 0):
        """
        image_add应为三通道图像
        """
        label_id = {'脑内高':0,'脑外高':1,'脑内低':2,'脑外低':3}
        ret = []
        if image_add is not None: #在第0维度，即图片长与CT补齐
            if image_add.shape[0]<CT.shape[0]:
                temp = CT.shape[0]-image_add.shape[0]
                image_add = np.pad(image_add, ((temp//2, (temp+1)//2),(0,0),(0,0)), 'constant', constant_values=(0,0))
            else:
                temp = image_add.shape[0]-CT.shape[0]
                image_add = image_add[temp//2:temp//2+CT.shape[0]]
        for i in range(0, CT.shape[2], 4):
            CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2)
            output = CT3.copy()
            if use_manual_seg:
                add = CT3.copy().astype(np.uint8)
                colors = [x for x in mcolors.TABLEAU_COLORS]
                if Path(label_path+'/CT_%s.json'%(str(i).zfill(2))).exists():
                    with open(label_path+'/CT_%s.json'%(str(i).zfill(2)), 'r', encoding='utf-8') as fp:
                        seg = json.load(fp)
                    for shape in seg['shapes']:
                        c = mcolors.to_rgb(colors[label_id[shape['label']]])
                        c = np.array((c[2],c[1],c[0]))*255
                        cv2.fillPoly(add, [np.array(shape['points']).astype(np.int32)], c)  #fillPoly里点坐标需要是整数
                output = np.concatenate((output, add), 1)
            for a in range(4):
                color_img = cv2.applyColorMap(np.uint8(pred[a,:,:,i]*255), cv2.COLORMAP_JET)
                add = cv2.addWeighted(CT3, 0.5, color_img, 0.5, 0)
                cv2.imwrite(str(path)+'/%s_%s_%d.png'%(view, str(i).zfill(2), a), add)
                output = np.concatenate((output, add), 1)
            if image_add is not None:
                output = np.concatenate((output, image_add), 1)
                output[:,slice_location] = 255
            if save:
                cv2.imwrite(str(path)+'/%s_%s.png'%(view, str(i).zfill(2)), output)
            ret.append(output)
        return ret
    def change(color): #rgb和bgr之类的反一下
        return np.array([color[2], color[1], color[0]])

    CT = np.uint8((CT-np.min(CT))/(np.max(CT)-np.min(CT))*255)
    CT = cv2.flip(np.transpose(CT, [1,0,2]), 0)
    
    temp = np.zeros([4]+list(CT.shape))
    for i in range(4):
        t = cv2.flip(np.transpose(pred[i], [1,0,2]), 0)
        temp[i] = resize(t, CT.shape)#, cv2.INTER_LINEAR)
    pred = temp
    if use_brain_mask:
        brain_mask = np.uint8(np.not_equal(np.min(CT), CT))
        brain_mask = resize(brain_mask, pred[0].shape)
        for a in range(4):
            pred[a] *= brain_mask

    #输出binary的异常预测图
    pred5 = np.concatenate([np.ones([1]+list(pred.shape[1:]))*0.5, pred])
    binary_pred = np.argmax(pred5, axis=0)
    color = ['#000000']+[mcolors.TABLEAU_COLORS[x] for x in mcolors.TABLEAU_COLORS]
    color = [mcolors.hex2color(x) for x in color]
    
    '''
    for i in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2)
        color_img = np.zeros(list(CT.shape[:2])+[3], np.uint8)
        for j in range(binary_pred.shape[0]):
            for k in range(binary_pred.shape[1]):
                color_img[j,k] = change(color[binary_pred[j,k,i]])*255
        add = cv2.addWeighted(CT3, 0.5, color_img, 0.5, 0)
        cv2.imwrite(str(path)+'/binary abnormality_%s.png'%(str(i).zfill(2)), add)
    '''
    #做矢状位图像：
    '''
    new_shape = actual_shape(CT.shape, affine)
    print('sagittal shape', new_shape)
    if new_shape[0]!=-1:
        sagittal_CT = resize(CT, new_shape, cv2.INTER_CUBIC)
        sagittal_pred = []
        for i in range(4):
            sagittal_pred.append(resize(pred[i], new_shape, cv2.INTER_LINEAR))
        sagittal_pred = np.asarray(sagittal_pred)
        sagittal_CT = np.flip(np.transpose(sagittal_CT, (2,0,1)), 0)
        sagittal_pred = np.flip(np.transpose(sagittal_pred, (0,3,1,2)), 1)
        print(sagittal_CT.shape, sagittal_pred.shape)
        
    #nrrd.write(str(path)+'/CT.nrrd', sagittal_CT)
    #nrrd.write(str(path)+'/pred.nrrd', np.max(sagittal_pred, 0))
    #return

    if view=='axial':
        image_add = None
        slice_location = 0
        if add_sagittal is not None:
            sagittal_output = output_abnormality(sagittal_CT, sagittal_pred, save = False)
            width = sagittal_output[0].shape[1]//6
            ori_sagittal = sagittal_output[add_sagittal[1]//4][:,0:width]
            pred_sagittal = sagittal_output[add_sagittal[1]//4][:,width*(2+add_sagittal[0]):width*(3+add_sagittal[0])]
            image_add = np.concatenate((ori_sagittal, pred_sagittal), 1)
            slice_location = CT.shape[1]*(2+add_sagittal[0])+add_sagittal[1]
        ret = output_abnormality(CT, pred, image_add=image_add, slice_location=slice_location)
    elif view=='sagittal':
        ret = output_abnormality(sagittal_CT, sagittal_pred)
    '''
    
    '''
    for i in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2)
        output = CT3.copy()
        for a in range(4):
            color_img = cv2.applyColorMap(np.uint8(pred[a,:,:,i]*255), cv2.COLORMAP_JET)
            add = cv2.addWeighted(CT3, 0.5, color_img, 0.5, 0)
            output = np.concatenate((output, add), 1)
        if save:
            cv2.imwrite(str(path)+'/abnormality_%s.png'%(str(i).zfill(2)), output)
        else:
            ret.append(output)
    '''
    
    for i in range(0, CT.shape[2], 4):
        CT3 = np.stack((CT[:,:,i], CT[:,:,i], CT[:,:,i]), 2)
        cv2.imwrite(str(path)+'/CT%s.png'%(str(i).zfill(2)), CT3)
        
        for a in range(4):
            color_img = cv2.applyColorMap(np.uint8(pred[a,:,:,i]*255), cv2.COLORMAP_JET)
            add = cv2.addWeighted(CT3, 0.5, color_img, 0.5, 0)
            #output = np.concatenate((output, add), 1)
            #cv2.imwrite(str(path)+'/abnormality%s_%d.png'%(str(i).zfill(2), a), color_img)
            cv2.imwrite(str(path)+'/abnormality%s_%d.png'%(str(i).zfill(2), a), pred[a,:,:,i]*255)
        
    return ret

def output_nrrd(CT, pred, affine):
    """
    CT: (160,160,80)
    pred: (24,80,80,40)
    """
    CT = np.uint8((CT-np.min(CT))/(np.max(CT)-np.min(CT))*255)
    CT = cv2.flip(np.transpose(CT, [1,0,2]), 0)
    
    temp = np.zeros([24]+list(CT.shape))
    for i in range(24):
        t = cv2.flip(np.transpose(pred[i], [1,0,2]), 0)
        temp[i] = resize(t, CT.shape, cv2.INTER_LINEAR)
    pred = temp
    
    #做矢状位图像：
    new_shape = actual_shape(CT.shape, affine)
    print('actual shape', new_shape)
    CT = resize(CT, new_shape, cv2.INTER_CUBIC)
    temp = []
    for i in range(24):
        temp.append(resize(pred[i], new_shape, cv2.INTER_LINEAR))
    pred = np.asarray(temp)
    #sagittal_CT = np.flip(np.transpose(sagittal_CT, (2,0,1)), 0)
    #sagittal_pred = np.flip(np.transpose(sagittal_pred, (0,3,1,2)), 1)
    print(CT.shape, pred.shape)
    '''
    for i in range(CT.shape[2]):
        img, mask = extract_brain_plus(CT[:,:,i], seed_mask=None, location=i/CT.shape[2], k_size = 13, cal_center = False)
        print(img.shape)
        cv2.imwrite('temp/temp%d.png'%i, img)
    '''
    print(CT.dtype, pred.dtype, np.max(CT), np.min(CT))
    #nrrd.write('temp/CT.nrrd', CT)
    #nrrd.write('temp/pred.nrrd', np.max(pred[20:], 0))
    
#color_list = [mcolors.TABLEAU_COLORS[x] for x in mcolors.TABLEAU_COLORS]
'''
color_list = ['#000000', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e',
              '#2ca02c', '#2ca02c', '#000000', '#000000', 
              '#d62728', '#d62728', '#9467bd', '#9467bd',
              '#000000', '#8c564b', '#8c564b', '#e377c2', '#e377c2',
              '#7f7f7f', '#bcbd22', '#bcbd22']
'''
region_color_list = ['black', 'tab:purple', 'tab:purple', 'tab:brown', 'tab:brown', 
                     'tab:pink', 'tab:pink', 'black', 'black',
                     'tab:gray', 'tab:gray', 'tab:olive',  'tab:olive',
                     'black', 'bisque', 'bisque', 'yellow', 'yellow', 
                     'palegreen', 'skyblue', 'skyblue']

colors = {'Occipital lobe':'#bcbd22', 'Temporal lobe':'#ff7f0e',
          'Frontal lobe':'#d62728', 'Parietal lobe':'#9467bd',
          'Cerebellum': '#2ca02c', 'Lateral ventricle':'#8c564b', 
          'Basal ganglia': '#e377c2','Centrum ovale':'#1f77b4',
          'Brain stem': '#7f7f7f'}
colors = {x:mcolors.hex2color(colors[x]) for x in colors}
plot_colortable(colors, "Anatomical regions",
                sort_colors=False, emptycols=0)

plt.rcParams['font.sans-serif'] = ['SimHei']
color_list = [mcolors.TABLEAU_COLORS[x] for x in mcolors.TABLEAU_COLORS]
colors = {'脑内高':color_list[0], 
          '脑外高':color_list[1],
          '脑内低':color_list[2],
          '脑外低':color_list[3],}
plot_colortable(colors, "Abnormalities",
                sort_colors=False, emptycols=0, ncols=4)

'''
n = 256
img = np.arange(n)
img = np.tile(img, 5)
img = np.resize(img, (5,n))
img = cv2.applyColorMap(np.uint8(img*255/(n-1)), cv2.COLORMAP_JET)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img.shape)
cv2.imwrite('temp.png', img) #这是一种方法，但是得到的不是矢量图。
fig, ax = plt.subplots(1,1,figsize=(12,1))
ax.set_yticks([])
ax.set_xticks(np.arange(0,n,(n-1)/5))
ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0])
ax.imshow(img, cmap=plt.get_cmap('jet')) #cmap是控制下面的colorbar的
#plt.colorbar() #这个colorbar是matplotlib自己的colorbar，相当于套娃。这个冗余了
plt.savefig('temp.svg', bbox_inches = 'tight')
'''
def reference_location(i,j,k, affine):
    return np.dot(affine[:3,:3], np.array((i,j,k)))+affine[:3,3]

def load(path, output_path, use_brain_mask = False, cq500 = False, use_manual_seg = False, label_path='',
         view = 'axial', add_sagittal = None):
    organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
    #organs = [0,1,2,3,4,5,8,9]
    #organs = [8]
    Path(output_path).mkdir(exist_ok = True, parents = True)
    try:
        CT = nib.load(path+'/data_CT_brain.nii.gz').get_data()
        affine = nib.load(path+'/data_CT_brain.nii.gz').get_affine()
        print(CT.shape)
        print('affine', affine.shape)
        print(affine)
        print('0,0,0 ', reference_location(0,0,0,affine))
        print('0,0,80 ', reference_location(0,0,80,affine))
        print('0,160,0 ', reference_location(0,160,0,affine))
        print('0,160,80 ', reference_location(0,160,80,affine))
        print('160,160,80 ', reference_location(160,160,80,affine))
    except:
        CT = nib.load(path+'/data_CT.nii.gz').get_data()
    pred_paths = list(Path(path).glob('prediction_*.nii.gz'))
    preds = []
    for p in pred_paths:
        preds.append(nib.load(str(p)).get_data())
    try:
        pred = nib.load(str(path)+'/prediction.nii.gz').get_data()
    except:
        pred = preds[0] #控制输出哪个scale，0是最低分辨率，-1是最高分辨率
    gt = nib.load(str(path)+'/truth.nii.gz').get_data()
    
    draw_segmentation(CT, gt, pred[:20], path=output_path, organs = organs,
                      save = True, #color = [mcolors.hex2color(x) for x in color_list]
                      color = [mcolors.to_rgb(x) for x in region_color_list])
    
    draw_abnormality(CT, pred[20:24], path=output_path, organs = organs, save = True, 
                     use_brain_mask=use_brain_mask, use_manual_seg=use_manual_seg,
                     label_path=label_path, view=view, affine = affine, add_sagittal=add_sagittal)
    
    #output_nrrd(CT, pred, affine)
    
def load_cq500_score(dir):
    """
    dir是一个序列的文件夹
    """
    idx = [1,4,2,3,0]
    gts = np.zeros(5) #[颅内出血、脑内出血、左侧、右侧、脑外出血]
    preds = np.zeros(5)
    for i in range(5):
        with open(str(Path(dir).parent)+'/cq500_%d_max.txt'%i, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            gt = float(line.split()[0])
            pred = float(line.split()[1])
            path = ' '.join(line.split()[2:]).strip()
            #print(path, Path(dir).name)
            if 'cq500_'+path==Path(dir).name:
                gts[idx[i]] = gt
                preds[idx[i]] = pred
    return gts, preds

def summary_cq500_score(path, output_path, labels):
    Path(output_path).mkdir(exist_ok = True, parents = True)
    agt,apred = load_cq500_score(path)
    np.savetxt(output_path+'/score.txt', np.stack((agt, apred), 1), fmt='%.03f')
    np.savetxt(output_path+'/original.txt', np.array(labels[Path(path).name.split('_')[1]]))

def output_CT(path, output_path):
    """
    把预处理后的CT（模型输入） 输出为png，用于异常分割标注
    """
    Path(output_path).mkdir(exist_ok = True, parents = True)
    try:
        CT = nib.load(path+'/data_CT_brain.nii.gz').get_data()
    except:
        CT = nib.load(path+'/data_CT.nii.gz').get_data()

    CT = np.uint8((CT-np.min(CT))/(np.max(CT)-np.min(CT))*255)
    CT = cv2.flip(np.transpose(CT, [1,0,2]), 0)
    for i in range(0,80,4):
        cv2.imwrite(str(output_path)+'/CT_%s.png'%(str(i).zfill(2)), CT[:,:,i])

#output_manual_segmentation()

#models = ['test_multitask82', 'test_multitask80', 'test_multitask83']
models = [#'test_multitask85', 
          #'test_multitask85.1',
          #'test_multitask89',
          ]

series = [#'02_28776928.0_1.3.12.2.1107.5.1.4.32219.30000017101601364356200000000',
          #'03_30687919.0_1.3.12.2.1107.5.1.4.32219.30000018050314061196800054831',
          #'04_18212598.0_1.3.12.2.1107.5.1.4.32219.30000014041606084537500016855',
          #'04_18929612.0_1.3.12.2.1107.5.1.4.32219.30000014071400250309300018505',
          #'04_19279722.0_1.3.12.2.1107.5.1.4.43501.30000014082623560560900009408',
          #'05_19816733.0_1.2.840.113619.2.289.3.296546961.399.1415144326.132',
          #'05_20496183.0_1.2.840.113619.2.289.3.296546961.227.1423712732.191',
          #'05_24937312.0_1.2.156.112605.161339402051.20160803070120.3.10124.5',
          #'05_25265715.0_1.3.12.2.1107.5.1.4.32219.30000016090814283835900000441',
          #'05_25617795.0_1.2.840.113619.2.289.3.296546961.512.1476867427.240',
          #'05_26560135.0_1.2.840.113619.2.289.3.296546961.798.1487225860.750',
          #'05_26618191.0_1.2.840.113619.2.289.3.296546961.113.1487835111.94',
          #'05_26958769.0_1.2.156.112605.14038005400544.20170330061315.3.5904.5',
          #'06_27586941.0_1.2.156.112605.14038011836937.20170605024318.3.5932.5',
          #'06_28198649.0_1.3.12.2.1107.5.1.4.32219.30000017080803263712500034701',
          #'06_28542841.0_1.3.12.2.1107.5.1.4.32219.30000017091401304628100007401',
          #'10_19508032.0_1.3.12.2.1107.5.1.4.43501.30000014092615111420300000140',
          #'12_26327427.0_1.3.12.2.1107.5.1.4.32219.30000017011308042951500028864',
          #'12_27020469.0_1.2.156.112605.14038011836937.20170413004020.3.6216.5',
          #'12_27046687.0_1.2.156.112605.14038011836937.20170411000007.3.6216.5',
          '13_28555793.0_1.2.840.113619.2.289.3.296546961.674.1505346616.5',
          
          ]

for s in series:
    output_CT('prediction/test_multitask85/%s'%s, 'temp/manual_seg/%s'%s)

'''
series = [#'03_30687919.0_1.3.12.2.1107.5.1.4.32219.30000018050314061196800054831',
          #'06_28542841.0_1.3.12.2.1107.5.1.4.32219.30000017091401304628100007401', #薄层
          
          #'06_28198649.0_1.3.12.2.1107.5.1.4.32219.30000017080803263712500034701',
          #'04_19279722.0_1.3.12.2.1107.5.1.4.43501.30000014082623560560900009408',
          #'05_25265715.0_1.3.12.2.1107.5.1.4.32219.30000016090814283835900000441',
          
          #'15_19522759.0_1.2.840.113619.2.289.3.296546961.416.1411977714.305',
          #'14_31817949.0_1.3.12.2.1107.5.1.4.32219.30000018081908215795300000101',
          #'13_28555793.0_1.2.840.113619.2.289.3.296546961.674.1505346616.5', #system overview
          #'13_28393693.0_1.2.840.113619.2.289.3.296546961.705.1503985609.157',
          #'01_23987434.0_1.3.12.2.1107.5.1.4.32219.30000016041615231731200000530',
          #'04_19279722.0_1.3.12.2.1107.5.1.4.43501.30000014082623560560900009408',
          '05_26560135.0_1.2.840.113619.2.289.3.296546961.798.1487225860.750',
          #'12_26327427.0_1.3.12.2.1107.5.1.4.32219.30000017011308042951500028864',
          ]
'''
for model in models:
    for s in series:
        load('prediction/%s/%s'%(model, s), 'temp/%s/%s'%(model, s),
             use_brain_mask = False, use_manual_seg = False, view='axial')



series_s = [#('03_30687919.0_1.3.12.2.1107.5.1.4.32219.30000018050314061196800054831',3,108),
            #('06_28542841.0_1.3.12.2.1107.5.1.4.32219.30000017091401304628100007401',2,112), #薄层
            #('06_28198649.0_1.3.12.2.1107.5.1.4.32219.30000017080803263712500034701',2,60),
            #('04_19279722.0_1.3.12.2.1107.5.1.4.43501.30000014082623560560900009408',0,40),
            #('05_25265715.0_1.3.12.2.1107.5.1.4.32219.30000016090814283835900000441',0,52),
            ]
for model in models:
    for s in series_s:
        load('prediction/%s/%s'%(model, s[0]), 'temp/%s/%s'%(model, s[0]),
             use_brain_mask = False, use_manual_seg = True,
             label_path='temp/manual_seg/%s'%s[0], view='axial',
             add_sagittal = s[1:])

'''
models = [#'test_cq500_80', 'test_cq500_82', 'test_cq500_3_83'
          'test_cq500_3_85'
         ]

series = ['cq500_438_CT 2.55mm',
          'cq500_436_CT Plain',
          'cq500_371_CT Plain 3mm',
          'cq500_217_CT 55mm Plain',
          'cq500_227_CT 55mm Plain',
          'cq500_33_CT Plain',
          'cq500_40_CT 5mm',
          'cq500_239_CT PRE CONTRAST 5MM STD',
          ]

labels = load_cq500_original_label('../../../public/cq500/label.xlsx', naoshi=True)
for model in models:
    for s in series:
        load('prediction/%s/%s'%(model, s), 'temp/%s/%s'%(model, s),
             use_brain_mask = False)
        #summary_cq500_score('prediction/%s/%s'%(model, s), 'temp/%s/%s'%(model, s), labels)
'''

'''
series_paths = ['10/18271350.0/1.2.840.113820.861002.41752.3582364005625.2.1606117/1.3.12.2.1107.5.1.4.54143.30000014042207543556200001510',
                '15/32629579.0/1.2.840.113820.861002.43416.344156713140.2.2513984/1.2.156.112605.14038011838267.20181112010357.3.6204.6',
                '10/18468991.0/1.2.840.113820.861002.41774.9232581019500.2.6430/1.3.12.2.1107.5.1.4.43501.30000014051422394510900000092',
                ]
for path in series_paths:
    png_paths = list(Path('Y:\\aligned_registration/'+path).glob('*ori*.png'))
    print(len(png_paths))
    output = []
    for i in range(6):
        img = cv2.imread(str(png_paths[i*len(png_paths)//6]), 0)
        output.append(img)
    output = np.concatenate(output, 1)
    print(output.shape)
    cv2.imwrite('temp/bad/%s.png'%Path(path).name, output)
'''


def experiment_nii():
    #series_path = '/home/liuaohan/pm-stroke/aligned_registration/10/18902293.0/1.2.840.113820.861002.41830.666668287140.2.1659338/1.3.12.2.1107.5.1.4.54143.30000014070616151831200002206'
    series_path = '/home/liuaohan/pm-stroke/aligned_registration/06/28542841.0/1.2.840.113820.861002.42992.5697448264953.2.1758676/1.3.12.2.1107.5.1.4.32219.30000017091401285248400000109'
    def check_nii(nii_path):
        nii = nib.load(nii_path)
        shape = nii.get_data().shape
        affine = nii.get_affine()
        origin = reference_location(0,0,0,affine)
        print('0,0,0', origin)
        print('0,1,0', reference_location(0,1,0,affine)-origin)
        print('1,0,0', reference_location(1,0,0,affine)-origin)
        print('0,0,1', reference_location(0,0,1,affine)-origin)
        print(reference_location(511,511,shape[2]-1,affine))
        print('actual shape', actual_shape(shape, affine))
    check_nii('/home/liuaohan/pm-stroke/aligned_registration/06/28542841.0/1.2.840.113820.861002.42992.5697448264953.2.1758676/1.3.12.2.1107.5.1.4.32219.30000017091401285248400000109/CT.nii.gz')
    check_nii('prediction/test_multitask85/06_28542841.0_1.3.12.2.1107.5.1.4.32219.30000017091401304628100007401/data_CT_brain.nii.gz')
    with open('/home/liuaohan/pm-stroke/project/EHR/json/all_report.json', 'r') as fp:
        report = json.load(fp)
    reportID = Path(series_path).parent.parent.name
    print(report[reportID])
    for studyID in report[reportID]:
        for series in report[reportID][studyID]:
            if series['seriesID']==Path(series_path).name:
                ori_path = series['path']
    print(ori_path)
    dcm_paths = list(Path('/data/ctclass/RAW/'+ori_path).glob('*.dcm'))
    for path in dcm_paths:
        dcm = pydicom.read_file(path)
        print(dcm.PixelSpacing)
        print(dcm.ImagePositionPatient)
        #print(dcm.SpacingBetweenSlices)

#experiment_nii()

img = cv2.imread('CT36.png')

region = cv2.imread('segmentation36_02.png')
heat_img = cv2.applyColorMap(region, cv2.COLORMAP_JET)
temp = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
cv2.imwrite('temp.png', temp)

ab = cv2.imread('abnormality36_0.png')
heat_img = cv2.applyColorMap(ab, cv2.COLORMAP_JET)
temp = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
cv2.imwrite('temp2.png', temp)

x = region/255*(ab/255)*255
print(np.max(x), np.min(x))
heat_img = cv2.applyColorMap(x.astype(np.uint8), cv2.COLORMAP_JET)
temp = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
cv2.imwrite('temp3.png', temp)
