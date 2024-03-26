# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 22:03:59 2020

@author: Admin
"""

import os
import glob
from pathlib import Path
import sys
import numpy as np
import json
import tables
from tqdm import tqdm

project_path = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model.mil_multitask import mil_multitask_model
from unet3d.training import load_old_model, train_model


config = dict()
config['version'] = 85
config['sub_version'] = 0
config['public_test'] = 'cq500_2' #'', 'cq500', 'cq500_2', 'prospective'
#cq500是只有脑部区域，大部分模型用的数据
#83和85输入头部数据，要用cq500_2或cq500_3。
#cq500_3，resize使用cv2。左右是和其他相反的，测试要用flip

#prospective实际不算是public了，但是由于处理流程和原本数据集不同，用public数据集的方式处理。

if config['sub_version']==0:
    config['full_version'] = str(config['version'])
else:
    config['full_version'] = str(config['version'])+'.'+str(config['sub_version'])
print(config['full_version'])

config['output_shape'] = None #默认和输入一样

if config['version']==53 or config['version']==55 or config['version']==56 or config['version']==70\
or config['version']==71 or config['version']==72 or config['version']==73 or config['version']==74\
or config['version']==75 or config['version']==76 or config['version']==77 or config['version']==80\
or config['version']==81 or config['version']==82 or config['version']==83 or config['version']==84\
or config['version']==85 or config['version']==86 or config['version']==87 or config['version']==88\
or config['version']==89 or config['version']==92:
    config['output_shape'] = (80,80,40)
elif config['version']==90 or config['version']==91:
    config['output_shape'] = (20,20,10) #其实90只做分类任务，没有输出。但是generator没有改，还是会resize一个只是不返回。这里弄一个比较小的节省时间

config['truth_file'] = None

config['ab_seg_file'] = None
#config['ab_seg_file'] = os.path.abspath("/data/liuaohan/co-training/abnormality_seg4.h5") #新增了一小部分序列
config['load_loss_weight'] = True
config['separate_mil'] = True
config['no_segmentation'] = False
config['triple_loss'] = True
config['classify_only'] = False

config['label_file'] = None

if config['version']==80 or config['version']==81 or config['version']==82 or config['version']==83\
or config['version']==84 or config['version']==85 or config['version']==86 or config['version']==87\
or config['version']==88 or config['version']==89 or config['version']==90 or config['version']==91\
or config['version']==92 or config['version']==93:
    config['label_file'] = 'all_refined_label_ischemia_false_naoshi.h5'
    #因为要用不同准确度NLP标签训练CT模型，下面的label_file不能带真实标签
    #但是测试时又需要测试集真实标签，因此使用test_label_file
    config['test_label_file'] = 'all_refined_label_ischemia_false_naoshi_doctor.h5'
    if config['full_version']=='85.100': #用1000个句子训练的NLP预测的标注
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_1000.h5'
    if config['full_version']=='85.101': 
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_2000.h5'
    if config['full_version']=='85.102': 
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_4000.h5'
    if config['full_version']=='85.103': 
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_300.h5'
    if config['full_version']=='85.104': 
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_3000.h5'
    if config['full_version']=='85.105': 
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_1500.h5'
    if config['full_version']=='85.106':
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_99.h5'
    if config['full_version']=='85.107':
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_100.h5'
    if config['full_version']=='85.110':
        config['label_file'] = 'all_refined_label_ischemia_false_naoshi_smallsilver.h5'
    print('label file', config['label_file'])

#config['dynamic_mask']只在predict.py用到
config['dynamic_mask'] = True

config['valid_epoch'] = None
if config['version']>=80:
    config['train_epoch'] = 8000
    #config['valid_epoch'] = None

#这个rotate是指生成数据的时候是否rotate，而不是generator
config['rotate'] = True

config['multitask'] = True #在predict.py中用到
config["image_shape"] = (160, 160, 80)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  #train on the whole image

config["labels"] = (0, 1) # the label numbers on the input image

config["n_base_filters"] = 16 #16

config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["CT"]

config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 50  # cutoff the training after this many epochs

config["initial_learning_rate"] = 5e-4

#patience和learning_rate_epochs不共存。如果有learning_rate_epochs则patience不起作用
config['learning_rate_epochs'] = None #表示使用ReduceLROnPlateau

config['learning_rate_drop'] = 0.5
#默认使用reduce on plateau，learning_rate_epochs为None。
config['patience'] = 8*2000//config['train_epoch']
config["early_stop"] = 24*2000//config['train_epoch']


config["validation_split"] = 0.9  # portion of the data that will be used for training

config["flip"] = True  # augments the data by randomly flipping an axis during

config["permute"] = False  # data not a cube. cannot permute axis
config["distort"] = None  # switch to None if you want no distortion
#config["augment"] = config["flip"] or config["distort"]
config["augment"] = False

config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
#patch_shape=None的时候这个应该没用：
config["training_patch_start_offset"] = (0, 0, 0)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = False  # if True, then patches without any target will be skipped


#config["model_file"] = os.path.abspath("mil_multitask_model%s.h5"%config['full_version'])
config["model_file"] = "../models/mil_multitask_model%s.h5"%config['full_version']

if config['version']>16:
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/more_refined_data_160_80.h5")
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/more_refined_training_ids_160_80.pkl")
    config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/more_refined_validation_ids_160_80.pkl")


elif config['version']==80 or config['version']==81 or config['version']==82 or config['version']==84\
  or config['version']==86:
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80_ns.h5") #ns是no ischemia
    config['truth_file'] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80_naoshi.h5")
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl")
    config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/38713_valid_1000p_0_ischemia_false.pkl")
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl")
    config['test_dir'] = 'prediction/test_multitask%s'%config['full_version']

elif config['version']==83 or config['version']==85 or config['version']==87 or config['version']==88\
  or config['version']==89 or config['version']==90 or config['version']==91 or config['version']==92: #输入不再抠出brain，而是用head直接crop
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/all_head_data_160_80_ns.h5") #ns是no ischemia
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false.pkl")
    config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/38713_valid_1000p_0_ischemia_false.pkl")
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/38713_test_1000p_0_ischemia_false.pkl")
    config['test_dir'] = 'prediction/test_multitask%s'%config['full_version']
    if config['full_version']=='85.2': #85.1是不使用脑区异常loss的模型
        config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small1.pkl")
    if config['full_version']=='85.3':
        config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small2.pkl")
    if config['full_version']=='85.4':
        config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small3.pkl")
    if config['full_version']=='85.110': #小规模手标label数据集
        config["training_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small4_train.pkl")
        config["validation_file"] = os.path.abspath("/data/liuaohan/co-training/38713_training_17877p_0_ischemia_false_small4_valid.pkl")
    if config['full_version']=='85.111': #其他的脑区分割标注
        config['truth_file'] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80_27380689.h5")
    if config['full_version']=='85.112': #其他的脑区分割标注
        config['truth_file'] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80_28757653.h5")
    if config['full_version']=='85.113': #其他的脑区分割标注
        config['truth_file'] = os.path.abspath("/data/liuaohan/co-training/all_brain_data_160_80_28773576.h5")

#外部数据集测试：
if config['public_test']=='cq500_2':
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/%s_160_80.h5"%config['public_test'])
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/cq500_training.pkl") #cq500和cq500_2只是数据模态有些区别，id相同
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/cq500_test.pkl") 
    config['test_dir'] = 'prediction/test_%s_%d'%(config['public_test'], config['version'])
    config['rotate'] = False
if config['public_test']=='cq500' or config['public_test']=='cq500_3':
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/%s_160_80.h5"%config['public_test'])
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/cq500_training_full.pkl") #cq500和cq500_2只是数据模态有些区别，id相同
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/cq500_test_full.pkl") 
    config['test_dir'] = 'prediction/test_%s_%d'%(config['public_test'], config['version'])
    config['rotate'] = False

if config['public_test']=='prospective':
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/prospective_160_80.h5")
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/prospective_training.pkl")
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/prospective_good6_1500.pkl")
    
    config['test_dir'] = 'prediction/test_prospective_%s'%config['full_version']
    config['rotate'] = False
    config['test_label_file'] = 'all_refined_label_ischemia_false_naoshi_prospective5.h5' #5.30.2022尝试

if config['public_test']=='prospective2':
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/prospective2_160_80.h5")
    config["training_file"] = os.path.abspath("/data/liuaohan/co-training/prospective_training.pkl")
    config["test_file"] = os.path.abspath("/data/liuaohan/co-training/prospective_good7.pkl")
    
    config['test_dir'] = 'prediction/test_prospective2_%s'%config['full_version']
    config['rotate'] = False
    config['test_label_file'] = 'all_refined_label_ischemia_false_naoshi_prospective2_2.h5'
    

config['valid_dir'] = 'prediction/valid_multitask%s'%config['full_version']
config['training_dir'] = 'prediction/training_multitask%s'%config['full_version']

config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config['overwrite_model'] = False

def judge(dir, truth_name = 'refined_mask', public = False, save_truth=True):
    """
    
    """
    if not Path(dir+'/%s.nii.gz'%config["all_modalities"][0]).exists() \
    or (save_truth and not Path(dir+'/%s.nii.gz'%truth_name).exists()) \
    or not Path(dir+'/label.txt').exists():
        print('no nii', dir)
        return False
    if os.path.getsize(dir+'/%s.nii.gz'%config["all_modalities"][0])>10000000:
        return False #太大的nii，三维resize会出问题
    try:
        with open(dir+'/TransformParameters.0.txt') as fp:
            x = fp.read()
            if len(x)==0:
                print('no tp', dir)
                return False
        with open(dir+'/TransformParameters.1.txt') as fp:
            x = fp.read()
            if len(x)==0:
                print('no tp', dir)
                return False
    except:
        if save_truth: #保存脑区分割才需要配准结果
            print('open transform parameters error', dir)
            return False
    if public:
        return True
    try:
        with open(dir+'/good.txt') as fp: #其实prospective也应该检查时间间隔，但是第一次生成没这个。放到check_series里面修改
            x = fp.read()
            #if '2020.12.21' not in x: #1万规模数据集生成时使用的
            if 'nearest within 1 day' not in x: #全部数据集
                print('not good report', dir)
                return False
            #return True #7.19 当时这怎么写了个这个东西？我生成大规模数据集的时候这里有这个，导致判断序列长度等都没用了。
    except:
        print('cannot find good', dir)
        return False
    if not Path(dir+'/CT.json').exists():
        print('no CT json', dir)
        return False
    with open(dir+'/CT.json') as fp:
        temp = json.load(fp)
        if 'SliceTiming' not in temp: 
            #有的CT.json没有这个条目，但是还是有正常CT图像的
            #all_CT = list(Path(dir).glob('*CT*.png')) #使用CT不太行，因为有CT、CT_brain、CT_check
            all_CT = list(Path(dir).glob('*brain*.png')) #生成全部数据集
            if len(all_CT)<10:
                print('series too short', dir)
                return False
        elif len(temp['SliceTiming'])<10: #序列长度过短的，去除
            print('series too short', dir)
            return False
        if 'SeriesNumber' in temp and temp['SeriesNumber']==202: #5.28.2021生成全部数据集时加的
            print('bone window', dir) #噪声粒度比较大
            return False
    return True

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

def fetch_training_data_files(dirs, return_subject_ids=False, ori_data_file=None, truth_name='refined_mask',
                              public = False, save_truth=True, ab_seg_file=None):
    '''
    dirs下所有dir下的三层子序列作为数据集
    如果ori_data_file不为None，则不进行文件夹筛查，直接返回ori_data_file中subject_ids对应的序列
    （主要为了保证数据集样本、顺序完全一致的情况下对数据集进行调整）
    '''
    if ori_data_file is not None:
        file_opened = tables.open_file(ori_data_file, 'r')
        subject_ids = [x.decode('utf-8') for x in file_opened.root.subject_ids]
        file_opened.close()
        training_data_files = []
        print(len(subject_ids))
        for subject_dir in tqdm(subject_ids[:]):
            subject_files = list()
            for modality in config["training_modalities"] + [truth_name]: #truth
                subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
            subject_files.append(os.path.join(subject_dir, "label.txt"))
            training_data_files.append(tuple(subject_files))
        return training_data_files, subject_ids
    training_data_files = list() #[(序列1文件1，序列1文件2),(序列2文件1，序列2文件2)]
    subject_ids = list() #[序列1文件夹，序列2文件夹]
    all_paths = []
    bad = {} #标志序列有无错误，包括配准和中轴线矫正的筛查结果被标为3或4的。
    for dir in dirs:
        #如果已经有dir/series_angles.json，就直接用里面的路径，不用搜了
        #即使是使用series_link的序列，series_angles里也有，只不过预测值为[]。
        series_angles = None
        if Path(dir+'/series_angles.json').exists():
            with open(dir+'/series_angles.json', 'r') as fp:
                series_angles = json.load(fp)
            paths = [x for x in series_angles] #注意！7.20.2021发现，这里x应该是linux下的路径格式，即使用/。虽然没有问题，但是之前没有发现，提醒一下。
        else:
            paths = list(glob.glob(os.path.join(dir, "*", "*", "*")))
            print(dir, len(paths))
            #paths = list(glob.glob(os.path.join(dir, "*", "*"))) #rsna/full下暂时用两层
        all_paths.extend(paths)
        if public==False:
            with open(dir+'/registration_check_result.json', 'r') as fp:
                registration_check = json.load(fp)
            for x in registration_check:
                if registration_check[x]>=3:
                    bad[standard_path(x)] = True
            if Path(dir+'/correction_check_result.json').exists(): #后配准的序列，不包括fold_1。因为fold_1没有经过中轴线矫正
                with open(dir+'/correction_check_result.json', 'r') as fp:
                    correction_check = json.load(fp)
                for x in correction_check:
                    if correction_check[x]>=3:
                        bad[standard_path(x)] = True
            else: 
                if series_angles is not None:#指fold_1，此时series_angles应该是此文件夹的，否则为None
                    for x in series_angles:
                        angles = series_angles[x]
                        if np.abs(np.asarray(angles[3:len(angles)*4//5]).mean())>5:
                            bad[standard_path(x)] = True
                else: #那些我没有检查矫正情况的
                    pass
        else:
            try: #cq500
                with open(dir+'/best_series.json', 'r') as fp:
                    best_series = json.load(fp)
                best_series = [standard_path(x[0]) for x in best_series]
            except: #rsna
                best_series = [standard_path(path) for path in all_paths]
        pass
    print('bad',bad)
    print(len(all_paths))
    remove_cnt = 0
    for subject_dir in all_paths:
        #series_link，即此路径下实际没有东西，实际序列在另外一个目录（相同序列）下
        if Path(subject_dir+'/series_link.txt').exists():
            with open(subject_dir+'/series_link.txt', 'r') as fp:
                image_dir = fp.read().strip() #image_dir用于judge及bad判断此序列是否可用。去除文末换行
        else:
            image_dir = subject_dir
        if standard_path(image_dir) in bad:
            print('bad series', image_dir)
            remove_cnt += 1
            continue
        if public and standard_path(image_dir) not in best_series:
            print('not best series', image_dir)
            remove_cnt += 1
            continue
        if not judge(image_dir, truth_name, public, save_truth=save_truth):
            remove_cnt += 1
            continue
        reportID = str(Path(subject_dir).parent.parent.name) #因为一个series可能对应了多个report，所以只用其seriesID作为subjectID不够
        #subject_ids.append(reportID+'_'+os.path.basename(subject_dir)) #reportID+studyID
        subject_ids.append(subject_dir) #2020.12.23，现在数据可能生成于很多文件夹dirs，单纯用reportID+studyID无法定位文件夹。所以直接记录路径
        
        subject_files = list()
        for modality in config["training_modalities"] + ([truth_name] if save_truth else []): #truth
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        #lah added:
        
        subject_files.append(os.path.join(subject_dir, "label.txt"))
        training_data_files.append(tuple(subject_files))
    print('remove', remove_cnt)
    print(len(training_data_files))
    if return_subject_ids: #True
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite:
        o = input('confirm overwrite data file? y/n')
        if o!='y':
            return
    if overwrite or not os.path.exists(config["data_file"]):
        truth_only = False
        use_cv2_resize = False
        save_truth=True
        if config['version']==60:
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids = True, 
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5")
        if config['version']==70:
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids = True, 
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = 'refined_mask2')
        if config['version']==83:
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids = True, 
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = 'refined_mask3')
        if config['public_test']=='cq500':
            training_files, subject_ids = \
            fetch_training_data_files(['/home/liuaohan/pm-stroke/public/cq500'], return_subject_ids=True,
                                      truth_name = 'refined_mask2', public = True)
        if config['public_test']=='cq500_2':
            training_files, subject_ids = \
            fetch_training_data_files(['/home/liuaohan/pm-stroke/public/cq500'], return_subject_ids=True,
                                      truth_name = 'refined_mask2', public = True)
        if config['public_test']=='cq500_3':
            training_files, subject_ids = \
            fetch_training_data_files(['/home/liuaohan/pm-stroke/public/cq500'], return_subject_ids=True,
                                      truth_name = 'refined_mask2', public = True)
            use_cv2_resize = True
        if config["public_test"]=='prospective':
            training_files, subject_ids = \
            fetch_training_data_files(['/home/liuaohan/pm-stroke/prospective'], return_subject_ids=True,
                                      truth_name = '', public = True, save_truth=False)
            use_cv2_resize = True
            save_truth = False
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids, truth_dtype = np.uint32, organ_label = True, rotate = config['rotate'],
                           truth_only=truth_only, use_cv2_resize=use_cv2_resize, save_truth=save_truth)

    data_file_opened = open_data_file(config["data_file"])
    if config['truth_file'] and (overwrite or not os.path.exists(config["truth_file"])):
        if config['version']==80:
            truth_only = True
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids=True,
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = 'refined_mask3')
        if config['version']==85 and config['sub_version']==111:
            truth_only = True
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids=True,
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = '27380689/refined_mask3')
        if config['version']==85 and config['sub_version']==112:
            truth_only = True
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids=True,
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = '28757653/refined_mask3')
        if config['version']==85 and config['sub_version']==113:
            truth_only = True
            training_files, subject_ids = \
            fetch_training_data_files([], return_subject_ids=True,
                                      ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                      truth_name = '28773576/refined_mask3')
        
        write_data_to_file(training_files, config["truth_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids, truth_dtype = np.uint32, organ_label = True, rotate = config['rotate'],
                           truth_only=True)
    if config['ab_seg_file'] and (overwrite or not os.path.exists(config["ab_seg_file"])):
        training_files, subject_ids = \
        fetch_training_data_files([], return_subject_ids=True,
                                  ori_data_file = "/data/liuaohan/co-training/all_brain_data_160_80.h5",
                                  truth_name = 'abnormality_mask') #用truth，即脑区分割的地方来存储异常分割
        print('make abnormality segmentation file!', config["ab_seg_file"])
        write_data_to_file(training_files, config["ab_seg_file"], image_shape=config["image_shape"],
                       subject_ids=subject_ids, truth_dtype = np.uint32, organ_label = True, rotate = config['rotate'],
                       truth_only=True)
    
    if config['truth_file']:
        print('use truth file')
        truth_file = open_data_file(config['truth_file'])
    else:
        print('no truth file')
        truth_file = None
    if not config['overwrite_model'] and os.path.exists(config["model_file"]):
        model, w_seg = load_old_model(config["model_file"], load_loss_weight=config['load_loss_weight'],
                                      full_version = config['full_version'])
    else:
        # instantiate new model
        model, w_seg = mil_multitask_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                  version = config['version'], sub_version = config['sub_version'])
    try: #使用和数据h5文件不一样的label的时候，用label file
        print(config['label_file'])
        label_file_opened = open_data_file(config["label_file"])
        for i in range(5):
            print('use label file')
    except: #预训练，不用label file，直接用data文件里的label
        label_file_opened = None
        print('do not use label file')
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = \
    get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"], #目前看来patchshape=None时这个是没用的
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"],
        truth_dtype = np.uint32,
        organ_label = True,
        multi_task = True,
        truth_file = truth_file,
        label_file = label_file_opened,
        separate_mil = config['separate_mil'],
        output_shape = config['output_shape'],
        no_segmentation = config['no_segmentation'],
        train_epoch = config['train_epoch'],
        valid_epoch = config['valid_epoch'],
        triple_loss = config.get('triple_loss', False),
        classify_only = config.get('classify_only', False),
        dynamic_mask = config.get('dynamic_mask', True),)

    print('start training')
    # run training
    train_model(model=model, w_seg=w_seg, 
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                version = config['version'],
                full_version=config['full_version'],
                learning_rate_epochs=config['learning_rate_epochs'],
                config=config)
    data_file_opened.close()
    if truth_file is not None:
        truth_file.close()

if __name__ == "__main__":
    main(overwrite=config["overwrite"])
