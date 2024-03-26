# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:44:30 2021

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
import argparse
import copy

project_path = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from unet3d.data import write_cq500_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model.cls_model import cls_model
from unet3d.training import load_old_model, train_model


config = dict()

def dfs(grid, pars, i, now, results):
    if i>=len(pars):
        results.append({pars[x]:now[x] for x in range(len(pars))})
        return
    p = pars[i]
    for x in grid[p]:
        now[i] = x
        dfs(grid, pars, i+1, now, results)

def grid_search():
    grid = {'nc':[8,16,32], 'depth':[3,4,5], 'input':[2,3],
            'flip':[False, True]}
    pars = [x for x in grid]
    print(pars)
    results = []
    dfs(grid, pars, 0, [None for x in grid], results)
    print(len(results))
    results = {i+100:results[i] for i in range(len(results))}
    return results
    
def make_config(args):
    config['version'] = 11
    if args.v is not None:
        config['version'] = args.v
    config['fold'] = args.f
    config['upstream_version'] = 76
    if config['version']>=200:
        config['upstream_version'] = 80
    if config['version']>=250:
        config['upstream_version'] = 82
    if config['version']==300 or config['version']==301:
        config['upstream_version'] = 82
    if config['version']>=350:
        config['upstream_version'] = 80
    hp = {0:{'nc':4, 'depth':4, 'input':0, 'flip':False},
          1:{'nc':4, 'depth':3, 'input':0, 'flip':False},
          2:{'nc':4, 'depth':2, 'input':0, 'flip':False},
          3:{'nc':8, 'depth':4, 'input':0, 'flip':False},
          4:{'nc':4, 'depth':4, 'input':1, 'flip':False},
          5:{'nc':4, 'depth':3, 'input':1, 'flip':False},
          6:{'nc':4, 'depth':2, 'input':1, 'flip':False},
          7:{'nc':8, 'depth':4, 'input':1, 'flip':False},
          8:{'nc':8, 'depth':3, 'input':1, 'flip':False},
          9:{'nc':8, 'depth':2, 'input':1, 'flip':False},
          
          10:{'nc':4, 'depth':4, 'input':2, 'flip':False},
          11:{'nc':8, 'depth':4, 'input':2, 'flip':False},
          12:{'nc':4, 'depth':3, 'input':2, 'flip':False},
          13:{'nc':4, 'depth':2, 'input':2, 'flip':False},
          14:{'nc':8, 'depth':3, 'input':2, 'flip':False},
          15:{'nc':8, 'depth':2, 'input':2, 'flip':False},
          16:{'nc':16, 'depth':4, 'input':2, 'flip':False},
          17:{'nc':16, 'depth':5, 'input':2, 'flip':False},
          18:{'nc':16, 'depth':4, 'input':2, 'flip':True},
          19:{'nc':32, 'depth':4, 'input':2, 'flip':True, 'rotate':True},
          
          20:{'nc':16, 'depth':5, 'input':3, 'flip':False},
          21:{'nc':16, 'depth':5, 'input':3, 'flip':True},
          30:{'nc':16, 'depth':4, 'input':4, 'flip':True},
          31:{'nc':16, 'depth':4, 'input':4, 'flip':True, 'rotate':True},
          32:{'nc':16, 'depth':4, 'input':4, 'flip':True},
          33:{'nc':16, 'depth':4, 'input':4, 'flip':True, 'bn':True},
          34:{'nc':16, 'depth':4, 'input':4, 'flip':True, 'rotate':True},
          35:{'nc':16, 'depth':4, 'input':2, 'flip':True},
          
          40:{'nc':16, 'depth':4, 'input':5, 'flip':True},
          #从200开始是用加入脑室的80输入
          200:{'nc':16, 'depth':4, 'input':0, 'flip':True},
          201:{'nc':16, 'depth':4, 'input':1, 'flip':True},
          202:{'nc':16, 'depth':4, 'input':2, 'flip':True},
          203:{'nc':16, 'depth':4, 'input':3, 'flip':True},
          204:{'nc':16, 'depth':4, 'input':4, 'flip':True},
          205:{'nc':16, 'depth':4, 'input':5, 'flip':True},
          210:{'nc':16, 'depth':4, 'input':0, 'flip':True, 'seed':0},
          211:{'nc':16, 'depth':4, 'input':1, 'flip':True, 'seed':0},
          212:{'nc':16, 'depth':4, 'input':2, 'flip':True, 'seed':0},
          213:{'nc':16, 'depth':4, 'input':3, 'flip':True, 'seed':0},
          214:{'nc':16, 'depth':4, 'input':4, 'flip':True, 'seed':0},
          215:{'nc':16, 'depth':4, 'input':5, 'flip':True, 'seed':0},
          #从250开始是用82的输入
          250:{'nc':16, 'depth':4, 'input':0, 'flip':True},
          251:{'nc':16, 'depth':4, 'input':1, 'flip':True},
          252:{'nc':16, 'depth':4, 'input':2, 'flip':True},
          253:{'nc':16, 'depth':4, 'input':3, 'flip':True},
          254:{'nc':16, 'depth':4, 'input':4, 'flip':True},
          255:{'nc':16, 'depth':4, 'input':5, 'flip':True},
          260:{'nc':16, 'depth':4, 'input':0, 'flip':True, 'seed':0},
          261:{'nc':16, 'depth':4, 'input':1, 'flip':True, 'seed':0},
          262:{'nc':16, 'depth':4, 'input':2, 'flip':True, 'seed':0},
          263:{'nc':16, 'depth':4, 'input':3, 'flip':True, 'seed':0},
          264:{'nc':16, 'depth':4, 'input':4, 'flip':True, 'seed':0},
          265:{'nc':16, 'depth':4, 'input':5, 'flip':True, 'seed':0},
          #300开始是只输入CT，有些用upstream模型做finetune
          300:{'nc':16, 'depth':5, 'input':6, 'flip':True, 'seed':0, 'finetune_upstream':True},
          301:{'nc':16, 'depth':4, 'input':6, 'flip':True, 'seed':0},
          #350开始是在rsna上的模型
          350:{'nc':16, 'depth':4, 'input':0, 'flip':False, 'dataset':'rsna', 'n_labels':5},
          351:{'nc':16, 'depth':4, 'input':7, 'flip':False, 'dataset':'rsna', 'n_labels':5},
          352:{'nc':16, 'depth':4, 'input':8, 'flip':False, 'dataset':'rsna', 'n_labels':5},
          360:{'nc':16, 'depth':4, 'input':9, 'flip':False, 'dataset':'rsna', 'n_labels':2},
          361:{'nc':16, 'depth':4, 'input':10, 'flip':False, 'dataset':'rsna', 'n_labels':2},
          362:{'nc':16, 'depth':4, 'input':10, 'flip':True, 'dataset':'rsna', 'n_labels':2}, #flip没不flip的好，可能是因为flip导致保存模型的随机性
          370:{'nc':16, 'depth':4, 'input':10, 'flip':False, 'dataset':'rsna4', 'n_labels':2},
          }
    hp.update(grid_search())
    print(len(hp))
    config["n_base_filters"] = hp[config['version']]['nc']
    config['depth'] = hp[config['version']]['depth']
    config['input_version'] = hp[config['version']]['input']
    config['flip'] = hp[config['version']]['flip']
    config['rotate'] = hp[config['version']].get('rotate', False)
    config['bn'] = hp[config['version']].get('bn', False) #batch normalize效果很不好，验证集和训练集的loss会差很多
    config['follow_upstream_structure'] = hp[config['version']].get('finetune_upstream', False)
    config['dataset'] = hp[config['version']].get('dataset', 'cq500')
    config["n_labels"] = hp[config['version']].get('n_labels', 4)
    
    if hp[config['version']].get('seed', 10)!=10:
        config['data_seed'] = '_%d'%hp[config['version']].get('seed')
    else:
        config['data_seed'] = ''
    input_par = {0:{'shape': (5,160,160,80), 'normalize':True},
                 1:{'shape': (5,160,160,80), 'normalize':True},
                 2:{'shape': (5,80,80,40), 'normalize':True},
                 3:{'shape': (5,80,80,40), 'normalize':False},
                 4:{'shape': (4,80,80,40), 'normalize':False},
                 5:{'shape': (4,80,80,40), 'normalize':False},
                 6:{'shape': (1,160,160,80), 'normalize':True},
                 7:{'shape': (5,80,80,40), 'normalize':True}, #0的缩小版
                 8:{'shape': (6,80,80,40), 'normalize':True},
                 9:{'shape': (5,80,80,40), 'normalize':True}, #和7模型输入一样，但是输出不一样，预测两个标签
                 10:{'shape': (3,80,80,40), 'normalize':True},
                 }
    config["image_shape"] = input_par[config['input_version']]['shape'][1:]
    config["input_shape"] = input_par[config['input_version']]['shape']
    config['n_channels'] = input_par[config['input_version']]['shape'][0]
    
    config['normalize'] = input_par[config['input_version']]['normalize']
    
    
    
    config["batch_size"] = 8
    config["validation_batch_size"] = 8
    config["n_epochs"] = 200  # cutoff the training after this many epochs
    config["patience"] = 2  # learning rate will be reduced after this many epochs if the validation loss is not improving
    config["early_stop"] = 7  # training will be stopped after this many epochs without the validation loss improving
    if config['input_version']==6:
        config["patience"] = 3
        config["early_stop"] = 11
    config["initial_learning_rate"] = 1e-4
    config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    config["validation_split"] = 0.8  # portion of the data that will be used for training
    
    config["model_file"] = os.path.abspath("models/%s_model%d_%d.h5"%(config['dataset'],config['version'], config['fold']))
    config["data_file"] = os.path.abspath("/data/liuaohan/co-training/%s_data%d_%d.h5"%(config['dataset'],config['upstream_version'],config['input_version']))
    #config["training_file"] = os.path.abspath("cq500_training_f%d.pkl"%config['fold'])
    #config["validation_file"] = os.path.abspath("cq500_validation_f%d.pkl"%config['fold'])
    
    
    config["training_file"] = os.path.abspath("%s_training_f10%s_%d.pkl"%(config['dataset'],config['data_seed'],config['fold']))
    config["validation_file"] = os.path.abspath("%s_validation_f10%s_%d.pkl"%(config['dataset'],config['data_seed'],config['fold']))
    config["test_file"] = os.path.abspath("%s_test_f10%s_%d.pkl"%(config['dataset'],config['data_seed'],config['fold']))
    
    config["overwrite_data"] = False  # If True, will previous files. If False, will use previously written files.
    config["overwrite_model"] = False  # If True, will previous files. If False, will use previously written files.
    
    config["training_dir"] = os.path.abspath("prediction/training_%s_downstream%d_%d"%(config['dataset'],config['version'], config['fold']))
    config["valid_dir"] = os.path.abspath("prediction/valid_%s_downstream%d_%d"%(config['dataset'],config['version'], config['fold']))
    config["test_dir"] = os.path.abspath("prediction/test_%s_downstream%d_%d"%(config['dataset'],config['version'], config['fold']))
    
def judge(dir):
    try:
        with open(dir+'/angle.json', 'r') as fp:
            angle = json.load(fp)
        if len(angle)!=2:
            return None
        if max([x[1] for x in angle[1]])-min([x[1] for x in angle[1]])>10:
            print('angle difference too big', angle)
            return None
        return angle[0]
    except:
        return None

def fetch_training_data_files(label_file):
    training_data_files = list()
    subject_ids = []
    label_paths = list(Path('prediction/test_%s_%d'%(config['dataset'],config['upstream_version'])).rglob('*%s.txt'%label_file))
    for label_path in tqdm(label_paths[:]):
        subject_dir = label_path.parent
        pred_paths = list(Path(subject_dir).glob('*prediction*.nii.gz'))

        training_data_files.append((str(subject_dir)+'/data_CT_brain.nii.gz', str(pred_paths[-1]), 
                                    str(subject_dir)+'/%s.txt'%label_file))
        subject_ids.append(str(subject_dir))
    print(len(training_data_files))
    return training_data_files, subject_ids


def main():
    # convert input images into an hdf5 file
    if config["overwrite_data"] or not os.path.exists(config["data_file"]):
        if config['dataset']=='cq500':
            training_files, subject_ids = fetch_training_data_files(label_file='real_label')
        else:
            training_files, subject_ids = fetch_training_data_files(label_file='label')
        write_cq500_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids = subject_ids, version=config['input_version'], normalize=config['normalize'],
                           n_channels = config['n_channels'], n_labels = config['n_labels'])
    
    data_file_opened = open_data_file(config["data_file"])
    print(len(data_file_opened.root.data))
    print(len(data_file_opened.root.truth))
    print(len(data_file_opened.root.label))
    
    if not config["overwrite_model"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = cls_model(input_shape=config["input_shape"], depth = config['depth'], n_labels=config["n_labels"],
                          initial_learning_rate=config["initial_learning_rate"],
                          n_base_filters=config["n_base_filters"],
                          version = config['version'], batch_normalization=config['bn'],
                          follow_upstream_structure=config['follow_upstream_structure'])
        if config['follow_upstream_structure']:
            print('\n\n\n use upstream to finetune model', config['upstream_version'])
            model.summary()
            model.load_weights('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/mil_multitask_model%d.h5'%config['upstream_version'], by_name=True)
    
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=False, #重新生成数据h5文件后，不希望变动训练集和测试集划分（因为模型已经在原来训练集上训过了）
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        skip_blank = False, #或者每次让y是1而不是0
        n_labels=1,
        labels=None,
        organ_label=True,
        classify=True,
        augment_flip=config['flip'],
        rotate = config['rotate'],
        version=config['input_version'])

    # run training
    train_model(model=model,
                version=config['version'],
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
                cq500=True)
    data_file_opened.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=int)
    parser.add_argument('-v', type=int)
    parser.add_argument('-temp', action='store_true')
    args = parser.parse_args()
    if args.temp:
        model = cls_model((1,160,160,80), n_labels=4, depth=5, n_base_filters=16,
                          follow_upstream_structure=True)
        model.summary()
        pre_model,_ = load_old_model('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/mil_multitask_model82.h5',
                                   load_loss_weight = True, full_version = '82')
        pre_model.summary()
        model.load_weights('/home/liuaohan/pm-stroke/project/3DUnetCNN-master/brats/mil_multitask_model82.h5', by_name=True)
    else:
        if args.f>=0:
            make_config(args)
            main()
        else:
            nfolds = -args.f
            for i in range(0, nfolds):
                args.f = i
                make_config(args)
                main()
            pass
    
