import math
from functools import partial

import keras
from keras.callbacks import Callback
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.backend import eval
from keras.activations import softmax
from keras.layers import Lambda

import numpy as np

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient,
                            mil_loss, mil_max_loss, mil_loss_feature, mil_loss_feature_adv,
                            organ_segmentation_loss, weighted_binary_cross_entropy,
                            MIL_loss, zero_loss,
                            MIL_loss_ms, organ_segmentation_loss_ms,
                            binary_cross_entropy,
                            weighted_binary_cross_entropy2)

K.set_image_dim_ordering('th')


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    for i in range(2):
        print('')
    lr = initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))
    print('learning rate decay', lr)
    return lr

class MyCallback(Callback): #2021.3.10
    def __init__(self, w, version, full_version, config):
        self.w = w
        self.version = version
        self.full_version = full_version
        self.decay = 0.5
        if config['fix_epoch_size']:
            self.decay = math.pow(0.9, config['train_epoch']/4000)
        print('w seg decay', self.decay)
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if self.w is None:
            return
        new_value = K.get_value(self.w)*self.decay
        new_value = np.maximum(new_value, 1e-2)
        #new_value = K.maximum(new_value, 1e-3) 
        with open(('log/weight_%s.txt'%self.full_version), 'w') as fp:
            fp.write(str(new_value))
        
        K.set_value(self.w, new_value)

def scheduler(epoch, lr, floor = 3, drop = 0.6):
    if epoch<floor:
        return lr
    return lr * math.pow(drop, epoch-floor+1)

    
def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None, w_seg=None, version=None, full_version=None, config={}):
    callbacks = list()
    #save_weights_only=False
    if w_seg is not None:
        callbacks.append(MyCallback(w_seg, version, full_version, config))
    if full_version=='85.1':
        callbacks.append(ModelCheckpoint(model_file, monitor='val_pixel_pred3_loss', save_best_only=True))
    elif config['classify_only']:
        callbacks.append(ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True))
    else:
        callbacks.append(ModelCheckpoint(model_file, monitor='val_pixel_pred2_loss', save_best_only=True))
    
    if version is not None:
        logging_file = 'log/training_%s.log'%full_version
    callbacks.append(CSVLogger(logging_file, append=True))
    print('\n\n')
    print('learning rate epochs', learning_rate_epochs, learning_rate_drop)
    if learning_rate_epochs:
        #这种策略尝试用在fixed epoch size上。
        if learning_rate_epochs>1: #使用fixed epoch size的时候尝试设置为32
            callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
        else: #76.1使用此设置
            callbacks.append(LearningRateScheduler(partial(scheduler, lr = initial_learning_rate))) #目前这个用在76.0上是最好的
        
    else:
        if full_version=='85.1': #当不优化脑区级别异常loss时，使用其他指标判断：
            callbacks.append(ReduceLROnPlateau(monitor='val_pixel_pred3_loss', factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
        elif config['classify_only']:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
        else:
            callbacks.append(ReduceLROnPlateau(monitor='val_pixel_pred2_loss', factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        if full_version=='85.1':
            callbacks.append(EarlyStopping(monitor='val_pixel_pred3_loss', verbose=verbosity, patience=early_stopping_patience))
        elif config['classify_only']:
            callbacks.append(EarlyStopping(monitor='val_loss', verbose=verbosity, patience=early_stopping_patience))
        elif version==42 or version==77:
            callbacks.append(EarlyStopping(monitor='val_pixel_pred2_loss', verbose=verbosity, patience=early_stopping_patience))
        else:
            callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks

def get_callbacks_cq500(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None, w_seg=None, version=None, full_version=None, config={}):
    callbacks = list()
    #save_weights_only=False
    callbacks.append(ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True))
    if version is not None:
        logging_file = 'log/training_cq500_%d.log'%version
        callbacks.append(CSVLogger(logging_file, append=True))
    
    if learning_rate_epochs:
        print('fuck')
        #callbacks.append(LearningRateScheduler(partial(scheduler, lr = initial_learning_rate)))
    else:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=verbosity, patience=early_stopping_patience))
    return callbacks

def load_old_model(model_file, load_loss_weight = False, full_version = None):
    w_seg = None
    print('load loss weight', load_loss_weight)
    if load_loss_weight:
        with open('log/weight_%s.txt'%(full_version), 'r') as fp:
            w = float(fp.read())
            print('w_seg', w)
            w_seg = K.variable(w)
    print("Loading pre-trained model")
    custom_objects = {
             'keras':keras, #模型定义用了keras.backend.sum，不加这个会出错NameError: name 'keras' is not defined
             'loss':organ_segmentation_loss(depth=3, w=1.0, organs=[8,9], global_neg_loss=False, dynamic_mask=True),
              'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
              'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
              'weighted_dice_coefficient': weighted_dice_coefficient,
              'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
              'binary_cross_entropy': binary_cross_entropy,
              'softmax': softmax,
              'mil_loss': mil_loss,
              'mil_max_loss': mil_max_loss,
              'mil_loss_feature5': partial(mil_loss_feature, depth=5),
              'mil_loss_feature_adv5': partial(mil_loss_feature_adv, depth=5),
              'mil_loss_feature_adv5_minpool': partial(mil_loss_feature_adv, depth=5, minpool=True),
              'mil_loss_feature_adv5_refinedw': partial(mil_loss_feature_adv, depth=5, w=None),
              'mil_loss_feature_adv5_v4': partial(mil_loss_feature_adv, depth=5, w = None, organs=[8,9]),
              'mil_loss_feature_adv5_v5': partial(mil_loss_feature_adv, depth=5, w = None, organs=[15,16]),
              'organ_segmentation_loss5': partial(organ_segmentation_loss, depth=5, w=0.0), #此条目不适用于继续训练，因为不知道w值是多少
              'organ_segmentation_loss_v0': partial(organ_segmentation_loss, depth=5, w=0.0),
              'organ_segmentation_loss_v1': partial(organ_segmentation_loss, depth=5, w=0.0),
              'organ_segmentation_loss_v2': partial(organ_segmentation_loss, depth=5, w=0.01),
              'organ_segmentation_loss_v3': partial(organ_segmentation_loss, depth=1, w=0.0),
              'organ_segmentation_loss_v4': partial(organ_segmentation_loss, depth=1, w=0.1, organs=slice(8,9)),
              'organ_segmentation_loss_v5': partial(organ_segmentation_loss, depth=1, w=1.0, organs=slice(8,9)),
              'organ_segmentation_loss_v6': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(8,9)),
              'organ_segmentation_loss_v7': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(8,9), global_neg_loss=False),
              #'organ_segmentation_loss_v8': partial(organ_segmentation_loss, depth=3, w=1.0, organs=[8,9], global_neg_loss=False, dynamic_mask=True),
              #改用下面形式的loss函数试试：
              'organ_segmentation_loss_v8': organ_segmentation_loss(depth=3, w=1.0, organs=[8,9], global_neg_loss=False, dynamic_mask=True),
              
              'organ_segmentation_loss_v9': partial(organ_segmentation_loss, depth=3, w=0.1, organs=slice(8,10), global_neg_loss=False, dynamic_mask=True),
              # 注意用v10的时候f里面的segmentation loss要删掉
              'organ_segmentation_loss_v10':partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(8,9), global_neg_loss=False, dynamic_mask = False),
              'organ_segmentation_loss_v11': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(15,17), global_neg_loss=False, dynamic_mask=True),
              'organ_segmentation_loss_v12': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(15,16), global_neg_loss=False, dynamic_mask=True),
              'organ_segmentation_loss_v13': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(15,16), global_neg_loss=False, dynamic_mask=True, stop_gradient = False),
              'organ_segmentation_loss_v14': organ_segmentation_loss(depth=3, w=1.0, organs = [8,9], global_neg_loss=False, dynamic_mask=True),
              'organ_segmentation_loss_v15': organ_segmentation_loss(depth=3, w=1.0, organs = [8,9], global_neg_loss=False, dynamic_mask=True),
              'organ_segmentation_loss_v16': organ_segmentation_loss(depth=3, w=1.0, organs = [8,9], global_neg_loss=False, dynamic_mask=True, loss_type='dice'),
              'organ_segmentation_loss_v17': organ_segmentation_loss(depth=3, w=0.0, organs = [8,9], global_neg_loss=False, dynamic_mask=True, loss_type='dice'),
              'organ_segmentation_loss_v18': organ_segmentation_loss(depth=3, w=0.1, organs = [8,9], global_neg_loss=False, dynamic_mask=True, loss_type='dice'),
              'organ_segmentation_loss_v19': organ_segmentation_loss(depth=3, w=0.0, organs = list(range(20)), global_neg_loss=False,dynamic_mask = True, loss_type = 'dice'),
              'organ_segmentation_loss_v21': organ_segmentation_loss(depth=3, w=1.0, organs = [8,9], global_neg_loss=False, dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v22': organ_segmentation_loss(depth=3, w=1.0, organs = range(20), global_neg_loss=False, dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v23': organ_segmentation_loss(depth=3, w=1.0, organs = range(20), global_neg_loss=False, dynamic_mask=True, loss_type='dice', single_label=False, soft_loss=True),
              'organ_segmentation_loss_v24': organ_segmentation_loss(depth=3, w=1.0, organs = range(20), global_neg_loss=False, dynamic_mask=False, loss_type='none', single_label=False),
              'organ_segmentation_loss_v25': organ_segmentation_loss(depth=3, w=1.0, organs = range(20), global_neg_loss=False,dynamic_mask = False, loss_type = 'none', single_label=False, ignore_top = 0.8),
              'organ_segmentation_loss_v26': organ_segmentation_loss(depth=3, w=1.0, organs = range(20), global_neg_loss=False, dynamic_mask=True, loss_type='dice', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v27': organ_segmentation_loss(depth=3, w=0.0, w_seg=w_seg, organs = range(20), global_neg_loss=False,
                   dynamic_mask = True, loss_type = 'dice', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v28': organ_segmentation_loss(depth=3, w=0.0, organs = range(20), global_neg_loss=False, dynamic_mask=True, loss_type = 'dice', single_label=False),
              'organ_segmentation_loss_v29': organ_segmentation_loss(depth=3, w=0.0, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v30': organ_segmentation_loss(depth=3, w=0.0, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v31': organ_segmentation_loss(depth=3, w=0.0, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              
              'organ_segmentation_loss_v33':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='soft', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v34':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v35':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='soft', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v36':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v37':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False, stop_gradient = True),
              'organ_segmentation_loss_v39':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False, stop_gradient = False),
              'organ_segmentation_loss_v44':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v45':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v46':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v47':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v48':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v49':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,2,4,8,10,15,17,18], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v52':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v53':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v56':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              
              'organ_segmentation_loss_v60':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v61':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v70':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v72':organ_segmentation_loss_ms(scale=3, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v73':organ_segmentation_loss_ms(scale=3, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v74':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v75':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v76':organ_segmentation_loss_ms(scale=2, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v80':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v80':organ_segmentation_loss_ms(scale=2, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v81':organ_segmentation_loss_ms(scale=2, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_ms_v89':organ_segmentation_loss_ms(scale=1, depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              
              'organ_segmentation_loss_v86':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              'organ_segmentation_loss_v88':organ_segmentation_loss(depth=1, w=0.0, w_seg=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, loss_type='dice', single_label=False),
              
              'mil_loss_v27': MIL_loss(depth=3, organs = list(range(20)), global_neg_loss=False,single_label=False, stop_gradient=False, soft_loss=False, ignore_top=1.0),
              'mil_loss_v28': MIL_loss(depth=3, organs = list(range(20)), global_neg_loss=False,single_label=False, stop_gradient=True, soft_loss=False, r=10.0),
              'mil_loss_v29': MIL_loss(depth=3, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, soft_loss = False),
              'mil_loss_v30': MIL_loss(depth=3, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, soft_loss = False, strict_weight = True),
              'mil_loss_v31': MIL_loss(depth=3, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, soft_loss = False, symmetry_suppress = True),
              'mil_loss_v33': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, soft_loss = False, strict_weight=True),
              'mil_loss_v34': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, soft_loss = False, strict_weight=True),
              'mil_loss_v35': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, soft_loss = False),
              'mil_loss_v36': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, soft_loss = False),
              'mil_loss_v37': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = True, soft_loss = False),
              'mil_loss_v39': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, soft_loss = False),
              'mil_loss_v40': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], single_label = False, coarse_bag = True),
              'mil_loss_v41': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], single_label = False, coarse_bag = True, top_k=10, ignore_top=1.0),
              'mil_loss_v42_coarse': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], single_label = False, coarse_bag = True, top_k=10, ignore_top=1.0),
              'mil_loss_v42_organ': organ_segmentation_loss(depth=1, w=w_seg, w_seg=0.0, organs=[0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,dynamic_mask=True, stop_gradient=False, single_label=False, check_blank=False),
              'mil_loss_v44': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1),
              'mil_loss_v45': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, intensity_w=1.0),
              'mil_loss_v46': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, position_w=1.0),
              'mil_loss_v47': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, intensity_w=1.0, w_4=0.0),
              'mil_loss_v48': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True),
              'mil_loss_v49': MIL_loss(depth=1, organs = [0,2,4,8,10,15,17,18], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True),
              'mil_loss_v52': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True),
              'mil_loss_v53': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True),
              'mil_loss_v56': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, abnormals=slice(2,3)),
              
              'mil_loss_v60': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True),
              'mil_loss_v61': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, abnormals=slice(2,3), top_k=10),
              'mil_loss_v70': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=10),
              'mil_loss_ms_v72': MIL_loss_ms(scale=3,depth=1, use_pos_area=False, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=1),
              'mil_loss_ms_v73': MIL_loss_ms(scale=3,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=1),
              'mil_loss_v74': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = True, label_version=1, strict_weight=True, top_k=10, position_w=1.0),
              'mil_loss_v75': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=1.0),
              'mil_loss_ms_v76': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=10),
              'mil_loss_v77_organ': MIL_loss(depth=1, w=w_seg, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=1, strict_weight=True, top_k=10),
              'mil_loss_v77_coarse': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19], single_label = False, coarse_bag = True, ignore_top = 1.0, top_k = 10, label_version=1, strict_weight=True),
              'mil_loss_v80': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=10),
              'mil_loss_ms_v80': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=10),
              'mil_loss_ms_v81': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = True, label_version=2, strict_weight=True, top_k=10),
              'mil_loss_ms_v82_coarse': MIL_loss_ms(scale = 2, depth=1, use_pos_area=False, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], single_label=False, coarse_bag = True, label_version=2, strict_weight = True, top_k=10),
              'mil_loss_ms_v89': MIL_loss_ms(scale=1,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=1),
              'mil_loss_ms_v89_coarse': MIL_loss_ms(scale = 1, depth=1, use_pos_area=False, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], single_label=False, coarse_bag = True, label_version=2, strict_weight = True, top_k=1),
              
              'mil_loss_v86': MIL_loss(depth=1, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=10),
              'mil_loss_ms_v87': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=8, top_k_average=True),
              'mil_loss_ms_v87_coarse': MIL_loss_ms(scale = 2, depth=1, use_pos_area=False, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], single_label=False, coarse_bag = True, label_version=2, strict_weight = True, top_k=8, top_k_average=True),
              'mil_loss_ms_v88': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = True, label_version=2, strict_weight=True, top_k=10),
              'mil_loss_ms_v92': MIL_loss_ms(scale=2,depth=1, use_pos_area=True, organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19], global_neg_loss=False,single_label=False, stop_gradient = False, label_version=2, strict_weight=True, top_k=10, dynamic_mask=False),

              'organ_segmentation_loss_new_v8': partial(organ_segmentation_loss, depth=3, w=1.0, organs=[8,9], global_neg_loss=True, dynamic_mask=True),
              'weighted_binary_cross_entropy2':  weighted_binary_cross_entropy2([0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]),
              
              
              'organ_segmentation_loss_multiinput_v0': partial(organ_segmentation_loss, depth=3, w=1.0, organs=slice(8,9), global_neg_loss=True, dynamic_mask = True),
              'organ_segmentation_loss_multiinput_v2': partial(organ_segmentation_loss, depth=3, w = 1.0, organs = slice(8,9), global_neg_loss=True, dynamic_mask = False),
              'organ_segmentation_multiinput_v3': organ_segmentation_loss(depth=3, w = 1.0, organs = [8,9], global_neg_loss=False,dynamic_mask = False, loss_type = 'none'),
              'organ_segmentation_multiinput_v4': organ_segmentation_loss(depth=3, w = 1.0, organs = [8,9], global_neg_loss=False,dynamic_mask = False, loss_type = 'dice'),
              'organ_segmentation_multiinput_v5': organ_segmentation_loss(depth=5, w = 0.0, organs = [8], global_neg_loss=False,dynamic_mask = False, loss_type = 'none'),
              'organ_segmentation_multiinput_v6': organ_segmentation_loss(depth=5, w = 0.0, organs = [8], global_neg_loss=False,dynamic_mask = False, loss_type = 'none'),
              
              'weighted_binary_cross_entropy':weighted_binary_cross_entropy,
              'zero_loss': zero_loss}
        
    for x in custom_objects:
        custom_objects[x].__setattr__('__name__', x)
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        '''
        f = partial(organ_segmentation_loss, depth=3, w = 1.0, organs = [8,9], global_neg_loss=False,
                    dynamic_mask = True)
        f.__setattr__('__name__', 'organ_segmentation_loss_v8')
        global_loss = weighted_binary_cross_entropy
        global_loss.__setattr__('__name__', 'weighted_binary_cross_entropy')
        global_w = 0.0

        model = load_model(model_file, custom_objects=custom_objects, complie = False)
        model.compile(optimizer=optimizer(lr=initial_learning_rate), 
                      loss={'pixel_pred': f, 'global_pred': global_loss},
                      loss_weights={'pixel_pred': 1.0, 'global_pred': global_w})
        '''
        
        model = load_model(model_file, custom_objects=custom_objects)
        #print(model.summary())
        #print(model.get_config())
        print(model.loss)
        try:
            for x in model.loss:
                print(model.loss[x], type(model.loss[x]), getattr(model.loss[x], '__name__'), dir(model.loss[x]))
        except:
            pass
        print('learning rate', eval(model.optimizer.lr))
        
        return model, w_seg
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                w_seg=None, initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None, version=None, full_version = None, cq500=False,
                config = {}):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    if not cq500:
        f = get_callbacks
    else:
        f = get_callbacks_cq500
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=f(model_file,
                                    initial_learning_rate=initial_learning_rate,
                                    learning_rate_drop=learning_rate_drop,
                                    learning_rate_epochs=learning_rate_epochs,
                                    learning_rate_patience=learning_rate_patience,
                                    early_stopping_patience=early_stopping_patience,
                                    w_seg = w_seg, version=version,
                                    full_version=full_version, config=config),
                        #use_multiprocessing=True,
                        workers=0,
                        )
