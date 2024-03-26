# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:18:29 2020

@author: Admin
"""

from functools import partial

import keras
import tensorflow as tf
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, Lambda
from keras.activations import softmax
import keras.layers
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate, get_up_convolution
from ..metrics import mil_loss_feature
from ..metrics import mil_loss_feature_adv
from ..metrics import organ_segmentation_loss, organ_segmentation_loss_ms, MIL_loss, MIL_loss_ms
from ..metrics import weighted_binary_cross_entropy, weighted_binary_cross_entropy2, zero_loss

import numpy as np


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)
#create_convolution_block = partial(create_convolution_block, activation=LeakyReLU)


def mil_multitask_model(input_shape=(1, 160, 160, 80), batch_size = 1, n_base_filters=16, depth=5, up_depth=0, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=2, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function = mil_loss_feature, activation_name="sigmoid",
                      version = None, conv2d = False, sub_version = 0):
    """
    Build the Image Analyzer model.
    The backbone of the model is similar to that of Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    keras.layers.Input
    #keras.backend.set_image_data_format('channels_first')
    #assert up_depth<depth, 'up_depth should be less than depth'
    
    if version==32 or version==53 or version==55 or version==56 or version==70\
      or version==71 or version==72 or version==73 or version==74 or version==75\
      or version==76 or version==77 or version==80 or version==81 or version==82\
      or version==83 or version==84 or version==85 or version==86 or version==87\
      or version==88 or version==89 or version==92:
        up_depth = 3
    
    if version>=44:
        abnormal_channel = 5
    
    remain_z = 0
    
    multi_scale = 2
    
    triple_loss = True

    kernel = (3,3,3) #默认卷积核
    pool_size = (2,2,2)
    if conv2d:
        kernel = (3,3,1)
        pool_size = (2,2,1)

    inputs = Input(input_shape) 
    current_layer = inputs
    levels = []
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), kernel=kernel)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, kernel=kernel)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    feature_map = current_layer
    print('feature size', feature_map.get_shape())
    
    multi_scale_pred = []
    for layer_depth in range(depth-2, depth-2-up_depth, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=False,kernel=kernel)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer, batch_normalization=False, kernel=kernel)
        if multi_scale>1 and layer_depth-(depth-2-up_depth)<=multi_scale: 
            temp_ab = Conv3D(abnormal_channel, (1, 1, 1))(current_layer)
            temp_ab = Lambda(lambda x:softmax(x, axis=1))(temp_ab)
            temp_organ = Conv3D(20, (1, 1, 1))(current_layer)
            temp_organ = Activation("sigmoid")(temp_organ)
            temp_pred = keras.layers.Concatenate(1)([temp_organ, temp_ab])
            multi_scale_pred.append(keras.layers.ZeroPadding3D( \
                    padding=((0,input_shape[1]//(2**(depth-up_depth-1)) - input_shape[1]//(2**layer_depth)),
                             (0,input_shape[2]//(2**(depth-up_depth-1)) - input_shape[2]//(2**layer_depth)),
                             (0,input_shape[3]//(2**(depth-up_depth-1)) - input_shape[3]//(2**layer_depth))))(temp_pred))
    
    if multi_scale>1:
        for i,x in enumerate(multi_scale_pred):
            print('scale ', i, x.get_shape()) #(?,25,80,80,40)
        pixel_pred = keras.layers.Concatenate(1, name = 'pixel_pred')(multi_scale_pred)
        pixel_pred2 = keras.layers.Concatenate(1, name = 'pixel_pred2')(multi_scale_pred)
        if triple_loss:
            pixel_pred3 = keras.layers.Concatenate(1, name='pixel_pred3')(multi_scale_pred)
        print(pixel_pred.shape)
        #keras.layers.Concatenate(1, name='pixel_pred')([organ_pred, ab_pred])
    else:
        current_layer_2 = current_layer
        
        if abnormal_channel==1:
            pixel_pred = Conv3D(21, (1, 1, 1))(current_layer) 
            pixel_pred = Activation("sigmoid", name='pixel_pred')(pixel_pred)
        else:
            print(type(current_layer))
            ab_pred = Conv3D(abnormal_channel, (1, 1, 1))(current_layer)
            ab_pred = Lambda(lambda x:softmax(x, axis=1))(ab_pred)
            print(type(ab_pred))
            organ_pred = Conv3D(20, (1, 1, 1))(current_layer_2) 
            organ_pred = Activation("sigmoid", name='organ_pred')(organ_pred)
            pixel_pred = keras.layers.Concatenate(1, name='pixel_pred')([organ_pred, ab_pred])
            pixel_pred2 = keras.layers.Concatenate(1, name='pixel_pred2')([organ_pred, ab_pred])
            if triple_loss:
                pixel_pred3 = keras.layers.Concatenate(1, name='pixel_pred3')([organ_pred, ab_pred])
    print('output size', pixel_pred.get_shape())
    
    
    global_feature = keras.layers.GlobalAveragePooling3D(data_format='channels_first')(feature_map)
    global_pred = keras.layers.Dense(20*abnormal_channel)(global_feature)
    if abnormal_channel!=1:
        global_pred = keras.layers.Reshape((20, abnormal_channel))(global_pred) 
    global_pred = Activation("sigmoid", name='global_pred')(global_pred)
    
    global_loss = 'binary_crossentropy'
    w_seg = None
    seg_weight = 1.0
    
    if version==82 or version==85:
        assert depth==5 and up_depth==3
        w_seg = keras.backend.variable(1.0)
        organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
        
        f = organ_segmentation_loss_ms(scale = 2, depth=1, w = 0.0, w_seg = w_seg, 
                                    organs = organs, global_neg_loss=False,
                    dynamic_mask = True, loss_type = 'dice', single_label=False)
        f.__setattr__('__name__', 'organ_segmentation_loss_ms_v80') #就用80
        
        mil_loss = MIL_loss_ms(scale = 2, depth=1, use_pos_area=True, organs = organs, global_neg_loss=False,single_label=False, 
                            stop_gradient = False, label_version=2, strict_weight = True, top_k=10)
        mil_loss.__setattr__('__name__', 'mil_loss_ms_v80')
        mil_w = 0.2
        
        mil_loss_coarse = MIL_loss_ms(scale = 2, depth=1, use_pos_area=False, organs = organs, single_label=False, 
                            coarse_bag = True, label_version=2, strict_weight = True, top_k=10) #global_neg_loss和stop_gradient对coarse无影响
        mil_loss_coarse.__setattr__('__name__', 'mil_loss_ms_v82_coarse')
        mil_w_coarse = 0.02

    elif version==89:
        #和85的区别是single scale
        assert depth==5 and up_depth==3
        w_seg = keras.backend.variable(1.0) 
        organs = [0,1,2,3,4,5,8,9,10,11,13,14,15,16,17,18,19]
        f = organ_segmentation_loss_ms(scale = 1, depth=1, w = 0.0, w_seg = w_seg, 
                                    organs = organs, global_neg_loss=False,
                    dynamic_mask = True, loss_type = 'dice', single_label=False)
        f.__setattr__('__name__', 'organ_segmentation_loss_ms_v89') 
        
        mil_loss = MIL_loss_ms(scale = 1, depth=1, use_pos_area=True, organs = organs, global_neg_loss=False,single_label=False, 
                            stop_gradient = False, label_version=2, strict_weight = True, top_k=1)
        mil_loss.__setattr__('__name__', 'mil_loss_ms_v89')
        mil_w = 0.2
        
        mil_loss_coarse = MIL_loss_ms(scale = 1, depth=1, use_pos_area=False, organs = organs, single_label=False, 
                            coarse_bag = True, label_version=2, strict_weight = True, top_k=1) #global_neg_loss和stop_gradient对coarse无影响
        mil_loss_coarse.__setattr__('__name__', 'mil_loss_ms_v89_coarse')

        mil_w_coarse = 0.02
    

    if version==82 or version==85 or version==87 or version==88 or version==89\
      or version==92:
        model = Model(inputs=inputs, outputs=[pixel_pred, pixel_pred2, pixel_pred3])
        model.compile(optimizer=optimizer(lr=initial_learning_rate), 
                      loss={'pixel_pred': f, 'pixel_pred2': mil_loss, 'pixel_pred3': mil_loss_coarse},
                      loss_weights={'pixel_pred': seg_weight, 'pixel_pred2': mil_w, 'pixel_pred3': mil_w_coarse})
    return model, w_seg


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



