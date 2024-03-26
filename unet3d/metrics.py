from functools import partial

from keras import backend as K
from keras.layers import MaxPooling3D, UpSampling3D
import keras
import tensorflow as tf
import numpy as np
import math


"""
我感觉loss里面的y_true应该是generator（训练代码中的train_generator和validation_generator）
决定的，在generator.py中yield了x和y，其中y是不是就是这里的y_true？但是generator生成的
是array，到这里怎么变成tensor的呢？
y_pred就是model的outputs吧？
"""
def mil_loss(y_true, y_pred, smooth = 1e-5):
    """
    y_pred:每个点是此点有病灶的概率
    smooth是因为loss刚开始是inf。
    """
    #计算正样本loss：
    for i in range(10):
        print()
    print(type(y_true), type(y_pred))
    print('true', K.int_shape(y_true))
    print('pred', K.int_shape(y_pred))
    print('true', y_true.get_shape())
    print('pred', y_pred.get_shape())
    print('multiply', (y_pred * y_true).get_shape())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(y_true_f.get_shape())
    print(y_pred_f.get_shape())
    print((y_pred_f*y_true_f).get_shape())
    
    y_true = y_true[:,1,:,:,:]
    print('true', y_true.get_shape())
    #我无论pred输出1个channel还是2个channel这里都不报错？？？都是11.5128
    #到底是什么鬼？y_true应该是几个channel的？
    pos = 1 - K.prod((1 - y_pred * y_true)) #mask为1处预测有异常的概率，最佳应为1
    neg = K.prod(1 - y_pred * (1 - y_true)) #mask为0处预测无异常的概率，最佳应为1
    return -10*K.log(pos+smooth)-K.log(neg+smooth) #两个概率和真实标签（1）求交叉熵，再加权求和

def mil_max_loss(y_true, y_pred, smooth = 1e-5):
    """
    y_pred:每个点是此点有病灶的概率
    smooth是因为loss刚开始是inf。
    """
    #计算正样本loss：
    for i in range(10):
        print()
    print(type(y_true), type(y_pred))
    print('ori true', K.int_shape(y_true))
    print('pred', K.int_shape(y_pred))
    print('true', y_true.get_shape())
    print('pred', y_pred.get_shape())
    print('multiply', (y_pred * y_true).get_shape())

    
    y_true = y_true[:,1:,:,:,:]
    print('true', y_true.get_shape())
    #我无论pred输出1个channel还是2个channel这里都不报错？？？都是11.5128
    #到底是什么鬼？y_true应该是几个channel的？
    pos = K.max(y_pred * y_true)#mask为1处预测有异常的概率，最佳应为1
    neg = 1 - K.max(y_pred * (1 - y_true))#mask为0处预测无异常的概率，最佳应为1
    blank = K.equal(K.sum(y_true), 0)
    loss = K.switch(blank, -K.log(neg+smooth), -K.log(pos+smooth)-K.log(neg+smooth))
    return loss #两个概率和真实标签（1）求交叉熵，再加权求和
'''
def mil_loss_feature(y_true, y_pred, depth):
    """
    y_pred:每个点是此点有病灶的概率
    smooth是因为loss刚开始是inf。
    """
    #计算正样本loss：
    label = y_true[:,2,0,0,0:20] #generator中将label和mask合并到y中
    #限制batch必须为1
    #assert y_pred.get_shape()[0]==1 #不能这样assert，因为是?
    #current_layer = y_true[:,1:,:,:,:] #不直接用1是为了保持维度，依旧是5维tensor，和pred相同
    pos_loss = tf.Variable(0, trainable=False, dtype = tf.float32)
    shape = list(y_pred.get_shape())
    shape[1] = 1
    pos_map = None
    print('y_true', y_true) #tf.float32
    true = tf.cast(y_true, tf.uint32) #位运算需要类型为int
    for i in range(20):
        current_layer = tf.bitwise.bitwise_and(tf.bitwise.right_shift(true[:,1:,:,:,:], i), 1)
        current_layer = tf.cast(current_layer, tf.float32) #maxpooling又需要输入为float类型
        for level_number in range(depth - 2, -1, -1):
            current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
        pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
        pos_loss = pos_loss - label[0,i] * K.log(pos + 1e-30)
        current_layer = tf.cast(current_layer, tf.uint32)
        if pos_map is None:
            pos_map = current_layer
        else:
            pos_map = tf.bitwise.bitwise_or(pos_map, current_layer)
    print('pred', y_pred.get_shape())
    print('true', current_layer.get_shape())
    #for level_number in range(depth - 2, -1, -1):
    #    current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
    
    print('pred type', y_pred)
    #blank = K.equal(K.sum(current_layer), 0)
    """
    y_pred = K.pow(y_pred*0.8, 10)
    pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    neg = K.prod(1 - y_pred * (1 - current_layer)) #mask为0处预测无异常的概率，最佳应为1
    loss = K.switch(blank, -K.log(neg+1e-30), -K.log(pos+1e-30)-K.log(neg+1e-30))
    """
    #pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    #pos_loss = -K.log(pos + 1e-30)
    pos_map = tf.cast(pos_map, tf.float32)
    #temp = 1 - pos_map
    neg_loss = K.mean(-K.log(1 - y_pred * (1 - pos_map) + 1e-30))
    #loss = K.switch(blank, neg_loss, pos_loss + neg_loss)
    loss = pos_loss + neg_loss
    return loss #两个概率和真实标签（1）求交叉熵，再加权求和
'''
def mil_loss_feature(y_true, y_pred, depth):
    """
    y_pred:每个点是此点有病灶的概率
    """
    label = y_true[:,20,0,0,0:20]
    #限制batch必须为1
    #assert y_pred.get_shape()[0]==1 #不能这样assert，因为是?
    pos_loss = 0.0
    
    pos_map = None
    print('y_true', y_true) #tf.float32
    y_pred = y_pred
    
    for i in range(20):
        current_layer = y_true[:,i:i+1,:,:,:]
        for level_number in range(depth - 2, -1, -1):
            current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
        pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
        blank = K.equal(K.sum(current_layer), 0)
        pos_loss = pos_loss - K.switch(blank, 0.0, label[0,i] * K.log(pos + 1e-30))
        #current_layer = K.equal(current_layer, 1)
        
        if pos_map is None:
            pos_map = current_layer * label[0,i]
        else:
            pos_map = tf.maximum(pos_map, current_layer * label[0,i])
        del current_layer

    #for level_number in range(depth - 2, -1, -1):
    #    current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
    
    print('pred type', y_pred)
    #blank = K.equal(K.sum(current_layer), 0)
    """
    y_pred = K.pow(y_pred*0.8, 10)
    pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    neg = K.prod(1 - y_pred * (1 - current_layer)) #mask为0处预测无异常的概率，最佳应为1
    loss = K.switch(blank, -K.log(neg+1e-30), -K.log(pos+1e-30)-K.log(neg+1e-30))
    """
    #pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    #pos_loss = -K.log(pos + 1e-30)
    
    #temp = 1 - pos_map
    neg_loss = K.mean(-K.log(1 - y_pred * (1 - pos_map) + 1e-30))
    #loss = K.switch(blank, neg_loss, pos_loss + neg_loss)
    loss = pos_loss + neg_loss
    return loss #两个概率和真实标签（1）求交叉熵，再加权求和

def organ_segmentation_loss_ms(scale, depth, w = 0.0, w_seg = 1.0, organs = slice(0,20), global_neg_loss = True,
                              dynamic_mask = False, stop_gradient = True, loss_type = 'binary',
                              single_label = True, soft_loss = False, ignore_top = 1.0, check_blank = True
                              ):
    def loss(y_true, y_pred):
        print('multi scale pred shape', y_pred.get_shape()) #(3,25,80,80,40)
        print('multi scale truth shape', y_true.get_shape())
        l = 0
        shape = list(y_pred.get_shape()[2:]) #例如[80,80,40]
        nc = y_pred.get_shape()[1]//scale
        print('nc', nc)
        for i in range(scale-1, -1, -1):
            print(i,shape)
            l += organ_segmentation_loss(depth=depth, w=w, w_seg=w_seg, organs = organs, global_neg_loss = global_neg_loss,
                                        dynamic_mask = dynamic_mask, stop_gradient = stop_gradient, loss_type = loss_type,
                                        single_label = single_label, soft_loss = soft_loss, ignore_top = ignore_top, check_blank = check_blank
                                        )(y_true, y_pred[:, i*nc:(i+1)*nc, 0:shape[0], 0:shape[1], 0:shape[2]])
            shape = [x//2 for x in shape]
            y_true = MaxPooling3D(pool_size=(2, 2, 2))(y_true) #这里maxpool会导致y_true里异常label信息烂掉，但是默认w=0时不会用异常label信息。
        l /= scale
        return l
    return loss

def organ_segmentation_loss(depth, w = 0.0, w_seg = 1.0, organs = slice(0,20), global_neg_loss = True,
                            dynamic_mask = False, stop_gradient = True, loss_type = 'binary',
                            single_label = True, soft_loss = False, ignore_top = 1.0, check_blank = True
                            ):

    #tensorflow支持slice作为tensor下标，但是不能用ndarray作为下标
    assert loss_type=='binary' or loss_type=='dice' or loss_type=='none' or loss_type=='soft', loss_type
    
    print('organs', organs)
    #print('abnormal w', w)
    
    def loss(y_true, y_pred):
        if not isinstance(organs, slice):
            temp = []
            for i in organs:
                temp.append(y_true[:, i])
            y_true_organ = MaxPooling3D(pool_size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)))(tf.stack(temp, axis = 1))
        else:
            y_true_organ = MaxPooling3D(pool_size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)))(y_true[:,organs])
        for i in range(3):
            print('y true organ', y_true_organ.get_shape())
        if not isinstance(organs, slice):
            temp = []
            for i in organs:
                temp.append(y_pred[:, i])
            y_pred_organ = tf.stack(temp, axis = 1)
            print('y_pred_organ', y_pred_organ.get_shape())
        else:
            y_pred_organ = y_pred[:, organs]
        for i in range(3):
            print('y pred organ', y_pred_organ.get_shape())
        y_pred_ab = y_pred[:,20:]
        #print('true abnormal', y_true_ab.get_shape())
        print('pred abnormal', y_pred_ab.get_shape())
        if loss_type=='binary':
            segment_loss = keras.losses.binary_crossentropy(y_true_organ, y_pred_organ)
        elif loss_type=='dice':
            segment_loss = 0
            for i in range(len(organs)): #3.26.2021之前写的是organs，错了！
                segment_loss += dice_coefficient_loss(y_true_organ[:,i:i+1], y_pred_organ[:,i:i+1])
            segment_loss /= len(organs)
            no_seg_label = K.less(K.sum(y_true_organ), 0.1)
            segment_loss = K.switch(no_seg_label, 0.0, segment_loss)
        elif loss_type=='soft':
            segment_loss = 0
            for i in range(len(organs)): #3.26.2021之前写的是organs，错了！
                segment_loss += soft_dice_coefficient_loss(y_true_organ[:,i:i+1], y_pred_organ[:,i:i+1])
            segment_loss /= len(organs)
        else:
            for i in range(3):
                print('no segmentation loss')
            segment_loss = 0
        if w==0:#这里当w是variable的时候，==0会自动判断为False
            for i in range(3):
                print('only segmentation loss')
            return w_seg * segment_loss
        if dynamic_mask: #在计算MIL loss的时候不用gt organ mask，而是用预测的organ segmentation mask
            #y_pred可能是小的，先变回原图尺度。
            temp = UpSampling3D(size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)),data_format="channels_first")(y_pred)
            #难道是因为这里y_true给自己赋值改变了的原因吗？？
            new_true = tf.concat([temp[:, 0:20], y_true[:, 20:]], 1)
            if stop_gradient:
                new_true = tf.stop_gradient(new_true) #不让MIL的计算影响organ mask的生成
        else:
            assert loss_type!='soft' #此情况还需修改，暂不使用
            new_true = y_true
        if isinstance(organs, slice):
            abnormal_loss = mil_loss_feature_adv(new_true, y_pred_ab, depth, minpool=False, pos_w=None, 
                                                 organs = list(range(organs.start, organs.stop)), global_neg_loss=global_neg_loss,
                                                 single_label = single_label, soft_loss=soft_loss, ignore_top=ignore_top,
                                                 check_blank=check_blank)
        else:
            abnormal_loss = mil_loss_feature_adv(new_true, y_pred_ab, depth, minpool=False, pos_w=None, 
                                                 organs = organs, global_neg_loss=global_neg_loss,
                                                 single_label = single_label, soft_loss=soft_loss, ignore_top=ignore_top,
                                                 check_blank=check_blank)
        return w_seg * segment_loss + w * abnormal_loss
    return loss

def MIL_loss_ms(scale, depth, use_pos_area = True, pos_w = None, organs = slice(0,20), global_neg_loss = True,single_label = True, 
                stop_gradient = True, soft_loss = False, ignore_top = 1.0, r = None, strict_weight = False,
                symmetry_suppress = False, coarse_bag = False, top_k=1, w = 1.0, check_blank = True,
                label_version = 0, intensity_w = .0, position_w = .0, w_4=1.0, abnormals=None,
                top_k_average = False, dynamic_mask=True):
    
    def loss(y_true, y_pred):
        l = 0
        shape = list(y_pred.get_shape()[2:]) #例如[80,80,40]
        shape = [x//(2**(scale-1)) for x in shape] #最低分辨率shape，不应低于(20,20,10)，否则y_true做crop时会丢失异常label
        nc = y_pred.get_shape()[1]//scale #channel数，25
        topk = top_k #用于区分局部变量与函数参数
        for i in range(0, scale):
            print(i,shape)
            if dynamic_mask: #使用y_pred中预测的脑区分割。y_true[1,1,h,w,d] 只用其中的异常label。直接crop。
                new_true = y_true[:,:,0:shape[0], 0:shape[1], 0:shape[2]]
            else: #y_true[1,25,h,w,d] 前半部分的分割要resize，后半部分异常label直接crop。
                temp = MaxPooling3D(pool_size=(2**(scale-i-1), 2**(scale-i-1), 2**(scale-i-1)))(y_true[:,:20])
                new_true = tf.concat([temp, y_true[:,20:,0:shape[0], 0:shape[1], 0:shape[2]]], 1)
            temp = MIL_loss(depth=depth, pos_w = pos_w, organs = organs, global_neg_loss = global_neg_loss,single_label = single_label, 
                          stop_gradient = stop_gradient, soft_loss = soft_loss, ignore_top = ignore_top, r = r, strict_weight = strict_weight,
                          symmetry_suppress = symmetry_suppress, coarse_bag = coarse_bag, top_k=topk, w = w, check_blank = check_blank,
                          label_version = label_version, intensity_w = intensity_w, position_w = position_w, w_4=w_4, abnormals=abnormals,
                          ret_pos_area=use_pos_area, top_k_average=top_k_average, dynamic_mask=dynamic_mask)\
                          (new_true, y_pred[:, i*nc:(i+1)*nc, 0:shape[0], 0:shape[1], 0:shape[2]])
            l += temp if not use_pos_area else temp[0]
            if use_pos_area:
                topk = tf.maximum(temp[1]*8, tf.ones_like(temp[1])*(8**(i+1)))/2 #下一分辨率的每个脑区每种异常的top_k
                topk = tf.cast(topk, tf.int32)
            else:
                topk *= 8 #保证低分辨率与高分辨率优化的空间大小基本一致
            shape = [x*2 for x in shape]
            #这里y_true不maxpool，因为y_true实际只有异常label信息，不是分割，如果maxpool那么异常label信息就烂了
            #但是又要保持y_true和y_pred的空间大小一致，因此截取y_true的前半部分空间，这样异常信息不会丢失。注意当single_label=False的时候，能允许的空间大小最小是20*20*10，
            #因为有20个异常label。
        l /= scale
        return l
    return loss

def MIL_loss(depth, pos_w = None, organs = slice(0,20), global_neg_loss = True,single_label = True, 
             stop_gradient = True, soft_loss = False, ignore_top = 1.0, r = None, strict_weight = False,
             symmetry_suppress = False, coarse_bag = False, top_k=1, w = 1.0, check_blank = True,
             label_version = 0, intensity_w = .0, position_w = .0, w_4=1.0, abnormals=None, ret_pos_area=False,
             top_k_average = False, dynamic_mask=True):
    """
    2021.3.13此函数完全是为了补救。multitask 27
    现在在改新模型的同时，还要兼顾旧模型的可兼容性，所以就不断新加模块
    之前模型主要有两个loss，一个pixel级别的，包括分割+异常MIL；一个global级别的分类，但是基本没用到。
    因为分割+异常MIL的loss被融合成一个loss了，在keras输出的时候看不出分别的变化。
    尤其是现在加入分割的loss weight后，更需要分别知道分割和异常MIL的loss。
    所以加入此函数，替换第二个loss，即将MIL部分分出来
    但是这样做之后，速度明显变慢了。只有原来的大概2/3的速度。原因是generator返回时候
    返回了两个大数组吗？
    果然，把y_true简化后，速度变得和原来一样了。但是这样，就不支持dynamic_mask=False了
    strict_weight: 默认false，即之前用的不同脑区的正负样本weight。但是那个weight设计可能不科学，
    计算方法比较奇怪，而且没有对高密度和低密度分别计算weight。如果strict_weight=True则对高低密度用严格的正负样本weight
    
    coarse_bag: False的时候，只计算脑区级别loss；True时，只计算图像级别loss。是数字的时候，为图像级别loss权重，同时也计算脑区级别loss
    3.3应reviewer要求，使用elastix分割计算MIL loss。加入dynamic_mask。
    """
    def loss(y_true, y_pred):
        """
        通常dynamic_mask=True，y_true是[1,1,h,w,d]，只使用里面的异常label，没有脑区分割
        若dynamic_mask=False，y_true是[1,21,h,w,d]，前面是脑区分割，后面是异常label
        y_true和y_pred空间分辨率一样
        """

        y_pred_ab = y_pred[:,20:]
        
        #限定dynamic_mask=True。
        #y_pred可能是小的，先变回原图尺度。
        #难道是因为这里y_true给自己赋值改变了的原因吗？？
        #new_true = tf.concat([temp[:, 0:20], y_true[:, 20:]], 1)
        abnormal_loss = None
        if coarse_bag:
            abnormal_loss = mil_loss_image(y_true, y_pred_ab, organs=organs, single_label=single_label,
                                           ignore_top=ignore_top, top_k=top_k, label_version=label_version,
                                           strict_weight=strict_weight, top_k_average=top_k_average)
        if coarse_bag is not True:
            if dynamic_mask: #用动态预测的脑区mask计算MIL loss
                temp = UpSampling3D(size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)),data_format="channels_first")(y_pred)
                new_true = tf.concat([temp[:, 0:20], y_true], 1) #用预测的脑区分割 加上 GT异常标注。
                #上面的upsample和multi-scale无关，是输出分辨率和generator output_size不一致时的调整。
            else:
                new_true = y_true
            if stop_gradient:
                new_true = tf.stop_gradient(new_true) #不让MIL的计算影响organ mask的生成
        
            temp_organs = list(range(organs.start, organs.stop)) if isinstance(organs, slice) else organs
            abnormal_loss = mil_loss_feature_adv(new_true, y_pred_ab, depth, minpool=False, pos_w=pos_w, 
                                                 organs = temp_organs, global_neg_loss=global_neg_loss,
                                                 single_label = single_label, soft_loss=soft_loss, ignore_top=ignore_top, r=r,
                                                 strict_weight=strict_weight, symmetry_suppress=symmetry_suppress, check_blank=check_blank,
                                                 label_version=label_version, intensity_w=intensity_w, position_w=position_w, w_4=w_4,
                                                 abnormals=abnormals, top_k=top_k, ret_pos_area=ret_pos_area,
                                                 top_k_average=top_k_average)
        if not ret_pos_area:
            return abnormal_loss * w
        else:
            return (abnormal_loss[0] * w, abnormal_loss[1]) 
    return loss

def zero_loss(y_true, y_pred):
    return tf.zeros((1))
def weighted_binary_cross_entropy(y_true, y_pred):
    """
    y_true和y_pred应该是[batch, 20]大小的
    也有可能是[batch, 20, 4]大小的
    """
    p = [0.05137981,0.05664488,0.08841684,0.10421205,0.0177923,0.01397967,
     0.00108932,0.0019971,0.1906318,0.1969862,0.10893247,0.097313,
     0.03013798,0.03812636,0.0375817,0.221132,0.22185911,0.02487291,
     0.02360203,0.0308642]
    #在异常label不止一个的情况下还得改
    p = np.array(p)
    #weights = tf.constant(((1-p)/p).tolist())
    weights = tf.constant((1-p).tolist()) #3.19.2021发现之前的weight不好，会导致pos类别很小的标签，weight过大。
    weights = tf.expand_dims(weights, 0)
    if len(y_pred.get_shape())==3:
        weights = tf.expand_dims(weights, 2)
    print('binary', y_true.get_shape())
    print('binary', y_pred.get_shape())
    print('binary', weights.get_shape())
    losses = -y_true * K.log(y_pred+1e-8) * weights - (1-y_true) * K.log(1-y_pred+1e-8)
    return K.mean(losses)

def weighted_binary_cross_entropy2(organs=slice(0,20)):
    def loss_function(y_true, y_pred):
        print('pred', y_pred.shape)
        print('true', y_true.shape)
        #region level
        pos_weights, neg_weights = get_new_weights(label_version=2)
        losses = -y_true * K.log(y_pred+1e-8) * pos_weights - (1-y_true) * K.log(1-y_pred+1e-8)* neg_weights
        loss = .0
        for i in organs:
            loss += K.mean(losses[:,i])
        #image level
        label = tf.zeros((1, 4)) #[1,4]。即使B不是1，后面做操作也可以broadcast
        all_organs = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
        for i in all_organs:#organs: 预测的12号organ作为image level预测
            label = K.maximum(label, y_true[:,i])
        weights = np.loadtxt('positive_ratio_1.1_coarse.txt') #去除缺血低密度后用这个 
        weights = np.minimum(1-1e-8, weights+1e-8) #+1e-8防止0
        pos_weights = np.power((1-weights)/weights, 0.5)
        neg_weights = np.ones(weights.shape)
        loss += K.mean(-label * K.log(y_pred[:,12]+1e-8) * pos_weights - (1-label) * K.log(1-y_pred[:,12]+1e-8) * neg_weights)
        
        return loss/(len(organs)+1)
    return loss_function

def binary_cross_entropy(y_true, y_pred):
    """
    """
    losses = -y_true * K.log(y_pred+1e-6) - (1-y_true) * K.log(1-y_pred+1e-6)
    return K.mean(losses)

def get_new_weights(label_version = 0):
    if label_version==0:
        weights =[[0.00269322, 0.02636089, 0.        , 0.       ],
                 [0.00163225, 0.02366767, 0.        , 0.        ],
                 [0.11646127, 0.09915939, 0.        , 0.        ],
                 [0.11042194, 0.08977393, 0.        , 0.        ],
                 [0.0160777 , 0.0180364 , 0.        , 0.        ],
                 [0.01509834, 0.01950543, 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0         ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.16053211, 0.2615686 , 0.        , 0.        ],
                 [0.16567371, 0.26050763, 0.        , 0.        ],
                 [0.11303354, 0.18305721, 0.        , 0.        ],
                 [0.11833837, 0.17954787, 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.06512691, 0.17840529, 0.        , 0.        ],
                 [0.06243369, 0.17505917, 0.        , 0.        ],
                 [0.02146413, 0.03280829, 0.        , 0.        ],
                 [0.04717212, 0.03917408, 0.        , 0.        ],
                 [0.04872276, 0.03705215, 0.        , 0.        ]]
        weights = np.array(weights)
    elif label_version==1: #四种异常，没有脑室
        #weights = np.loadtxt('positive_ratio_%d_regular.txt'%label_version)
        weights = np.loadtxt('positive_ratio_1.1_regular.txt') #去除缺血低密度后用这个 
        weights = np.minimum(1-1e-8, weights+1e-8) #+1e-8防止0
    elif label_version==2: #有脑室
        weights = np.loadtxt('positive_ratio_naoshi_regular.txt')
        weights = np.minimum(1-1e-8, weights+1e-8) #+1e-8防止0

    #pos_weights = 1-weights
    #neg_weights = weights
    """
    之前是用(1-sqrt(w))/sqrt(w)，这样有点问题，因为是一直把pos weight弄小的，weight=0.5的时候，pos weight才0.4
    现在用sqrt((1-w)/w)，这样在w很小的时候几乎是差不多的，但是w=0.5的时候，pos weight=1，是正确的
    """
    pos_weights = np.power((1-weights)/weights, 0.5)
    neg_weights = np.ones(weights.shape)
    print(pos_weights)
    return pos_weights, neg_weights

def mil_loss_feature_adv(y_true, y_pred, depth, minpool = False, pos_w = 10.0, organs = list(range(20)), global_neg_loss = True,
                         single_label = True, soft_loss = False, ignore_top = 1.0, r = None, strict_weight=False,
                         symmetry_suppress = False, check_blank = True, label_version = 0, intensity_w = .0, position_w = .0, w_4=1.0,
                         abnormals = None, top_k = 1, ret_pos_area=False, top_k_average = False):
    """
    y_true: [:,i]是每个点属于第i个脑区的概率(i<20)。[:,20]不对应三维图像，而是包含真实异常信息
    y_pred:每个点是此点有病灶的概率
    
    9.9 之前的mil_loss_feature中，对positive和negative的loss是不同的，
    negative loss对所有negative点做。
    每个点受到positive和negative loss的影响完全不同，结果模型全都输出基底节区。
    现在我对每个结构，都对其是否有异常做一个交叉熵loss（即每个结构会有一个positive或negative loss）
    原来的全局negative loss依然保留，为了在结构未定义的地方也能输出正确结果。
    因为negative样本明显更多，所以给positive价格权重w。
    可能的问题在于，原始结构区域经过max pool之后，在低分辨率可能扩大了，因此
    可能那个区域本来没有病灶，但是在max pool之后的锚点中包含了相邻区域的病灶区域，因此negative loss可能不准确。
    可能的解决方案是弄成相反数后max pool，再变回去，这样低分辨率区域就缩小了，但是可能被缩没。
    minpool=True表示采用此方法
    
    
    然后试了一下，loss变成nan了。不知道为什么，因为我加了smooth即使崩也不应该是nan啊。
    然后尝试用max试一下。
    soft_loss指计算bag的有异常概率时，不使用instance最大值，而使用概率乘积。
    r表示logsumexp的控制参数。r=None表示不是用log sum exp。
    intensity_w: label_version==1时，高低密度loss的权重
    """

    print('mil_loss organs', organs)
    print(keras.backend.image_data_format())
    if single_label:
        label = y_true[:, 20, 0, 0, 0:20]
    else:
        label = y_true[:, 20, 0, 0:20, 0:4]
        if label_version>0:#4种异常
            #高低密度：
            label_intensity = tf.stack([tf.reduce_max(label[:,:,0:2], [2]), tf.reduce_max(label[:,:,2:4], [2])], 2)
            label_position = tf.stack([tf.reduce_max(label[:,:,slice(0,3,2)], [2]), tf.reduce_max(label[:,:,slice(1,4,2)], [2])], 2)
            
            print('label_intensity', label_intensity.get_shape())
    #限制batch必须为1
    
    cids = [1,0,3,2,5,4,7,6,9,8,11,10,12,14,13,16,15,17,19,18] #3.27.2021加入
    if not strict_weight:
        if label_version==0:
            weights = [0.05137981,0.05664488,0.08841684,0.10421205,0.0177923,0.01397967,
             0.00108932,0.0019971,0.1906318,0.1969862,0.10893247,0.097313,
             0.03013798,0.03812636,0.0375817,0.221132,0.22185911,0.02487291,
             0.02360203,0.0308642] #weights是有异常的概率，越小正权重越大
            weights = np.array(weights)
            weights = np.power(weights, 0.5)/3 #2.17修改为/3
        elif label_version>0:
            weights = np.ones((20))*0.1 #6.12.2021，因为细分脑内脑外后，有些区域阳性比例太低了。先尝试不加权
        pos_weights = ((1-weights) / weights)
        pos_weights = np.expand_dims(pos_weights, 1).repeat(4, axis=1)
        neg_weights = np.ones(20)
        neg_weights = np.expand_dims(neg_weights, 1).repeat(4, axis=1)
    else:
        pos_weights, neg_weights = get_new_weights(label_version)
    
    pos_loss = 0.0
    neg_loss = 0.0
    pos_map = None
    print('y_true', y_true) #tf.float32
    assert y_true.dtype==tf.float32
    y_pred = y_pred
    pos_area = [tf.zeros_like(y_pred[0,:,0,0,0]) for i in range(20)] #20个(4)或(5)的tensor
    
    for i in organs:
        if i!=cids[i] and symmetry_suppress:
            s_weights = 1-0.5*tf.abs(label[0,i]-label[0,cids[i]])
        else:
            s_weights = np.array([1,1,1,1])
        #current_layer = MaxPooling3D(pool_size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)),data_format="channels_first")(y_true[:,i:i+1,:,:,:])
        current_layer = MaxPooling3D(pool_size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)))(y_true[:,i:i+1,:,:,:])
        bag_size = K.sum(current_layer) #此bag含有的元素
        pred = y_pred * current_layer
        if r is not None:
            pred = tf.where(tf.ones_like(pred)*current_layer>0.1, pred, -10000*tf.ones_like(pred))
        #for level_number in range(depth - 2, -1, -1):
        #    current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
        #pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
        if single_label: #对于top_k没有在维护了
            #2021.2.28加入了ignore_top，目的是让输出异常区域覆盖尽量全面，而不陷入几个像素点
            if r is None:
                pos = K.max(tf.where(pred<=ignore_top, pred, tf.zeros_like(pred)))#前提是batch是1！！
            else:
                pos = (K.logsumexp(r*pred)-K.log(bag_size))/r
            soft_pos = 1 - K.prod(1 - pred)
        else: #此时y_pred第channel维度是4而不是1，乘法应该是广播的。得到pos应该是[batch, 4]才对
            #从网站上差的，axis为什么只能是一个integer，而不能是一个列表？？下标是从1开始？？
            #我吐了，从bing上都查不到。然而我用list而且从0开始下标是对的，从1开始下标是错的!
            pos_area[i] = K.sum(tf.where(pred>0.5, tf.ones_like(pred), tf.zeros_like(pred)), axis=[0,2,3,4]) #(4)或(5)
            if r is None:
                if top_k==1:
                    pos = K.max(tf.where(pred<=ignore_top, pred, tf.zeros_like(pred)), axis=[0,2,3,4]) #前提batch是1！
                else:
                    a = []
                    for j in range(4):
                        temp = K.flatten(pred[0,j]) #batch=1
                        if type(top_k)==int: #固定前top_k个参与计算
                            topk = top_k
                        elif type(top_k)==float: #采用自己预测的较高像素个数的top_k比例参与计算
                            topk = tf.cast(tf.maximum(top_k*pos_area[i,j], 10), tf.int32) #参考71最小10个像素，如果输出分辨率低应该用更少
                        else: #采用top_k[i,j]，即上一层预测值较高像素个数的一定比例进行计算
                            topk = top_k[i,j]
                        a.append(tf.math.top_k(temp, topk).values)
                    
                    pos = tf.stack(a, axis=1) if type(top_k)==int else a #(top_k, 4)，如果是动态topk，四个channel不一样不能stack，保持list
                    if top_k_average: 
                        #之前使用top k的时候其实不是MIL pool，而是在前k个点计算交叉熵取平均。结果是小异常激活区域较大
                        #现在改为将前k个预测值的平均作为bag预测值，然后做一个交叉熵。
                        if type(pos)!=list:
                            pos = K.mean(pos, 0)
                        else:
                            b = []
                            for j in range(4):
                                b.append(K.mean(a[j]))
                            pos = tf.stack(b)
                        print('top k average, pos', pos.shape)
            else: #对于top_k好像没有维护了
                pos = (K.logsumexp(r*pred, axis=[0,2,3,4])-K.log(bag_size))/r
                pos = K.maximum(pos, 0)
                #pos = K.logsumexp(pred, axis=[0,2,3,4])
            soft_pos = 1 - K.prod(1 - pred, axis=[0,2,3,4])
            if label_version>0: #四种异常
                #当intensity_w和position_w!=0时有用。但对于top_k好像会出问题
                #目前不用intensity_w和position_w，即下面这两个东西没用。
                pos_intensity = tf.stack([tf.reduce_max(pos[0:2]), tf.reduce_max(pos[2:4])]) #(2,)
                pos_position = tf.stack([tf.reduce_max(pos[slice(0,3,2)]), tf.reduce_max(pos[slice(1,4,2)])]) #(2,)
                print('pos_intensity', pos_intensity.get_shape())
            #在multiscale下，计算低分辨率下预测的阳性部分面积，以供高分辨率下计算
        
        if type(pos)!=list:
            print('pos shape', pos.get_shape()) #(4,) 或 (top_k,4)
        else:
            print('pos is list', len(pos))
        print('check_blank', check_blank)
        if check_blank:
            blank = K.less(K.sum(current_layer), 0.1)
        else:
            blank = tf.constant(0, tf.int16)
        if minpool:
            small_map = -MaxPooling3D(pool_size=(2**(depth - 1), 2**(depth - 1), 2**(depth - 1)))(-y_true[:,i:i+1,:,:,:])
            neg = K.max(y_pred * small_map) #此处在非single_label下还需修改
            neg_blank = K.less(K.sum(small_map), 0.1)
        else:
            if not type(pos)==list: #在pos是list的时候，不需要neg
                neg = K.maximum(1 - pos, 0) #用log exp sum的时候，因为我自己的方法，有时候计算不精确，防止neg为负
            soft_neg = 1-soft_pos
            if label_version>0:
                neg_intensity = K.maximum(1-pos_intensity, 0)
                neg_position = K.maximum(1-pos_position, 0)
                
            neg_blank = blank
        if type(pos)==list:
            #目前默认single_label=False, label_version=1, pos_w=None, soft_loss=False
            for j in range(4):
                pos_loss = pos_loss - K.mean(K.switch(blank, .0, label[0,i,j] * K.log(pos[j] + 1e-8)) * pos_weights[i,j]*s_weights[j]) / 4
                neg_loss = neg_loss - K.mean(K.switch(neg_blank, .0, (1-label[0,i,j]) * K.log(1-pos[j] + 1e-8))) * neg_weights[i,j]*s_weights[j] / 4
            
        elif single_label: #每个部位预测一个异常label
            #此模式下pos_weights、neg_weights即s_weights暂不可用
            if pos_w is None:
                pos_loss = pos_loss - K.switch(blank, 0.0, label[0,i] * K.log(pos + 1e-8) * pos_weights[i])
            else:
                pos_loss = pos_loss - K.switch(blank, 0.0, label[0,i] * K.log(pos + 1e-8)) * pos_w
            neg_loss = neg_loss - K.switch(neg_blank, 0.0, (1-label[0,i]) * K.log(neg + 0.1) * neg_weights[i])
        else: #每个部位预测多个异常label，此时pos应是一个长为4的向量（当然还有batch等维度）
            #实际pos和label[0,i]的长度都是4，但是因为用softmax激活，其中至少有一维是留给正常类的，
            #计算loss时应当忽略那一维。这里暂时只对前两维计算，忽略后两维
            if label_version==0:
                #nc = 3 #异常channel为4，分别为高、低、0、正常
                ch = slice(0,3)
            elif label_version>0:
                nc = 4 #异常channel为5，脑内高、脑外高、脑内低、脑外低、正常。
                ch = slice(0,4)
            if abnormals is not None:
                ch = abnormals
            if label_version!=1 or w_4:
                if pos_w is None: #一个情况下返回0，另一个情况返回(4,)的tensor，可以这样？
                    pos_loss = pos_loss - K.mean(K.switch(blank, .0, label[0,i,ch] * K.log((pos[ch] if len(pos.shape)==1 else pos[:,ch]) + 1e-8)) * pos_weights[i,ch]*s_weights[ch])
                    if soft_loss: #之前nc是2不是3，忘了为什么那样设置了
                        pos_loss = pos_loss - K.mean(K.switch(blank, .0, label[0,i,ch] * K.log(soft_pos[ch] + 1e-8)) * pos_weights[i,ch]*s_weights[ch])
                else:
                    pos_loss = pos_loss - K.mean(K.switch(blank, .0, label[0,i,ch] * K.log((pos[ch] if len(pos.shape)==1 else pos[:,ch]) + 1e-8))) * pos_w
                    if soft_loss:
                        pos_loss = pos_loss - K.mean(K.switch(blank, .0, label[0,i,ch] * K.log(soft_pos[ch] + 1e-8))) * pos_w
                neg_loss = neg_loss - K.mean(K.switch(neg_blank, .0, (1-label[0,i,ch]) * K.log((neg[ch] if len(pos.shape)==1 else neg[:,ch]) + 1e-8))) * neg_weights[i,ch]*s_weights[ch]
                if soft_loss:
                    neg_loss = neg_loss - K.mean(K.switch(neg_blank, .0, (1-label[0,i,ch]) * K.log(soft_neg[ch] + 1e-8)) * neg_weights[i,ch]*s_weights[ch])
            if label_version>0: #四种异常。除了每种异常算交叉熵外，计算高低密度的loss。当intensity_w和position_w为0时无效
                #assert pos_w is None and soft_loss==False
                pos_loss = pos_loss - K.mean(K.switch(blank, .0, label_intensity[0,i] * K.log(pos_intensity + 1e-8))) * intensity_w
                neg_loss = neg_loss - K.mean(K.switch(neg_blank, .0, (1-label_intensity[0,i]) * K.log(neg_intensity + 1e-8))) * intensity_w
                
                pos_loss = pos_loss - K.mean(K.switch(blank, .0, label_position[0,i] * K.log(pos_position + 1e-8))) * position_w
                neg_loss = neg_loss - K.mean(K.switch(neg_blank, .0, (1-label_position[0,i]) * K.log(neg_position + 1e-8))) * position_w
                
        #current_layer = K.equal(current_layer, 1)
        
        if pos_map is None:
            #pos_map = current_layer * label[0,i]
            #9.28修改，之前pos_map真的是有异常的那部分，但是这样每个bag除了MIL的loss外，在为negative bag的时候还会受到另一个negative 惩罚，效果很差
            #所以现在pos_map只是只我考虑的bag，例如考虑左额叶，那pos_map就不管左额叶有无异常都是左额叶。这样限制不被bag包含的地方有一个negative loss。
            pos_map = current_layer 
            
        else:
            #pos_map = tf.maximum(pos_map, current_layer * label[0,i])
            #9.28同上
            pos_map = tf.maximum(pos_map, current_layer)
        
        del current_layer

    #for level_number in range(depth - 2, -1, -1):
    #    current_layer = MaxPooling3D(pool_size=(2,2,2))(current_layer)
    
    print('pred type', y_pred)
    #blank = K.equal(K.sum(current_layer), 0)
    """
    y_pred = K.pow(y_pred*0.8, 10)
    pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    neg = K.prod(1 - y_pred * (1 - current_layer)) #mask为0处预测无异常的概率，最佳应为1
    loss = K.switch(blank, -K.log(neg+1e-30), -K.log(pos+1e-30)-K.log(neg+1e-30))
    """
    #pos = 1 - K.prod((1 - y_pred * current_layer)) #mask为1处预测有异常的概率，最佳应为1
    #pos_loss = -K.log(pos + 1e-30)
    
    #temp = 1 - pos_map
    if global_neg_loss: #默认选择
        neg_loss = neg_loss + K.mean(-K.log(1 - y_pred * (1 - pos_map) + 1e-8))
    #loss = K.switch(blank, neg_loss, pos_loss + neg_loss)
    loss = pos_loss + neg_loss
    loss /= len(organs)
    
    pos_area = tf.stack(pos_area, 0)
    #return loss
    if label_version==0:
        return loss*(1-label[0,6,3]) #4.25，有不明区域的不计算loss
    else:
        #return loss #6.26之前
        #return loss*label[0,13,3] #6.27到7.3。训练阶段只使用非手标数据，验证是验证集的非手标数据，测试是验证集的手标数据。
        if not ret_pos_area:
            return loss*(1-K.max(label[0], 1)[7])#7.3重新划分数据集之后，有不明区域不计算loss
        else:
            return (loss*(1-K.max(label[0], 1)[7]), pos_area)
    pass

def mil_loss_image(y_true, y_pred, organs = list(range(20)), single_label = False, ignore_top = 1.0,
                   top_k = 1, label_version = 0, strict_weight = False, top_k_average = False):
    """
    y_true:[1,1,h,w,d]
    ignore_top应保持为1，否则会有问题
    """
    print('coarse strict weight', strict_weight)
    if strict_weight:
        weights = np.loadtxt('positive_ratio_1.1_coarse.txt') #去除缺血低密度后用这个 
        weights = np.minimum(1-1e-8, weights+1e-8) #+1e-8防止0
        pos_weights = np.power((1-weights)/weights, 0.5)
        neg_weights = np.ones(weights.shape)
        print('coarse pos weight', pos_weights) #应该是[4,]形状的
    else:
        pos_weights = np.ones([4])
        neg_weights = np.ones([4])
        
    if single_label:
        label_organ = y_true[0, 0, 0, 0, 0:20]
        label = tf.zeros((1))
    else:
        label_organ = y_true[0, 0, 0, 0:20, 0:4]
        label = tf.zeros((4))
    print('mil loss image: label', label.get_shape(), label_organ.get_shape())
    #没有13，因为[13,3]是用来标记是否是NLP生成的，不是异常label。也不应有14，因为左右flip时也会产生影响
    #如果有脑室，应该要修改
    #包含6其他区域、7不明区域
    use_organs = [0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19]
    if label_version==2:
        use_organs.extend([13,14])
    for i in use_organs:#organs:
        label = K.maximum(label, label_organ[i])
    if single_label:
        #pos = K.max(y_pred)
        pos = K.max(tf.where(y_pred<ignore_top, y_pred, tf.zeros_like(y_pred)))
        #top_k需改
    else:
        #pos = K.max(y_pred, axis=[0,2,3,4])
        #pos = K.max(tf.where(y_pred<ignore_top, y_pred, tf.zeros_like(y_pred)), axis=[0,2,3,4])
        a = []
        for i in range(4):
            temp = K.flatten(y_pred[:,i])
            a.append(tf.math.top_k(temp, top_k).values)
            #7.30.2021突然意识到使用ignore_top是有问题的，之前只想了正样本的情况，但对负样本来说使用ignore_top无法压制预测值极高的点。
            #虽然之前通过实验现象发现了ignore_top有问题，但没有意识到原理上的错误
        pos = tf.stack(a, axis=1) #(top_k, 4)
        if top_k_average:
            pos = K.mean(pos, 0, keepdims = True) #这里用keepdims，下面计算就不用改了
            for i in range(5):
                print('top k average, pos', pos.shape)
    print('pos', pos.get_shape())
    print('coarse label version', label_version)
    loss = 0
    if single_label:
        loss = -label * K.log(pos+1e-8) - (1-label) * K.log(1-pos+1e-8)
        #top_k需改
    else:
        #应该能用广播吧？
        if label_version==0:
            loss = K.mean(-label[:3] * K.log(pos[:,:3]+1e-8) - (1-label[:3]) * K.log(1-pos[:,:3]+1e-8))
        else:
            loss = K.mean(-label[:4] * K.log(pos[:,:4]+1e-8) * pos_weights - (1-label[:4]) * K.log(1-pos[:,:4]+1e-8) * neg_weights)
    return loss
'''                       
def mil_loss_simple(y_true, y_pred, depth):
    """
    y_pred:每个点是此点有病灶的概率
    smooth是因为loss刚开始是inf。
    """
    #计算正样本loss：
    for i in range(10):
        print()
    print(type(y_true), type(y_pred))
    print('true', K.int_shape(y_true))
    print('pred', K.int_shape(y_pred))
    print('true', y_true.get_shape())
    print('pred', y_pred.get_shape())
    print('multiply', (y_pred * y_true).get_shape())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(y_true_f.get_shape())
    print(y_pred_f.get_shape())
    print((y_pred_f*y_true_f).get_shape())
    
    y_true = y_true[:,1,:,:,:]
    print('true', y_true.get_shape())
    
    for level_number in range(depth-1):
        
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
    #我无论pred输出1个channel还是2个channel这里都不报错？？？都是11.5128
    #到底是什么鬼？y_true应该是几个channel的？
    pos = 1 - K.prod((1 - y_pred * y_true)) #mask为1处预测有异常的概率，最佳应为1
    neg = K.prod(1 - y_pred * (1 - y_true)) #mask为0处预测无异常的概率，最佳应为1
    return -10*K.log(pos+smooth)-K.log(neg+smooth) #两个概率和真实标签（1）求交叉熵，再加权求和
'''
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_coefficient(y_true, y_pred, smooth=1.):
    y_weight_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = tf.cast(tf.greater(y_weight_f, 0), tf.float32)
    maxn = tf.reduce_max(y_weight_f)
    y_weight_f = y_weight_f/maxn #区域中心归一化为1。应该不会出现0吧？？
    y_weight_f = tf.minimum(tf.abs(y_weight_f), 3)+0.5 #边界太大的权值截断。之前没有+0.5,4.17加入+0.5
    intersection = K.sum(y_true_f * y_pred_f * y_weight_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_weight_f) + K.sum(y_pred_f*y_weight_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred) #2021.3.10加入了1

def soft_dice_coefficient_loss(y_true, y_pred):
    return 1-soft_dice_coefficient(y_true, y_pred) #2021.3.10加入了1


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
