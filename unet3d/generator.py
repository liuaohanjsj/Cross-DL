import os
import copy
from random import shuffle
import random
import itertools
import cv2
import numpy as np
import PIL

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y, my_augment_data, rotate_image
from .utils.utils import maxpool

generator_config = {} 

def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=False, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False, truth_dtype = np.int8,
                                           organ_label = False, multi_task = False, truth_file = None, organs = None,
                                           label_file = None, rotate = False, separate_mil = False, output_shape = None,
                                           no_segmentation = False, classify=False, version=None,
                                           train_epoch = None, valid_epoch = None, triple_loss = False, classify_only = False,
                                           dynamic_mask = True):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    truth_file是我加的，如果为None就用data_file里面的truth，否则用truth_file里面的truth
    organs只有在multi input的时候才用得上，用来确定输入的organ mask
    label_file默认为None，从data_file的h5文件中读取label。在co-training中使用动态label，此时需要从label_file中读取label
    label_file应该是已经打开的h5文件，而不是文件名
    
    separate_mil:如果为True，则返回第二部分不是global分类结果，而是和第一部分一样
    """
    global generator_config
    generator_config['organs'] = organs
    generator_config['label_file'] = label_file
    generator_config['rotate'] = rotate
    generator_config['separate_mil'] = separate_mil
    generator_config['output_shape'] = output_shape
    generator_config['flip'] = augment_flip
    generator_config['no_segmentation'] = no_segmentation
    generator_config['classify'] = classify
    generator_config['version'] = version #只适用于downstream
    generator_config['triple_loss'] = triple_loss
    generator_config['classify_only'] = classify_only
    generator_config['dynamic_mask'] = dynamic_mask
    
    augment = False #不使用原代码提供的增广方法，只在convert_data中使用flip
    
    if not validation_batch_size:
        validation_batch_size = batch_size

    #这两个list应该是两个ID的；list，；类似[6,1,3,2]，没有具体文件名或数据内容
    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,
                                        truth_dtype=truth_dtype,
                                        organ_label = organ_label,
                                        multi_task=multi_task,
                                        truth_file = truth_file,
                                        rotate=rotate,
                                        epoch_size=train_epoch)
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          truth_dtype=truth_dtype,
                                          organ_label = organ_label,
                                          multi_task=multi_task,
                                          truth_file = truth_file,
                                          rotate=False,
                                          epoch_size=valid_epoch)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = train_epoch if train_epoch else \
                        get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                            skip_blank=skip_blank,
                                            patch_start_offset=training_patch_start_offset,
                                            patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = valid_epoch if valid_epoch else\
                            get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                skip_blank=skip_blank,
                                                patch_overlap=validation_patch_overlap),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        random.seed(10) #2021.1.4加入，使不同实验训练测试集相同
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False, truth_dtype = np.int8, organ_label = False,
                   multi_task=False, truth_file = None, rotate=False, epoch_size=None):
    """
    此函数无return，但是用yield返回东西，是一个generator
    """
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        l_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        if epoch_size:
            while len(index_list)<epoch_size:
                index_list.extend(index_list)
            index_list = random.sample(index_list, epoch_size)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
                     skip_blank=skip_blank, permute=permute,
                     l_list = (l_list if organ_label else None),
                     truth_file = truth_file, rotate=rotate)
            #生成完一个batch后，返回此batch的x_list和y_list（两个list变成array，x是datafile里相同的类型，应该是float32。y变成了np.int8类型的）
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                #下面经过conver_data后，y变成了n_labels通道的
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels, truth_dtype=truth_dtype,
                                   l_list = (l_list if organ_label else None),
                                   multi_task = multi_task)
                #del x_list
                #del y_list
                #del l_list
                x_list = list()
                y_list = list()
                l_list = list()


def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape: #应该不执行此路径
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False, l_list = None, truth_file = None, rotate = False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape, truth_file = truth_file)
    if augment: #应该不会执行，暂未修改里面的truth_file部分
        print('fuck')
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)
    if rotate:
        #print('generator rotate')
        data, truth = my_augment_data(data, truth)
        
    if permute: #应该不会执行
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]
    #这里truth是单通道的，[1,256,256,32]
    #print('shape', data.shape, truth.shape)
    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)
        if l_list is not None: #lah added
            if generator_config['label_file'] is None:
                l_list.append(data_file.root.label[index])
            else:
                l_list.append(generator_config['label_file'].root.label[index])

def get_data_from_file(data_file, index, patch_shape=None, truth_file = None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None, truth_file = truth_file)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        if truth_file is None:
            x = data_file.root.data[index]
            if len(data_file.root.truth)>index:
                y = data_file.root.truth[index, 0]
            else: #用于下游任务，没有分割truth的时候
                y = np.zeros((1)) #generator不生成分割的y
        else:
            x, y = data_file.root.data[index], truth_file.root.truth[index, 0]
    #print('data type', x.dtype, y.dtype) 
    #print(x.shape, y.shape)
    #这里y还是单通道的
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None, 
                 truth_dtype = np.int8, l_list = None, stride=1, multi_task=False):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if l_list is None: #纯分割，没有label的分类
        if n_labels == 1:
            y[y > 0] = 1
        elif n_labels > 1:
            y = get_multi_class_labels(y, n_labels=n_labels, labels=labels, truth_dtype=truth_dtype)
        return x, y
    elif generator_config['classify']: #下游任务如cq500，没有分割，只有分类
        meml = copy.deepcopy(l_list)
        for i in range(len(l_list)):
            if generator_config['flip'] and random.random()<0.5:
                #如果是[batch, channel, 前后, 左右, 上下]存储的，就是axis=3。
                #但是这里图像是直接从nii读取的，nii存储图像的方向和普通图片是不一样的。应该是axis=2
                #[batch, channel, 左右, 前后, 上下]
                #print('flip')
                if generator_config['version']<=3:
                    x[i,:3] = np.flip(x[i,:3], 1)
                    memx = x[i,3].copy()
                    x[i,3] = np.flip(x[i,4], 0)
                    x[i,4] = np.flip(memx, 0)
                elif generator_config['version']==4 or generator_config['version']==5:
                    x[i,:2] = np.flip(x[i,:2], 1)
                    memx = x[i,2].copy()
                    x[i,2] = np.flip(x[i,3], 0)
                    x[i,3] = np.flip(memx, 0)
                elif generator_config['version']==10:
                    x[i] = np.flip(x[i], 1)
                if len(l_list[i])==4: #cq500上，左右出血互换。rsna不进行操作
                    l_list[i][2] = meml[i][3]
                    l_list[i][3] = meml[i][2]
            
        return x, np.asarray(l_list)
    else:
        #不能把label单独作为第三个值返回，这样好像会使得keras认为这是sample_weight
        #也不能作为一个元组返回data should be a Numpy array, or list/dict of Numpy arrays
        #也不能作为list返回，否则还要改model，不知道怎么改
        #the list of Numpy arrays that you are passing to your model is not the size the model expected
        
        #return x, [y, np.asarray(l_list)]
    
        
        #print('label shape', np.asarray(l_list).shape) #(1, 20)
        #print('y', y.shape, y.dtype)
        #print(np.sum(y))
        s_index = [1,0,3,2,5,4,6,7,9,8,11,10,12,14,13,16,15,17,19,18]
        flip = False
        if generator_config['flip'] and random.random()<0.5:
            #如果是[batch, channel, 前后, 左右, 上下]存储的，就是axis=3。
            #但是这里图像是直接从nii读取的，nii存储图像的方向和普通图片是不一样的。应该是axis=2
            flip = True
            #minn = np.min(x)
            #maxn = np.max(x)
            #cv2.imwrite('ori.png', (x[0,0,:,:,40]-minn)/(maxn-minn)*255)
            x = np.flip(x, 2)
            if not generator_config['no_segmentation']: #节约时间 
                y = np.flip(y, 2) #y只翻转是不够的，还需要将y中脑区标签左右替换，这一步在下面位运算时完成，注意6和7分割并没有对换
            #cv2.imwrite('flip.png', (x[0,0,:,:,40]-minn)/(maxn-minn)*255)
            #cv2.imwrite('seg.png', np.uint8(y[0,0,:,:,40]))
            """
            #meml = l_list.copy() 
            吐了！之前引入左右增广的时候用的这个copy，实际list内部元素没有copy
            左侧会变成右侧label是对的，但是右侧变成左侧的时候其实就是原来的右侧label，所以右侧label是错的！
            我说为什么右侧分割总是包含左侧假阳，而且右侧异常检测也有些检不出来
            """
            meml = copy.deepcopy(l_list)
            for i in range(len(l_list)):
                for j in range(20):
                    l_list[i][j] = meml[i][s_index[j]]
        #print(flip)
        
        shape = list(y.shape) #tuple不能赋值
        shape[1] = 21
        if generator_config['output_shape'] is None: #计算loss时输入y_true为原始分辨率
            ret = np.zeros(shape, np.int8)
        else:
            #for i in range(3):
            #    print('output shape', generator_config['output_shape'])
            ret = np.zeros((shape[0], 21)+generator_config['output_shape'], np.int8)
        
        #print('generator max min', np.max(y), np.min(y))
        """
        想在对y做位运算前先resize，这样就不用对每个位取出来后都做resize了。
        问题是，cv2.resize不能对uint8类型操作，我又不能把它换成uint8或者float32
        if generator_config['output_shape'] is not None:
            y_resize = np.zeros(shape[:2]+list(generator_config['output_shape']), y.dtype)
            print(y.dtype)
            for j in range(shape[0]):
                y_resize[j,0,:,:,:] = resize(y[j,0,:,:,:], generator_config['output_shape'])
        else:
            y_resize = y
        """
        for i in range(20):
            if generator_config['no_segmentation']:
                continue
            temp = np.bitwise_and(np.right_shift(y[:,0,:,:,:], i if not flip else s_index[i]), 1)
            temp = np.int8(temp) #可能没什么用
        
            #cv2.imwrite('seg_%s_%d.png'%('flip' if flip else '', i), np.uint8(temp[0,:,:,40])*255)
            #print(i, np.sum(temp))
            #print('temp', temp.shape, ret[:,i,:,:,:].shape)
            for j in range(shape[0]):
                ret[j,i,:,:,:] = temp[j] if generator_config['output_shape'] is None \
                                 else resize(temp[j], generator_config['output_shape'])
            #cv2.imwrite('%d.png'%i, np.uint8(temp[0,:,:,40]*100))
        #print(l_list)
        
        assert stride==1
        if l_list[0].ndim==1: #每个组织只有一个label，有无异常
            ret[:, 20, 0, 0, 0:20] = np.int8(np.asarray(l_list))
        else: #每个组织有多个label，有无各个类型的异常
            assert l_list[0].shape[0]==20 and l_list[0].shape[1]==4, l_list[0].shape
            ret[:, 20, 0, 0:20, 0:4] = np.int8(np.asarray(l_list))
        #print('ret', ret.shape, ret.dtype)
        if not multi_task:
            return x, ret
        elif generator_config['organs'] is None:
            if generator_config['classify_only']: #直接做N*M个二分类
                #print('classify only', np.int8(np.asarray(l_list)))
                return x, np.int8(np.asarray(l_list))
            elif generator_config['separate_mil']:
                if generator_config['no_segmentation']:
                    return x, [ret[:, 20:], ret[:, 20:]]
                elif generator_config['triple_loss']:
                    if generator_config['dynamic_mask']: 
                        return x, [ret, ret[:, 20:], ret[:, 20:]]
                    else: #模型不分割脑区，做MIL使用elastix脑区分割。
                        return x, [ret[:,:1,:,:,:], ret, ret[:, 20:]]
                        #第一个输出对应分割loss，不需要，随便返回个东西5维的东西（返回全ret太慢）。第二个是脑区级MIL loss，需要GT脑区。
                else:
                    return x, [ret, ret[:, 20:]]
            else:
                return x, [ret, np.int8(np.asarray(l_list))]
            
        else: #multi_input
            mask = np.zeros(x.shape, np.int8)
            labels = np.int8(np.asarray(l_list)) #shape (batch, 20)
            for i in range(20):
                if generator_config['organs'][i]:
                    temp = np.bitwise_and(np.right_shift(y, i), 1) #flip待实现
                    temp = np.int8(temp)
                    mask = np.bitwise_or(mask, temp)
                else: #为了不让模型困惑，求解的脑区之外的label都要置零
                    labels[:, i] = 0 #即使labels是[batch,20,4]形状应该也可以这样赋值吧
            #print('shape', x.shape, mask.shape)
            #print('positive points', np.sum(mask), np.max(mask), np.min(mask))
            #print(l_list)

            return [x, mask], [ret, labels]
        

def get_multi_class_labels(data, n_labels, labels=None, truth_dtype = np.int8):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, truth_dtype)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    #print('after label conversion', y.shape)
    return y

def distance_map_3D_improve(img):
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
    ret = din-dout
    # print(np.max(ret), np.min(ret))
    return ret

def resize(img, shape, inter=cv2.INTER_NEAREST):
    """
    三维resize
    """
    if img.shape==shape:
        print('same shape', shape)
        return img
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=inter) #行列resize
    img = np.transpose(img, (0,2,1)) #160,5,160
    img = cv2.resize(img, (shape[2], shape[0]), interpolation=inter) #纵向resize
    img = np.transpose(img, (0,2,1))
    return img

