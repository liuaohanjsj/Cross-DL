import os

import numpy as np
from tqdm import tqdm
import tables
import random
import PIL
import nibabel as nib
import time

from .normalize import normalize_data_storage, reslice_image_set, my_reslice_image_set
from .augment import rotate_image
from .generator import resize
from .utils import check_image_exist

def create_data_file(out_file, n_channels, n_samples, image_shape, use_int32 = False):
    hdf5_file = tables.open_file(out_file, mode='w')
    #filters是用来压缩的，后面考虑可否用更高的压缩系数或其他压缩方法减小目标文件大小
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    #for i in range(3):
    #    print('truth shape', truth_shape)
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    if use_int32:
        truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt32Atom(), shape=truth_shape,
                                                filters=filters, expectedrows=n_samples)
                                                #expectedrows=n_samples)
    else:
        truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True, organ_label = False, rotate = False, truth_only = False,
                             use_cv2_resize = False, save_truth=True):
    """
    truth_only=True则只保存truth（分割）
    save_truth=False用于外部测试集，即数据不用于训练仅用于测试，不需要保存脑区分割
    """
    #print('crop', crop)
    random.seed(10) #之前没有设置seed，导致下面旋转的部分随机都无法复现 10
    for set_of_files in tqdm(image_files):
        ti = time.time()
        print('start reslice')
        temp = set_of_files[:-1] if organ_label else set_of_files
        if any(not check_image_exist(x) for x in temp): #如果有任何一个file不存在的（分割mask文件不存在）
            #images = [nib.Nifti1Image(np.zeros((10,10,10)), np.zeros((4,4))) for x in temp]
            subject_data = [np.zeros((160,160,80)) for x in temp]
            affine = np.zeros((4,4))
            if rotate:
                angle = random.randint(-15,15) #必须保证也调用一次random，保持和之前random序列一致
        else:
            f = my_reslice_image_set if use_cv2_resize else reslice_image_set
            if organ_label: #set_of_files最后一个是label.txt
                #label_indices也得改！！刚开始就因为没有改，导致mask也用的linear插值，结果训练怎么都不对！！找了好半天问题
                images = f(set_of_files[:-1], image_shape, label_indices=len(set_of_files) - 2, crop=crop)
            else:
                images = f(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
            #print('reslice', time.time()-ti)
            subject_data = [image.get_data() for image in images]
            for x in subject_data:
                print('shape', x.shape)
            affine = images[0].affine
            #subject_data前面若干张（一张）应该是输入图像，最后一个是truth（分割）图像
            #每个元素应该是长、宽、高三维的图像吧
            if rotate:
                angle = random.randint(-15,15) #15
                for i,img in enumerate(subject_data):
                    #print('subject_data', img.shape, type(img))
                    if truth_only and i<len(subject_data)-1: #只保存分割的话，CT本身也不需要进行旋转操作了
                        continue
                    if save_truth and i==len(subject_data)-1:
                        inter = PIL.Image.NEAREST
                    else:
                        inter = PIL.Image.BILINEAR
                    print('start rotate', time.time())
                    subject_data[i] = rotate_image(img, angle, inter)
                    print('end rotate', time.time())
        ti = time.time()
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels,
                            truth_dtype, truth_only=truth_only, save_truth=save_truth)
        print('save\n', time.time()-ti)
    return data_storage, truth_storage

def write_label_data_to_file(image_files, label_storage, version=0):
    """
    lah added
    """
    
    #print('write label')
    for dir in tqdm(image_files):
        label = np.loadtxt(dir[-1])
        #print(label.shape)
        if version==9 or version==10: #rsna实验，5标签分类转换为2标签分类。
            label[0] = max([label[i] for i in [0,2,3,4]])
            label = label[0:2]
            print(label.shape)
        label_storage.append(label[np.newaxis])
    pass


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype,
                        truth_only=False, save_truth=True):
    if not truth_only:
        data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    if save_truth:
        truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])

def get_cq500_input(shape, CT, ab, version=0):
    def get_left_right():
        left = np.zeros(ab.shape[1:])
        right = np.zeros(ab.shape[1:])
        for i in [0,2,4,8,10,13,15,17,18]:
            left = np.maximum(left, ab[i])
        for i in [1,3,5,9,11,14,16,17,19]:
            right = np.maximum(right, ab[i])
        return left, right
    data = np.zeros(shape)
    print('data version', version)
    """
    2021.10.27，发现之前（用76预测做输入，无脑室）生成data的时候，
    最后两个channel，即左侧和右侧异常，都是用的ab[22]和ab[23]？这是两个低密度我去。
    我当时怎么这样写的？我看结果AUC还说得过去（0.9左右）就没注意到这点。
    version=5效果很差，只有0.8多，我以为是因为模型不能自己把异常和脑区结合来判断左和右，结果是因为根本没给高密度的预测。
    而version1,2,3,4这些，我给最后两个channel本来是想给左侧高密度、右侧高密度，结果给的是两侧的低密度，也没有给左侧脑区和右侧脑区，
    结果模型还是能比较好地判断左出血、右出血这两个类别，说明即使不给左右，模型也能自己学出来区分图像左侧和右侧？
    """
    if version==0 or version==7 or version==9:
        data[0] = resize(CT, shape[1:])
        print(CT.shape, ab.shape)
        #print(np.max(CT_image.get_data()), np.min(CT_image.get_data()))
        #print(np.max(ab_images.get_data()), np.min(ab_images.get_data()))
        for i in range(4):
            data[i+1] = resize(ab[i+20], shape[1:])
    elif version==1 or version==2 or version==3:
        data[0] = resize(CT, shape[1:])
        left, right= get_left_right()
        for i in range(2):
            data[i+1] = resize(ab[i+20]*np.maximum(left, right), shape[1:])
        data[3] = resize(np.maximum(ab[20], ab[21])*left, shape[1:]) #之前写的22,23日了
        data[4] = resize(np.maximum(ab[20], ab[21])*right, shape[1:])
        print(np.min(data[0]), np.max(data[0])) #-0.6633898019790649 3.7402985095977783
        print(np.min(data[1:]), np.max(data[1:])) #0.0 0.7111441510694547
    elif version==4:
        left, right= get_left_right()
        for i in range(2):
            data[i] = resize(ab[i+20]*np.maximum(left, right), shape[1:])
        data[2] = resize(np.maximum(ab[20], ab[21])*left, shape[1:]) #之前写的22,23
        data[3] = resize(np.maximum(ab[20], ab[21])*right, shape[1:])
    elif version==5:
        left, right= get_left_right()
        data[0] = resize(ab[20], shape[1:]) #之前写的22
        data[1] = resize(ab[21], shape[1:]) #之前写的23
        data[2] = resize(left, shape[1:])
        data[3] = resize(right, shape[1:])
    elif version==6:
        data[0] = resize(CT, shape[1:])
    elif version==8:
        data[0] = resize(CT, shape[1:])
        for i in range(4):
            data[i+1] = resize(ab[i+20], shape[1:])
        data[5] = resize(np.maximum(ab[13], ab[14]), shape[1:])
    elif version==10:
        data[0] = resize(CT, shape[1:])
        left, right = get_left_right()
        for i in range(2):
            data[i+1] = resize(ab[i+20]*np.maximum(left,right), shape[1:])
    return data
        
def write_cq500_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True, rotate = False, version=0, n_channels = 5, n_labels = 4):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :param organ_label: 我加的，为了记录具体哪个结构有病。如果organ_label=True则会读取label.txt（一个20维01向量并保存到hdf5文件中）
    rotate:因为在训练过程中generator动态进行旋转增强，对CPU负担大，训练速度慢，所以如果rotate，则在生成数据时直接将
    图像随机旋转。这样主要的目的不是增加数据量，而是增加输入的角度多样性，对抗由图像歪斜导致的配准错误
    :return: Location of the hdf5 file with the image data written to it. 
    
    """
    n_samples = len(training_data_files)
    print('n channels', n_channels)
    try:
        hdf5_file, data_storage, truth_storage, affine_storage = \
        create_data_file(out_file, n_channels, n_samples, image_shape=image_shape, use_int32=False)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e
        
    for set_of_files in tqdm(training_data_files):
        CT_image = nib.load(set_of_files[0])
        ab_image = nib.load(set_of_files[1])
        data_shape = [n_channels]+list(image_shape)
        data = get_cq500_input(data_shape, CT_image.get_data(), ab_image.get_data(), version)
        data_storage.append(data[np.newaxis])
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', tables.UInt8Atom(), shape=[0, n_labels],
                                           expectedrows=n_samples) #expectedrows估计大概这个可扩展array有多少元素
    write_label_data_to_file(training_data_files, label_storage, version)
    if normalize:
        normalize_data_storage(data_storage)
    for i in range(5): #downstream任务看一下输入需不需要归一化
        print(data_storage[i].shape)
        print(np.min(data_storage[i][0]), np.max(data_storage[i][0])) #-0.66011333 3.8172038
        #print(np.min(data_storage[i][1:]), np.max(data_storage[i][1:])) #-0.12618046 162.51103
        #归一化会有奇怪问题，比如原始预测最大值都是0.7左右，归一化出来一个165一个225
        #或者原始预测最大值0.2和0.6，归一化出来都是60多。
        #因为输入已经是比较有意义的数值，而且范围都在正负几，不归一化了
    hdf5_file.close()

    return out_file

def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True, organ_label=False, rotate = False, truth_only=False,
                       use_cv2_resize = False, save_truth=True):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :param organ_label: 我加的，为了记录具体哪个结构有病。如果organ_label=True则会读取label.txt（一个20维01向量并保存到hdf5文件中）
    rotate:因为在训练过程中generator动态进行旋转增强，对CPU负担大，训练速度慢，所以如果rotate，则在生成数据时直接将
    图像随机旋转。这样主要的目的不是增加数据量，而是增加输入的角度多样性，对抗由图像歪斜导致的配准错误
    :return: Location of the hdf5 file with the image data written to it. 
    
    truth_only=True则不保存CT图像本身，但是给file.root.data还是有的，只是没存东西
    """
    n_samples = len(training_data_files)
    if organ_label: #最后一个是label.txt
        n_channels = len(training_data_files[0]) - 2
    else:
        n_channels = len(training_data_files[0]) - 1
    if not save_truth:
        n_channels += 1
    print('n channels', n_channels)
    try:
        hdf5_file, data_storage, truth_storage, affine_storage = \
        create_data_file(out_file, n_channels, n_samples, image_shape=image_shape, use_int32=organ_label)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop,
                             organ_label=organ_label, rotate = rotate, truth_only=truth_only,
                             use_cv2_resize=use_cv2_resize, save_truth=save_truth)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if organ_label: #lah added
        #When creating EArrays, you need to set one of the dimensions of the Atom instance to zero
        label = np.loadtxt(training_data_files[0][-1])
        shape = [0]+list(label.shape)
        print('label shape', shape)
        label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', tables.UInt8Atom(), shape=shape,
                                           expectedrows=n_samples) #expectedrows估计大概这个可扩展array有多少元素
        #写data的时候应该没有使用shuffle，所以label部分可以和data、truth等部分分开写，只要以同样的顺序写
        write_label_data_to_file(training_data_files, label_storage)
    if not truth_only and normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
