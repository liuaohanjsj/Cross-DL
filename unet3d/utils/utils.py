import pickle
import os
import collections
from pathlib import Path

import nibabel as nib
import numpy as np
import time
from nilearn.image import reorder_img, new_img_like

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    '''
    files = []
    if (Path(image_files[0]).parent/'series_link.txt').exists():
        with open(str((Path(image_files[0]).parent/'series_link.txt')), 'r') as fp:
            link = fp.read().strip()
        for x in image_files:
            temp = x.replace(str(Path(x).parent), link)
            files.append(temp) #这里应该保证str(Path)得到的斜杠和原始的斜杠（os）是相同的？
            print('replace')
            print(x)
            print(temp)
            print(str(Path(x).parent))
            print(link)
    else:
        for x in image_files:
            files.append(x)
    print(Path(image_files[0]).parent)
    '''
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))
    return image_list

def stw(img):
    ti = time.time()
    img = np.minimum(np.maximum(img, -10), 80)
    #print('1', time.time()-ti)
    #img = (img+10)*255/90
    img = (img+10) #shabi的真的，有的时候不知道是数据服务器读数据的问题，还是服务器内存cpu的问题，上面那个就超慢，下面就快，差十几秒，傻逼
    #print('2', time.time()-ti)
    return img

def check_image_exist(in_file):
    """
    2023.3.20 因为我使用多个脑区分割GT配准时只配准了一部分。生成数据时要检查一下是否有分割配准，没有就不执行操作了，节省时间和空间
    """
    if (Path(in_file).parent/'series_link.txt').exists(): #当in_file是其他配准mask的时候，肯定不存在这个link。不用管，到下面处理
        with open(str((Path(in_file).parent/'series_link.txt')), 'r') as fp:
            link = fp.read().strip()
        in_file = in_file.replace(str(Path(in_file).parent), link)
    return Path(in_file).exists()

def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    #print("Reading: {0}".format(in_file))
    #in_file应该是nii.gz文件路径，一般情况下此路径都会存在，可以直接读取
    #但是在series_link情况下，此路径实际不存在，需要根据series_link.txt中路径进行替换
    if (Path(in_file).parent/'series_link.txt').exists(): #当in_file是其他配准mask的时候，肯定不存在这个link。不用管，到下面处理
        with open(str((Path(in_file).parent/'series_link.txt')), 'r') as fp:
            link = fp.read().strip()
        in_file = in_file.replace(str(Path(in_file).parent), link)
    ti = time.time()
    try:
        image = nib.load(os.path.abspath(in_file)) #使用nib读取的
        temp = image.get_data() #nib读出来已经是减1024之后的，而且是float
        print('nib shape', temp.shape, in_file, temp.dtype)
    except: #2023.3.18因为我用其他脑区分割文件只生成了一部分，会有文件不存在
        if (Path(in_file).parent.parent/'series_link.txt').exists():
            with open(str((Path(in_file).parent.parent/'series_link.txt')), 'r') as fp:
                link = fp.read().strip()
        else:
            link = str(Path(in_file).parent.parent)
        image = nib.load(link+'/CT_brain.nii.gz')
        temp = np.zeros(image.shape, np.uint32) #实际要的是mask，但是没有生成mask。所以用全0
        image = nib.Nifti1Image(temp, image.affine)
        print('not found file')
        #masknii = nib.Nifti1Image(temp, nii.affine)
    #print('get image data', time.time()-ti)
    #print('read image type', temp.dtype)
    #软组织窗：
    if np.min(temp)!=0: #CT, not mask
        print('CT', np.max(temp), np.min(temp))
        remove_skull = False #之前一直是False，10.28改成True了。
        ti1 = time.time()
        if remove_skull:
            mask = np.float32(temp<100)
            temp = temp*mask
        temp = stw(temp)
        #print('remove skull', time.time()-ti1)
        ti2 = time.time()
        image = nib.Nifti1Image(temp, image.affine)
        #print('nib image', time.time()-ti2)
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    ti = time.time()
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    print('resize', time.time()-ti)
    return new_img_like(image, new_data, affine=new_affine)

def maxpool(img, stride):
    """
    lah added
    img:三维图像,np.array
    stride=size
    目前只考虑整除的情况。
    """
    shape = list(img.shape)
    for i in range(len(shape)):
        shape[i] //= stride
    print(shape)
    ret = np.zeros(shape, img.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ret[i,j,k] = np.max(img[i*stride:(i+1)*stride, j*stride:(j+1)*stride, k*stride:(k+1)*stride])
    return ret
