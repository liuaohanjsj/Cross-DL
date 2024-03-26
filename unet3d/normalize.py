import os
import time
import cv2

import numpy as np
from nilearn.image import new_img_like
import nibabel as nib

from unet3d.utils.utils import resize, read_image_files
from .utils import crop_img, crop_img_to, read_image #这个crop_img在utils的init里面


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files) 
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    ti = time.time()
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)#in_files中所有序列、所有模态有某一个不为0的mask
    else: #reslice_image_set调用时in_files作为一个长为1的list，所以应该是这个分支
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    #原来只返回slices，我为了修改crop时知道原图像大小，增加了图像shape的返回值
    print('get foreground', time.time()-ti)
    return crop_img(foreground, return_slices=True, copy=True), foreground.get_data().shape

def cv2_resize(img, shape, inter=cv2.INTER_NEAREST):
    """
    三维resize
    """
    print(shape)
    dtype = img.dtype
    img = np.float32(img)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=inter) #行列resize
    img = np.transpose(img, (0,2,1)) #160,5,160
    img = cv2.resize(img, (shape[2], shape[0]), interpolation=inter) #纵向resize
    img = np.transpose(img, (0,2,1))
    return img.astype(dtype)

def my_reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False, return_crop=False):
    images = reslice_image_set(in_files, None, out_files, label_indices, crop, return_crop)
    ret = [cv2_resize(image.get_data(), image_shape, cv2.INTER_LINEAR) for image in images]
    ret = [nib.Nifti1Image(ret[i], images[i].affine) for i in range(len(images))]
    return ret

def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False, return_crop=False):
    ti = time.time()
    if crop:
        crop_slices, shape = get_cropping_parameters([in_files])
        #print('fuck you', time.time()-ti)
        #[slice(81, 391, None), slice(50, 437, None), slice(0, 21, None)]
        #得到的slice在宽和高方向是不相同的，一般更高一些，导致图像crop后会有形变
        #2020.11.4准备在crop的时候保持图像的长宽比。
        w = crop_slices[0].stop-crop_slices[0].start
        h = crop_slices[1].stop-crop_slices[1].start
        midx = (crop_slices[0].stop+crop_slices[0].start)//2
        midy = (crop_slices[1].stop+crop_slices[1].start)//2
        if w<h: #大部分情况，根据高增加宽
            start = midx-h//2
            stop = midx+h//2
            if start<0: #增宽后超出左边界，向右移
                stop -= start
                start = 0
            if stop>shape[0]: #增宽后超出右边界，向左移
                start -= stop-shape[0]
                stop = shape[0]
            if start<0: #如果还是不行，那就是图像本身太瘦了，一般不会出现
                print('image too thin', shape, w, h)
                start = 0
            crop_slices[0] = slice(start, stop, None)
        else:
            start = midy-w//2
            stop = midy+w//2
            if start<0:
                stop -= start
                start = 0
            if stop>shape[1]:
                start -= stop-shape[1]
                stop = shape[1]
            if start<0:
                print('image too flat', shape, w, h)
                start = 0
            crop_slices[1] = slice(start, stop, None)
        if crop_slices[1].stop==512:
            print('crop slices', crop_slices, in_files[0])
    else:
        crop_slices = None
    
    ti = time.time()
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    print('read image files', time.time()-ti)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        if return_crop:
            return images, crop_slices
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    for i, image_file in enumerate(set_of_files):
        ti = time.time()
        image = read_image(image_file) #这里为了求crop已经读了一遍图片了，后面再读一遍没必要。虽然这里应该没有resize，但是可以改进
        #print('read image', time.time()-ti)
        ti = time.time()
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        #print('compute foreground', time.time()-ti)
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


