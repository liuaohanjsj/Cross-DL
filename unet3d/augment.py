import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
import cv2
import PIL

def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)

def align(img, x, y, angle):
    '''
    这个函数是从中轴线对齐直接复制过来的，在这里没有用，只是参考一下改代码
    img是一张图像，绕(x,y)转angle度，然后再将(x,y)平移到中心
    img可以使任意数量通道的，opencv应该是支持不同通道数量的array进行计算。
    '''
    n, m = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((x, y), angle, 1)
    img = cv2.warpAffine(img, mat, (m, n))
    mat = np.float32([[1,0,m/2-x],[0,1,n/2-y]]) #opencv要求仿射矩阵必须是cv_32f或cv_64f类型
    img = cv2.warpAffine(img, mat, (m, n))
    return img
    

def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None):
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    return image

def rotate_image(image, angle, interpolation):
    fill = np.min(image)
    #print('fill', fill, type(fill))
    n, m, l = image.shape
    ret = np.copy(image)
    for i in range(l):
        temp = image[:,:,i]
        img = PIL.Image.fromarray(temp)
        #img = img.rotate(angle, resample = interpolation, fillcolor = fill)
        #不知道为什么加fillcolor就崩，可能是uint型数组不能这样fillcolor？
        img = img.rotate(angle, resample = interpolation)
        
        #print(type(img))
        ret[:,:,i] = np.asarray(img)
    return ret

def my_augment_data(data, truth):
    """
    我实在是不知道他那个augment_data里面的affine有啥用，训练需要那个东西吗？？
    """
    angle = random.randint(-10,10)
    #print('data shape', data.shape) #(1,160,160,80)
    #print('truth shape', truth.shape) #(160,160,80)
    for i in range(data.shape[0]):
        data[i] = rotate_image(data[i], angle, PIL.Image.BILINEAR)
    if len(truth.shape)>1: #如果truth用的话
        truth = rotate_image(truth, angle, PIL.Image.NEAREST)
    #print('result data shape', data.shape)
    #print('result truth shape', truth.shape)
    return data, truth

def augment_data(data, truth, affine, scale_deviation=None, flip=True):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    data_list = list()
    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        #我实在是看不懂这个resample_to_img是干啥的啊，还有为什么非要affine啊，这个数组有任何意义吗？？？
        data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis,
                                                       scale_factor=scale_factor), image,
                                         interpolation="continuous").get_data())
    data = np.asarray(data_list)
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis, scale_factor=scale_factor),
                                 truth_image, interpolation="nearest").get_data()
    
    return data, truth_data


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def reverse_permute_data(data, key):
    key = reverse_permutation_key(key)
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    if flip_z:
        data = data[:, :, :, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, ::-1]
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    return data


def reverse_permutation_key(key):
    rotation = tuple([-rotate for rotate in key[0]])
    return rotation, key[1], key[2], key[3], key[4]
