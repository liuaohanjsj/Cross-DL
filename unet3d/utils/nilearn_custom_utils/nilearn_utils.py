import numpy as np
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to
import time


def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """
    ti = time.time()
    img = check_niimg(img)
    data = img.get_data()
    print('1', time.time()-ti)
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
    print('2', time.time()-ti)
    print(data.ndim)
    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    print('2.3', time.time()-ti)
    coords = np.array(np.where(passes_threshold))
    print(passes_threshold.shape, coords.shape)
    print('2.5', time.time()-ti)
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    print('3', time.time()-ti)
    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices
    print('4', time.time()-ti)
    ret = crop_img_to(img, slices, copy=copy)
    print('5', time.time()-ti)
    return ret
