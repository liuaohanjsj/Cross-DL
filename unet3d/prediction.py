import os

import nibabel as nib
import numpy as np
import tables
from tqdm import tqdm
from pathlib import Path
import cv2
import keras.backend as K
from .training import load_old_model
from .utils import pickle_load
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data

def resolve_subject_id(ID, rsna=False):
    """
    2020.12.26
    ID是生成h5数据的时候的subject_id，之前都是reportID_seriesID格式的。
    但是现在用多个文件夹中的原始数据生成h5，subject_id为真实的序列路径。
    要从ID中分离出reportID和seriesID并返回
    ID是str
    rsna的比较特殊，用前一层而不是前两层
    """
    if ID[0]!='/': #不是/home/..这种全路径，那应该就是reportID_seriesID
        return ID.split('_')
    if rsna:
        return [Path(ID).parent.name, Path(ID).name]
    return [Path(ID).parent.parent.name, Path(ID).name]

def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        print(prediction[0][0][100][100][16], prediction[0][1][100][100][16])
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        #label_arrays.append(np.array(label_data, dtype=np.uint8)) #原始
        label_arrays.append(np.array(prediction[sample_number][1], dtype=np.float32)) #输出实际预测值，便于观察
        
    return label_arrays
    

def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    print(prediction.shape) #(1,2,256,256,32)
    print(label_map) #True
    print(labels)
    if prediction.shape[1] == 1: #输出为1通道时采用此方法
        data = prediction[0, 0]
        '''
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
        '''
    elif prediction.shape[1] > 1:
        if label_map: #使用此方法
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False,
                        refined = False, multitask = False, multiinput = None, truth_file = None, label_file = None,
                        run=True, flip = False, save_truth_file = True, ab_seg_file=None):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    这个threshold好像不是控制2分类的阈值的（例如，class1的置信度>threshold时才认为是1，否则是0）。
    而是用来多分类（包括2分类）时控制是否输出label的。它的代码好像认为0不是一个label，例如预测label 1,2,3，
    置信度为0.3,0.4,0.3，不超过threshold，那就输出0，表示不是一个label。
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    multiinput只是在multiinput模型中用的，为了喂给model正确的输入
    label_file是在不用data的h5文件中自带的label时（比如用高密度灶的label时）使用额外的h5文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not label_file:
        label = data_file.root.label[data_index]
    else:
        label = label_file.root.label[data_index]
    if flip:
        s_index = [1,0,3,2,5,4,6,7,9,8,11,10,12,13,14,16,15,17,19,18]
        mem = label.copy()
        for i in range(20):
            label[i] = mem[s_index[i]]
    #print(data_index, label[12,3])
    np.savetxt(output_dir+'/label.txt', label)
    
    if ab_seg_file and np.sum(ab_seg_file.root.truth[data_index])>0:
        print('save abnormality seg')
        image = nib.Nifti1Image(ab_seg_file.root.truth[data_index], data_file.root.affine[data_index])
        image.to_filename(os.path.join(output_dir, "ab_seg.nii.gz"))
    #if not run or not (ab_seg_file and np.sum(ab_seg_file.root.truth[data_index])>0):
    if not run:
        print('return')
        return
    
       
    
    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    #print('test data', test_data.shape)
    
    if flip:
        test_data = np.flip(test_data, 2)
    
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))
        #temp = test_data[0,i]
        #temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))*255
        #for j in range(temp.shape[2]):
        #    cv2.imwrite(output_dir+'/temp%d.png'%j, temp[:,:,j])
    if save_truth_file:
        if truth_file:
            truth = truth_file.root.truth[data_index][0]
        else:
            truth = data_file.root.truth[data_index][0]
        if flip:
            truth = np.flip(truth, 0)
        test_truth = nib.Nifti1Image(truth, affine)
        test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))
        
    if multiinput is not None:
        patch_shape = tuple([int(dim) for dim in model.input[0].shape[-3:]])
    else:
        patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    
    
    if patch_shape == test_data.shape[-3:]:
        if multiinput is not None:
            if truth_file is not None:
                test_truth = np.asarray([truth_file.root.truth[data_index]]) #(1,1,n,n,m)
            else:
                test_truth = np.asarray([data_file.root.truth[data_index]]) #(1,1,n,n,m)
            #这部分左右翻转尚未改
            mask = np.zeros(test_truth.shape, np.uint32)
            print(multiinput)
            print('shape', test_data.shape, mask.shape)
            for i in range(20):
                if multiinput[i]:
                    temp = np.bitwise_and(np.right_shift(test_truth, i), 1)
                    mask = np.bitwise_or(mask, np.int8(temp))
            print(np.sum(mask))
            #prediction = predict(model, [test_data], permute=permute)
            prediction = predict(model, [test_data, mask], permute=permute)
        else:
            prediction = predict(model, test_data, permute=permute)
            
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)[np.newaxis]
    #print('predicton', prediction[0].shape)
    
    if not multitask:
        prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)
    else:
        if not isinstance(prediction, list): #单一输出，分类
            print(prediction.shape)
            np.savetxt(output_dir+'/pred.txt', prediction[0])
            loc = gradCAM(model, test_data)
            prediction_image = nib.Nifti1Image(loc[0], affine)
            prediction_image.to_filename(os.path.join(output_dir, "GC.nii.gz"))

        if prediction[0].shape[1]<=25:
            prediction_image = nib.Nifti1Image(prediction[0][0], affine)
        else: #multi scale
            scale = prediction[0].shape[1]//25
            shape = list(prediction[0].shape[2:])
            shape = [x//(2**(scale-1)) for x in shape]
            prediction_image = []
            for i in range(0, scale):
                temp = prediction[0][0, i*25:(i+1)*25, :shape[0], :shape[1], :shape[2]]
                prediction_image.append(nib.Nifti1Image(temp, affine))
                shape = [x*2 for x in shape]
        #predicted_label = prediction[1][0]
        #print('save predicted labels', predicted_label)
        #np.savetxt(output_dir+'/predicted_label.txt', predicted_label)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    elif prediction_image is not None:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))

def run_validation_case_cq500(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False,
                        refined = False, multitask = False, multiinput = None, truth_file = None, label_file = None,
                        run=True, flip = False, save_truth_file=True):
    """
    有一些参数没用，只是为了和run_validation_case保持一致
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not label_file:
        label = data_file.root.label[data_index]
    else:
        label = label_file.root.label[data_index]
    np.savetxt(output_dir+'/label.txt', label)
    if not run:
        return
    test_data = np.asarray([data_file.root.data[data_index]])
    #for i, modality in enumerate(training_modalities):
    #    image = nib.Nifti1Image(test_data[0, i], np.zeros((3,3)))
    #    image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))
    prediction = predict(model, test_data, permute=False)
    np.savetxt(output_dir+'/prediction.txt', prediction)
    
def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False, num = None, paths=[],
                         refined = False, multitask = False, multiinput = None, truth_file = None, label_file = None,
                         run = True, filt = [], load_loss_weight = False, report_patient = {}, full_version = None, flip = False,
                         downstream = False, rsna=False, save_truth_file=True, ab_seg_file=None):
    print('downstream', downstream)
    ret_paths = []
    validation_indices = pickle_load(validation_keys_file)
    if num:
        if type(num) is not slice:
            num = slice(num.start, min(num.stop, len(validation_indices)))
        validation_indices = validation_indices[num]
    if run:
        model,_ = load_old_model(model_file, load_loss_weight, full_version = full_version)
    else:
        model = None
    data_file = tables.open_file(hdf5_file, "r")
    if truth_file:
        truth_file = tables.open_file(truth_file, "r")
    if ab_seg_file:
        ab_seg_file = tables.open_file(ab_seg_file, "r")
    label_file_opened = None
    if label_file:
        print('label file', label_file)
        label_file_opened = tables.open_file(label_file, "r")
    
    filt_patient = {} #训练集出现的病人
    for x in filt:
        if x[0] in report_patient:
            filt_patient[report_patient[x[0]][0]] = True
    #print('filt_patient', filt_patient)
    for index in tqdm(validation_indices):
        if 'subject_ids' in data_file.root:
            ID = data_file.root.subject_ids[index].decode('utf-8')
            IDs = resolve_subject_id(ID, rsna=rsna)
            in_train = False
            for x in filt:
                if IDs[0]==x[0] or IDs[1]==x[1] or (IDs[0] in report_patient and report_patient[IDs[0]][0] in filt_patient):
                    in_train = True
            if in_train:
                continue
            temp = IDs[0]+'_'+IDs[1]
            if ID[0]=='/':
                temp = Path(ID).parent.parent.parent.name+'_'+temp
            if downstream:
                temp = Path(ID).name
            #print(temp)
            case_directory = os.path.join(output_dir, temp)
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        if len(paths) and case_directory not in paths:
            continue
        ret_paths.append(case_directory)
        #print('output dir', case_directory)
        if downstream:
            f = run_validation_case_cq500
        else:
            f = run_validation_case
        f(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
          training_modalities=training_modalities, output_label_map=output_label_map, labels=labels,
          threshold=threshold, overlap=overlap, permute=permute, refined=refined, multitask=multitask,
          multiinput = multiinput, truth_file=truth_file, label_file = label_file_opened, run=run,
          flip=flip, save_truth_file=save_truth_file, ab_seg_file=ab_seg_file)
    data_file.close()
    if truth_file:
        truth_file.close()
    if label_file:
        label_file_opened.close()
    return ret_paths

def run_single_case(data_file, index, model):
    """
    lah added
    """
    test_data = np.asarray([data_file.root.data[index]])
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=False)
    else: #这个分支应该不用才对
        print('using the wrong branch')
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=0, permute=False)[np.newaxis]
    return prediction[0][0]

    
def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)

def gradCAM(model, data):
    #model.summary()
    #x = input()
    try:
        score = model.get_layer("reshape_1").output #[?,20,4]，sigmoid之前的score
    except:
        print('wrong')
        score = -K.log(1/model.output-1) #[?,20,4]，sigmoid之前的score
    
    #conv_layer = model.get_layer("leaky_re_lu_10") #[10,10,5]，太小了，效果不好。0.01-0.02
    #conv_layer = model.get_layer("leaky_re_lu_8") #[20,20,10] 0.03-0.04
    conv_layer = model.get_layer("leaky_re_lu_6") #[20,20,10] 0.05-0.1
    grads = []
    for i in range(4):
        #12是scan-level预测
        gradient = K.gradients(score[:,12,i], conv_layer.output)[0] #[?,D,10,10,5]。返回的是个tensor的list，要[0]才能取tensor，不知道为什么
        #print(gradient.shape)
        weight = K.mean(gradient, axis=(2,3,4), keepdims=True) #[?,D,1,1,1]
        grads.append(weight)
        #print(weight.shape)
    iterate = K.function([model.input], grads+[conv_layer.output])
    values = iterate([data]) #[B,D,1,1,1]*4 + [B,D,10,10,5]
    weights = np.stack(values[:4], axis=2) #[B,D,4,1,1,1]
    '''
    for i in range(4):
        print('weight', i)
        print(weights[:,:,i,0,0,0])
    x = input()
    '''
    #print(weight_value.shape, conv_layer_value.shape)
    weighted = weights * values[4][:,:,np.newaxis] #[B,D,4,10,10,5]
    result = np.mean(np.maximum(weighted, 0), axis=1) #[B,4,10,10,5]
    print(result.shape)
    return result

    
def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)
