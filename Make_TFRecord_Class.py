__author__ = 'Brian M Anderson'
# Created on 4/7/2020

import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os, pickle
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from _collections import OrderedDict
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return None


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_features(values):
    return tf.train.Features(float_list=tf.train.FloatList(values=[values]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def return_example_proto(base_dictionary):
    feature = {}
    for key in base_dictionary:
        data = base_dictionary[key]
        if type(data) is int:
            feature[key] = _int64_feature(data)
        elif type(data) is np.ndarray:
            feature[key] = _bytes_feature(data.tostring())
        elif type(data) is str:
            feature[key] = _bytes_feature(tf.constant(data))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def serialize_example(image_path, annotation_path, overall_dict={}, wanted_values_for_bboxes=None, extension=np.inf,
                      is_3D=True, max_z=np.inf, chop_ends=False):
    base_dictionary = get_features(image_path,annotation_path,extension=extension,chop_ends=chop_ends,
                                   wanted_values_for_bboxes=wanted_values_for_bboxes, is_3D=is_3D, max_z=max_z)
    for image_key in base_dictionary:
        example_proto = return_example_proto(base_dictionary[image_key])
        overall_dict['{}_{}'.format(image_path, image_key)] = example_proto.SerializeToString()


def get_bounding_boxes(annotation_handle,value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(annotation_handle,lowerThreshold=value,upperThreshold=value+1)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    stats.Execute(connected_image)
    bounding_boxes = np.asarray([stats.GetBoundingBox(l) for l in stats.GetLabels()]).astype('int32')
    volumes = np.asarray([stats.GetPhysicalSize(l) for l in stats.GetLabels()]).astype('float32')
    return bounding_boxes, volumes


def return_image_feature_description(wanted_values_for_bboxes=None, is_3D=True):
    if is_3D:
        image_feature_description = {
            'image_path': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'annotation': tf.io.FixedLenFeature([], tf.string),
            'start': tf.io.FixedLenFeature([], tf.int64),
            'stop': tf.io.FixedLenFeature([], tf.int64),
            'z_images': tf.io.FixedLenFeature([], tf.int64),
            'rows': tf.io.FixedLenFeature([], tf.int64),
            'cols': tf.io.FixedLenFeature([], tf.int64),
            'spacing': tf.io.FixedLenFeature([], tf.string)
        }
        if wanted_values_for_bboxes is not None:
            for val in wanted_values_for_bboxes:
                image_feature_description['bounding_boxes_{}'.format(val)] = tf.io.FixedLenFeature([], tf.string)
                image_feature_description['volumes_{}'.format(val)] = tf.io.FixedLenFeature([], tf.string)
    else:
        image_feature_description = {
            'image_path': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'annotation': tf.io.FixedLenFeature([], tf.string),
            'rows': tf.io.FixedLenFeature([], tf.int64),
            'cols': tf.io.FixedLenFeature([], tf.int64),
            'spacing': tf.io.FixedLenFeature([], tf.string)
        }
    return image_feature_description


def get_start_stop(annotation, extension=np.inf):
    non_zero_values = np.where(np.max(annotation,axis=(1,2)) > 0)[0]
    start, stop = -1, -1
    if non_zero_values.any():
        start = int(non_zero_values[0])
        stop = int(non_zero_values[-1])
        start = max([start - extension, 0])
        stop = min([stop + extension, annotation.shape[0]])
    return start, stop


def get_features(image_path, annotation_path, extension=np.inf, wanted_values_for_bboxes=None,
                 is_3D=True, max_z=np.inf, chop_ends=False):
    image_handle, annotation_handle = sitk.ReadImage(image_path), sitk.ReadImage(annotation_path)
    features = OrderedDict()
    annotation = sitk.GetArrayFromImage(annotation_handle).astype('int8')
    start, stop = get_start_stop(annotation, extension)
    image = sitk.GetArrayFromImage(image_handle).astype('float32')
    if start != -1 and stop != -1:
        image, annotation = image[start:stop,...], annotation[start:stop,...]
    z_images_base, rows, cols = annotation.shape
    image_base, annotation_base = image.astype('float32'), annotation.astype('int8')
    if is_3D:
        start_chop = 0
        step = min([max_z, z_images_base])
        for index in range(z_images_base//step+1):
            image_features = OrderedDict()
            if start_chop >= z_images_base:
                continue
            image_size, rows, cols = annotation_base[start_chop:start_chop+step,...].shape
            if chop_ends and image_size < step:
                continue
            annotation = annotation_base[start_chop:start_chop+step,...]
            start, stop = get_start_stop(annotation, extension)
            image_features['image_path'] = image_path
            image = image_base[start_chop:start_chop+step,...]
            image_features['image'] = image
            image_features['annotation'] = annotation
            image_features['start'] = start
            image_features['stop'] = stop
            image_features['z_images'] = image_size
            image_features['rows'] = rows
            image_features['cols'] = cols
            image_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
            start_chop += step
            if wanted_values_for_bboxes is not None:
                for val in wanted_values_for_bboxes:
                    slices = np.where(annotation == val)
                    image_features['volumes_{}'.format(val)] = np.asarray([0])
                    if slices:
                        bounding_boxes, volumes = get_bounding_boxes(sitk.GetImageFromArray(annotation), val)
                        image_features['bounding_boxes_{}'.format(val)] = bounding_boxes
                        image_features['volumes_{}'.format(val)] = volumes
            features['Image_{}'.format(index)] = image_features
    else:
        for index in range(z_images_base):
            image_features = OrderedDict()
            image_features['image_path'] = image_path
            image_features['image'] = image[index]
            image_features['annotation'] = annotation[index]
            image_features['rows'] = rows
            image_features['cols'] = cols
            image_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')[:-1]
            features['Image_{}'.format(index)] = image_features
    return features


def worker_def(A):
    q = A[0]
    base_class = serialize_example
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                base_class(**item)
            except:
                print('Failed?')
            q.task_done()


def return_parse_function(image_feature_description):

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)
    return _parse_image_function


def read_dataset(filename, features):
    raw_dataset = tf.data.TFRecordDataset([filename], num_parallel_reads=tf.data.experimental.AUTOTUNE)
    parsed_image_dataset = raw_dataset.map(return_parse_function(features))


def write_tf_record(path, record_name='Record', rewrite=False, thread_count=int(cpu_count() * .9 - 1),
                    wanted_values_for_bboxes=None, extension=np.inf, is_3D=True, max_z=np.inf, chop_ends=False):
    add = '_2D'
    if is_3D:
        add = '_3D'
        if max_z != np.inf:
            add += '_{}maxz'.format(max_z)
            if chop_ends:
                add += '_chopends'
    filename = os.path.join(path,'{}{}.tfrecord'.format(record_name,add))
    if os.path.exists(filename) and not rewrite:
        return None
    data_dict = {'Images':{}, 'Annotations':{}}
    image_files = [i for i in os.listdir(path) if i.find('Overall_Data') == 0]
    for file in image_files:
        iteration = file.split('_')[-1].split('.')[0]
        data_dict['Images'][iteration] = os.path.join(path,file)

    annotation_files = [i for i in os.listdir(path) if i.find('Overall_mask') == 0]
    for file in annotation_files:
        iteration = file.split('_y')[-1].split('.')[0]
        data_dict['Annotations'][iteration] = os.path.join(path,file)
    overall_dict = {}
    q = Queue(maxsize=thread_count)
    A = [q,]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    for iteration in list(data_dict['Images'].keys()):
        print(iteration)
        image_path, annotation_path = data_dict['Images'][iteration], data_dict['Annotations'][iteration]
        item = {'image_path':image_path,'annotation_path':annotation_path,'overall_dict':overall_dict,
                'wanted_values_for_bboxes':wanted_values_for_bboxes, 'extension':extension, 'is_3D':is_3D,
                'max_z':max_z, 'chop_ends':chop_ends}
        q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    print('Writing record...')
    examples = 0
    writer = tf.io.TFRecordWriter(filename)
    for image_path in overall_dict.keys():
        writer.write(overall_dict[image_path])
        examples += 1
    writer.close()
    fid = open(filename.replace('.tfrecord','_Num_Examples.txt'),'w+')
    fid.write(str(examples))
    fid.close()
    features = return_image_feature_description(wanted_values_for_bboxes, is_3D=is_3D)
    save_obj(filename.replace('.tfrecord','_features.pkl'),features)
    print('Finished!')
    return None


if __name__ == '__main__':
    pass
