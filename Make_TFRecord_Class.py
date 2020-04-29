__author__ = 'Brian M Anderson'
# Created on 4/7/2020

import tensorflow as tf
import os, pickle
import time
from .Image_Processors.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from .Image_Processors.Image_Processors_TFRecord import *
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


def return_feature(data):
    if type(data) is int:
        return _int64_feature(tf.constant(data, dtype='int64'))
    elif type(data) is np.ndarray:
        return _bytes_feature(data.tostring())
    elif type(data) is str:
        return _bytes_feature(tf.constant(data))
    elif type(data) is np.float32:
        return _float_feature(tf.constant(data, dtype='float32'))


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


def return_example_proto(base_dictionary, image_dictionary_for_pickle={}):
    feature = {}
    for key in base_dictionary:
        data = base_dictionary[key]
        if type(data) is int:
            feature[key] = _int64_feature(tf.constant(data, dtype='int64'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.int64)
        elif type(data) is np.ndarray:
            feature[key] = _bytes_feature(data.tostring())
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
        elif type(data) is str:
            feature[key] = _bytes_feature(tf.constant(data))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
        elif type(data) is np.float32:
            feature[key] = _float_feature(tf.constant(data, dtype='float32'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.float32)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def serialize_example(image_path, annotation_path, overall_dict={}, image_processors=None):
    base_dictionary = get_features(image_path,annotation_path, image_processors=image_processors)
    for image_key in base_dictionary:
        overall_dict['{}_{}'.format(image_path, image_key)] = base_dictionary[image_key]


def get_features(image_path, annotation_path, image_processors=None):
    image_handle, annotation_handle = sitk.ReadImage(image_path), sitk.ReadImage(annotation_path)
    base_features = OrderedDict()
    annotation = sitk.GetArrayFromImage(annotation_handle).astype('int8')
    image = sitk.GetArrayFromImage(image_handle).astype('float32')
    base_features['image'] = image
    base_features['annotation'] = annotation
    base_features['image_path'] = image_path
    base_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
    features = {'Base':base_features}
    if image_processors is not None:
        for image_processor in image_processors:
            features_keys = list(features.keys())
            for key in features_keys:
                features[key] = image_processor.parse(features[key])
    return features['Base']


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


def write_tf_record(path, record_name=None, rewrite=False, thread_count=int(cpu_count() * .9 - 1),
                    is_3D=True, extension=np.inf, shuffle=False, image_processors=None):
    '''
    :param path: path to where Overall_Data and mask files are located
    :param record_name: name of record, without .tfrecord attached
    :param rewrite: Do you want to rewrite old records? True/False
    :param thread_count: specify 1 if debugging
    :param wanted_values_for_bboxes: A list of values that you want to calc bbox for [1,2,etc.]
    :param extension: extension above and below annotation, recommend np.inf for validation and test
    :param is_3D: Take the whole patient or break up into 2D images
    :param shuffle: shuffle the output examples? Can be useful to allow for a smaller buffer without worrying about distribution
    :param image_processors: a list of image processes that can take the image and annotation dictionary, follow the
    :return:
    '''
    start = time.time()
    if image_processors is None:
        if is_3D:
            image_processors = [Clip_Images_By_Extension(extension=extension), Distribute_into_3D()]
        else:
            image_processors = [Clip_Images_By_Extension(extension=extension), Distribute_into_2D()]
    add = ''
    if record_name is None:
        record_name = 'Record'
        add = '_2D'
        if is_3D:
            add = '_3D'
    else:
        record_name = record_name.split('.tfrecord')[0]
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
        image_path, annotation_path = data_dict['Images'][iteration], data_dict['Annotations'][iteration]
        item = {'image_path':image_path,'annotation_path':annotation_path,'overall_dict':overall_dict,
                'image_processors':image_processors}
        print(image_path)
        q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    print('Writing record...')
    examples = 0
    writer = tf.io.TFRecordWriter(filename)
    dict_keys = overall_dict.keys()
    if shuffle:
        dict_keys = np.asarray(list(overall_dict.keys()))
        perm = np.arange(len(dict_keys))
        np.random.shuffle(perm)
        dict_keys = dict_keys[perm]
    features = OrderedDict()
    for image_path in dict_keys:
        example_proto = return_example_proto(overall_dict[image_path], features)
        writer.write(example_proto.SerializeToString())
        examples += 1
    writer.close()
    fid = open(filename.replace('.tfrecord','_Num_Examples.txt'),'w+')
    fid.write(str(examples))
    fid.close()
    save_obj(filename.replace('.tfrecord','_features.pkl'),features)
    print('Finished!')
    stop = time.time()
    print('Took {} seconds'.format(stop-start))
    return None


if __name__ == '__main__':
    pass
