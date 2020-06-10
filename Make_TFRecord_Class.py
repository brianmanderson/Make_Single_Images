__author__ = 'Brian M Anderson'
# Created on 4/7/2020

import time
from .Image_Processors_Module.Image_Processors_TFRecord import *
from .Image_Processors_Module.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from _collections import OrderedDict
from threading import Thread
from multiprocessing import cpu_count
from queue import *


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
                print('Failed? {}'.format(item))
            q.task_done()


def write_tf_record(niftii_path, out_path=None, rewrite=False, thread_count=int(cpu_count() * .5),
                    is_3D=True, extension=np.inf, image_processors=None, special_actions=False, verbose=False):
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
    if out_path is None:
        out_path = niftii_path
    if image_processors is None:
        if is_3D:
            image_processors = [Add_Images_And_Annotations(), Clip_Images_By_Extension(extension=extension), Distribute_into_3D()]
        else:
            image_processors = [Add_Images_And_Annotations(), Clip_Images_By_Extension(extension=extension), Distribute_into_2D()]
    if not special_actions and Add_Images_And_Annotations() not in image_processors:
        image_processors = [Add_Images_And_Annotations()] + image_processors

    has_writer = np.max([isinstance(i,Record_Writer) for i in image_processors])
    assert not has_writer, 'Just provide an out_path, the Record_Writer is already provided'
    data_dict = {'Images':{}, 'Annotations':{}}
    image_files = [i for i in os.listdir(niftii_path) if i.find('Overall_Data') == 0]
    for file in image_files:
        iteration = file.split('_')[-1].split('.')[0]
        data_dict['Images'][iteration] = os.path.join(niftii_path,file)

    annotation_files = [i for i in os.listdir(niftii_path) if i.find('Overall_mask') == 0]
    for file in annotation_files:
        iteration = file.split('_y')[-1].split('.')[0]
        data_dict['Annotations'][iteration] = os.path.join(niftii_path,file)
    q = Queue(maxsize=thread_count)
    A = [q,]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    iterations = list(data_dict['Images'].keys())
    for iteration in iterations:
        image_path, annotation_path = data_dict['Images'][iteration], data_dict['Annotations'][iteration]
        item = {'image_path':image_path,'annotation_path':annotation_path,
                'image_processors':image_processors, 'record_writer':Record_Writer(out_path),
                'verbose':verbose}
        image_name = os.path.split(image_path)[-1].split('.nii')[0]
        if not os.path.exists(os.path.join(out_path,'{}.tfrecord'.format(image_name))) or rewrite:
            print(image_path)
            q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    return None


if __name__ == '__main__':
    pass
