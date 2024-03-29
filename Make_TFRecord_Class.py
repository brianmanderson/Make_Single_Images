__author__ = 'Brian M Anderson'

# Created on 4/7/2020

from .Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *
from .Image_Processors_Module.src.Processors.TFRecordWriter import *
from .Image_Processors_Module.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from _collections import OrderedDict
from threading import Thread
import copy
from multiprocessing import cpu_count
from queue import *


def worker_def(a):
    q = a[0]
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


def return_data_dict(niftii_path):
    data_dict = {}
    image_files = [i for i in os.listdir(niftii_path) if i.find('Overall_Data') == 0]
    for file in image_files:
        iteration = file.split('_')[-1].split('.')[0]
        data_dict[iteration] = {'image_path': os.path.join(niftii_path, file),
                                'file_name': '{}.tfrecord'.format(file.split('.nii')[0])}

    annotation_files = [i for i in os.listdir(niftii_path) if i.find('Overall_mask') == 0]
    for file in annotation_files:
        iteration = file.split('_y')[-1].split('.')[0]
        data_dict[iteration]['annotation_path'] = os.path.join(niftii_path, file)
    return data_dict


def write_tf_record(niftii_path, out_path=None, rewrite=False, thread_count=int(cpu_count() * .5), max_records=np.inf,
                    is_3D=True, extension=np.inf, image_processors=None, special_actions=False, verbose=False,
                    file_parser=None, debug=False, recordwriter=None, **kwargs):
    """
    :param niftii_path: path to where Overall_Data and mask files are located
    :param out_path: path that we will write records to
    :param rewrite: Do you want to rewrite old records? True/False
    :param thread_count: specify 1 if debugging
    :param max_records: Can specify max number of records, for debugging purposes
    :param extension: extension above and below annotation, recommend np.inf for validation and test
    :param is_3D: Take the whole patient or break up into 2D images
    :param image_processors: a list of image processes that can take the image and annotation dictionary,
        see Image_Processors, TF_Record
    :param special_actions: if you're doing something special and don't want Add_Images_And_Annotations
    :param verbose: Binary, print processors as they go
    :return:
    """
    if out_path is None:
        out_path = niftii_path
    if image_processors is None:
        if is_3D:
            image_processors = [Add_Images_And_Annotations(),
                                Clip_Images_By_Extension(extension=extension),
                                Distribute_into_3D()]
        else:
            image_processors = [Add_Images_And_Annotations(),
                                Clip_Images_By_Extension(extension=extension),
                                Distribute_into_2D()]
    add_images_and_annotations = True
    if not special_actions:
        for processor in image_processors:
            if type(processor) is Add_Images_And_Annotations:
                add_images_and_annotations = False
                break
        if add_images_and_annotations:
            image_processors = [Add_Images_And_Annotations()] + image_processors
    if recordwriter is None:
        recordwriter = RecordWriter(out_path=out_path, file_name_key='file_name', rewrite=rewrite)
    threads = []
    q = None
    if not debug:
        q = Queue(maxsize=thread_count)
        a = [q, ]
        for worker in range(thread_count):
            t = Thread(target=worker_def, args=(a,))
            t.start()
            threads.append(t)
    if file_parser is None:
        data_dict = return_data_dict(niftii_path=niftii_path)
    else:
        data_dict = file_parser(**locals(), **kwargs)
    counter = 0
    for iteration in data_dict.keys():
        item = copy.deepcopy(data_dict[iteration])
        input_item = OrderedDict()
        input_item['input_features_dictionary'] = item
        input_item['image_processors'] = image_processors
        input_item['record_writer'] = recordwriter
        input_item['verbose'] = verbose
        if not debug:
            q.put(input_item)
        else:
            serialize_example(**input_item)
        counter += 1
        if counter >= max_records:
            break
    if not debug:
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()
    return None


if __name__ == '__main__':
    pass
