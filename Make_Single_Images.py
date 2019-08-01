import os, pickle
import numpy as np
if os.path.exists('\\\\mymdafiles\\di_data1\\'):
    from TensorflowUtils import plot_scroll_Image, visualize, plt
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
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


def main(path,write_data=True, extension=999, q=None, re_write_pickle=True, patient_info=None):
    if not write_data:
        print('Not writing out data')
    out_path_name = 'Single_Images3D'
    files = []
    dirs = []
    for root, dirs, files in os.walk(path):
        break
    if out_path_name not in dirs:
        os.makedirs(os.path.join(path,out_path_name))
    files_in_loc = []
    for root, _, files_in_loc in os.walk(os.path.join(path,out_path_name)):
        break
    status = 0
    total = len(files)/2
    out_dict = load_obj(os.path.join(path,out_path_name,'descriptions_start_and_stop.pkl'))
    for file in files:
        if file.find('Overall_Data') == 0 or file.find('instructions') == 0 or file.find('_mask') == -1:
            continue
        pat_desc = (file.split('Overall_mask_')[1])
        if pat_desc.find('_y') != -1:
            pat_desc = pat_desc.split('_y')[0]
        else:
            pat_desc = pat_desc.split('_')[0]
        if pat_desc.find('.npy') == -1:
            desc = pat_desc + '_' + file.split('_y')[-1][:-4]
            image_path = 'Overall_Data_' + (file.split('Overall_mask_')[1]).split('_y')[0] + '_' + file.split('_y')[-1][
                                                                                                   :-4] + \
                         '.npy'
        else:
            desc = path.split('\\')[-2] + '_' + file.split('_y')[-1][:-4]
            image_path = 'Overall_Data_' + file.split('_y')[-1]


        found = False
        for i in range(999):
            out_file_name = desc + '_' + str(i) + '.npy'
            if out_file_name in files_in_loc:
                found = True
                break
        if found and not re_write_pickle: # and not write_pickle
            continue
        annotation = np.load(os.path.join(path,file))
        if path.find('LiTs') != -1:
            if np.max(annotation) == 1:
                return None
        images = np.load(os.path.join(path,image_path))
        if images.max() < 500:
            print('Image intensities are odd..')
        if path.find('Numpy_GTV_Ablation') != -1:
            annotation[annotation>0]=1
        if annotation.shape[0] != 1:
            annotation = annotation[None, ...]
            images = images[None, ...]
        annotation = annotation.astype('int')
        if path.find('Liver_Segments') != -1:
            for i in range(annotation.shape[-1]):
                annotation[..., i] *= i
            annotation = np.sum(annotation,axis=-1) # Flatten it down
        image_axis = images.shape[-1]
        annotation_axis = annotation.shape.index(image_axis)
        annotation = np.moveaxis(annotation,annotation_axis,-1)
        if len(annotation.shape) == 4:
            max_vals = np.max(annotation, axis=(0, 1, 2))
        else:
            images = np.expand_dims(images,axis=3)
            max_vals = np.max(annotation,axis=(0,1,2,3))
        non_zero_values = np.where(max_vals != 0)[0]
        if not np.any(non_zero_values):
            print('Found nothing for ' + file)
            continue
        start = non_zero_values[0]
        stop = non_zero_values[-1]
        out_dict[desc] = {'start':start,'stop':stop}
        for val in range(1,np.max(max_vals)+1):
            slices = np.where(annotation == val)[-1]
            out_dict[desc][val] = np.unique(slices)
        start_images = max([start - extension,0])
        stop_images = min([stop + extension,images.shape[-1]])
        if write_data:
            for i in range(start_images,stop_images):
                q.put([desc, path, out_path_name, files_in_loc, i, images[...,i],annotation[...,i]])
                # pool.map(write_output, ([desc, path, out_path_name, files_in_loc, i, images[:,:,:,i],annotation[:,:,:,i]] for i in range(start,stop)))
        print((status+1)/total * 100)
        status += 1
    save_obj(os.path.join(path,out_path_name,'descriptions_start_and_stop.pkl'),out_dict)

def write_output(A):
        desc, path, out_path_name, files_in_loc, i, image, annotation = A
        file_name = desc + '_' + str(i) + '.npy'
        out_file_name = os.path.join(path,out_path_name,file_name)
        if file_name not in files_in_loc:
            output = np.concatenate((image, annotation), axis=0)
            np.save(out_file_name, output)
        return None


def worker_def(q):
    objective = write_output
    while True:
        item = q.get()
        if item is None:
            break
        else:
            objective(item)
            q.task_done()

def run_main(path= r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments',
         pickle_file=os.path.join(r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments\patient_info.pkl'),
         extension=999):
    '''
    :param path: Path to parent folder that has a 'Test','Train', and 'Validation' folder
    :param pickle_file: path to 'patient_info' file
    :param extension: How many images do you want above and below your segmentations
    :return:
    '''
    patient_info = load_obj(pickle_file)
    thread_count = int(cpu_count()*.75-1)  # Leaves you one thread for doing things with
    write_images = True
    re_write_pickle = True
    # thread_count = 1
    print('This is running on ' + str(thread_count) + ' threads')
    q = Queue(maxsize=thread_count)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(q,))
        t.start()
        threads.append(t)
    for added_ext in ['']:
        for ext in ['Test','Train','Validation']:
            main(write_data=write_images,path=os.path.join(path,ext+added_ext), extension=extension, q=q, patient_info=patient_info, re_write_pickle=re_write_pickle)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
if __name__ == '__main__':
    run_main()