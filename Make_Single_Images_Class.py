import os, pickle
import numpy as np
import SimpleITK as sitk
if os.path.exists(r'K:\Morfeus'):
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

class Resample_Class(object):
    '''
    Feed in images in form of #images, rows, cols
    '''
    def __init__(self):
        self.Resample = sitk.ResampleImageFilter()
    def resample_image(self,input_image, input_spacing=(0.975,0.975,2.5),output_spacing=(0.975,0.975,2.5),
                       is_annotation=False):
        '''
        :param input_image: Image of the shape # images, rows, cols
        :param spacing: Goes in the form of (row_dim, col_dim, z_dim) (I know it's confusing..)
        :param is_annotation: Whether to use Linear or NearestNeighbor, Nearest should be used for annotations
        :return:
        '''
        output_spacing = np.asarray(output_spacing)
        self.Resample.SetOutputSpacing(output_spacing)
        if is_annotation:
            self.Resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            self.Resample.SetInterpolator(sitk.sitkLinear)

        image = sitk.GetImageFromArray(input_image)
        image.SetSpacing(input_spacing)
        orig_size = np.array(image.GetSize(),dtype=np.int)
        orig_spacing = np.asarray(image.GetSpacing())
        new_size = orig_size * (orig_spacing / output_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [np.int(i) for i in new_size]
        self.Resample.SetSize(new_size)
        self.Resample.SetOutputDirection(image.GetDirection())
        self.Resample.SetOutputOrigin(image.GetOrigin())
        output = self.Resample.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

def main(path,write_data=True, extension=999, q=None, re_write_pickle=True, patient_info=dict(), resampler=None, desired_output_spacing=(None,None,2.5)):
    # Annotations should be up the shape [1, 512, 512, # classes, # images]
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
            out_file_name = desc + '_' + str(i) + '_image.npy'
            if out_file_name in files_in_loc:
                found = True
                break
        if found and not re_write_pickle and desc in out_dict: # if the desc isn't in the out dict, re-run it
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
        if annotation.shape[-2] == annotation.shape[-3]:
            annotation = np.expand_dims(annotation,axis=-2)

        if resampler is not None:
            descriptions = desc.split('_')
            info = None
            for i in range(len(descriptions)):
                if descriptions[i] in patient_info:
                    info = patient_info[descriptions[i]][descriptions[i + 1]].split(',')
                    break
            if info:
                pat_id, slice_thickness, x_y_resolution = info
                temp_images = np.transpose(images[0,...],axes=(-1,0,1))
                temp_annotations = np.transpose(annotation[0,...],axes=(-1,0,1,2))
                input_spacing = (float(x_y_resolution), float(x_y_resolution), float(slice_thickness))
                output_spacing = []
                if desired_output_spacing[0] is None:
                    output_spacing.append(float(x_y_resolution))
                    output_spacing.append(float(x_y_resolution))
                else:
                    output_spacing.append(desired_output_spacing[0])
                    output_spacing.append(desired_output_spacing[1])
                output_spacing.append(desired_output_spacing[-1])
                output_spacing = tuple(output_spacing)
                if output_spacing[-1] >= input_spacing[-1]:
                    print('Resampling to ' + str(output_spacing))
                    resized_images = resampler.resample_image(input_image=temp_images,input_spacing=input_spacing,
                                                              output_spacing=output_spacing,is_annotation=False)
                    resized_annotations = np.zeros(resized_images.shape + (annotation.shape[3],))
                    for i in range(temp_annotations.shape[-1]):
                        resized_annotations[...,i] = resampler.resample_image(input_image=temp_annotations[...,i], input_spacing=input_spacing,
                                                                              output_spacing=output_spacing)
                    images = np.transpose(resized_images,axes=(1,2,0))[None,...]
                    annotation = np.transpose(resized_annotations,axes=(1,2,3,0))[None,...]
                # else:
                #     print('Only downsampling, not upsampling')

        # Annotations should be up the shape [1, 512, 512, # classes, # images]
        max_vals = np.max(annotation,axis=(0,1,2,3))
        non_zero_values = np.where(max_vals != 0)[0]
        if not np.any(non_zero_values):
            print('Found nothing for ' + file)
            continue
        start = non_zero_values[0]
        stop = non_zero_values[-1]
        out_dict[desc] = {'start':start,'stop':stop}
        for val in range(annotation.shape[-2]):
            slices = np.where(annotation[0,:,:,val,:] == 1)[-1]
            out_dict[desc][val+1] = np.unique(slices)
        start_images = max([start - extension,0])
        stop_images = min([stop + extension,images.shape[-1]])
        if write_data:
            for i in range(start_images,stop_images):
                q.put([desc, path, out_path_name, files_in_loc, i, images[...,i],annotation[...,i], np.max(max_vals)])
                # pool.map(write_output, ([desc, path, out_path_name, files_in_loc, i, images[:,:,:,i],annotation[:,:,:,i]] for i in range(start,stop)))
        print((status+1)/total * 100)
        status += 1
    save_obj(os.path.join(path,out_path_name,'descriptions_start_and_stop.pkl'),out_dict)


def write_output(A):
    desc, path, out_path_name, files_in_loc, i, image, annotation, max_val = A
    file_name_image = desc + '_' + str(i) + '_image.npy'
    file_name_annotation = file_name_image.replace('_image.npy', '_annotation.npy')
    file_name_image = os.path.join(path, out_path_name, file_name_image)
    file_name_annotation = os.path.join(path, out_path_name, file_name_annotation)
    if file_name_image not in files_in_loc or file_name_annotation not in files_in_loc:
        np.save(file_name_image, image.astype('float32'))
        if max_val == 1:
            dtype = 'bool'
        else:
            dtype = 'int8'
        np.save(file_name_annotation, annotation.astype(dtype))
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

def run_main(path= r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments',desired_output_spacing=(None,None,2.5),
         extension=999,write_images=True,re_write_pickle=False, pickle_path=None, resample=False, resampler=None):
    '''
    :param path: Path to parent folder that has a 'Test','Train', and 'Validation' folder
    :param pickle_file: path to 'patient_info' file
    :param extension: How many images do you want above and below your segmentations
    :param write_images: Write out the images?
    :param re_write_pickle: re-write the pickle file? If true, will require loading images again
    :param desired_output_spacing: desired spacing of output images in mm (dx, dy, dz), None will not change
    :return:
    '''
    patient_info = dict()
    if pickle_path:
        patient_info = load_obj(pickle_path)
    thread_count = int(cpu_count()*.75-1)  # Leaves you one thread for doing things with
    # thread_count = 1
    if resample:
        resampler = Resample_Class()
    print('This is running on ' + str(thread_count) + ' threads')
    q = Queue(maxsize=thread_count)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(q,))
        t.start()
        threads.append(t)
    for added_ext in ['']:
        for ext in ['Test','Train','Validation']:
            main(write_data=write_images,path=os.path.join(path,ext+added_ext), extension=extension, q=q, re_write_pickle=re_write_pickle, patient_info=patient_info, resampler=resampler,
                 desired_output_spacing=desired_output_spacing)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
if __name__ == '__main__':
    # path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Ablation_Zones\Numpy_Ablation_Zones'
    # run_main(path=os.path.join(path,'CT'),
    #          pickle_path=os.path.join(path,'patient_info_Ablation_Zones.pkl'),
    #          resample=True, desired_output_spacing=(None,None,2.5))
    xxx = 1