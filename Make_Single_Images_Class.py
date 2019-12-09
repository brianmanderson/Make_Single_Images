import os, pickle, copy
import numpy as np
import SimpleITK as sitk
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import nibabel as nib
import matplotlib.pyplot as plt
from keras.utils import np_utils
from Resample_Class.Resampling_Utils import Resample_Class_Object

def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where(self.X != 0)[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, 3)
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


def run(path,write_data=True, extension=999, q=None, re_write_pickle=True, patient_info=dict(), resampler=None, desired_output_spacing=(None,None,2.5)):
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
    files = [i for i in files if i.find('Overall_Data') == -1 and i.find('instructions') == -1 and i.find('_mask') != -1]
    for file in files:
        ext = '.npy'
        if file.find(ext) == -1:
            ext = '.nii.gz'
        print(file)
        pat_desc = (file.split('Overall_mask_')[1])
        if pat_desc.find('_y') != -1:
            pat_desc = pat_desc.split('_y')[0]
        else:
            pat_desc = pat_desc.split('_')[0]
        if pat_desc.find(ext) == -1:
            desc = pat_desc + '_' + file.split('_y')[-1].split('.')[0]
            image_path = 'Overall_Data_' + (file.split('Overall_mask_')[1]).split('_y')[0] + '_' + file.split('_y')[-1].split('.')[0] + ext
        else:
            desc = path.split('\\')[-2] + '_' + file.split('_y')[-1].split('.')[0]
            image_path = 'Overall_Data_' + file.split('_y')[-1]


        found = False
        for i in range(999):
            out_file_name = desc + '_' + str(i) + '_image' + ext
            if out_file_name in files_in_loc:
                found = True
                break
        if found and not re_write_pickle and desc in out_dict: # if the desc isn't in the out dict, re-run it
            continue
        annotation_handle = sitk.ReadImage(os.path.join(path,file))
        # annotation = sitk.GetArrayFromImage(annotation_handle)
        image_handle = sitk.ReadImage(os.path.join(path, image_path))
        # images = sitk.GetArrayFromImage(image_handle)
        # if path.find('Numpy_GTV_Ablation') != -1:
        #     annotation[annotation>0]=1
        if resampler is not None:
            input_spacing = image_handle.GetSpacing()
            output_spacing = []
            for index in range(3):
                if desired_output_spacing[index] is None:
                    output_spacing.append(input_spacing[index])
                else:
                    output_spacing.append(desired_output_spacing[index])
            output_spacing = tuple(output_spacing)
            if output_spacing != input_spacing:
                print('Resampling to ' + str(output_spacing))
                image_handle = resampler.resample_image(input_image=image_handle,input_spacing=input_spacing,
                                                          output_spacing=output_spacing,is_annotation=False)
                annotation_handle = resampler.resample_image(input_image=annotation_handle,input_spacing=input_spacing,
                                                          output_spacing=output_spacing,is_annotation=True)
        # Annotations should be up the shape [1, 512, 512, # classes, # images]
        annotation = sitk.GetArrayFromImage(annotation_handle)
        non_zero_values = np.where(annotation>0)[0]
        if not np.any(non_zero_values):
            print('Found nothing for ' + file)
            continue
        start = non_zero_values[0]
        stop = non_zero_values[-1]
        out_dict[desc] = {'start':start,'stop':stop}
        for val in range(1,np.max(annotation)+1):
            slices = np.where(annotation == val)
            if slices:
                slices = slices[0]
                out_dict[desc][val] = np.unique(slices)
        start_images = max([start - extension,0])
        stop_images = min([stop + extension,annotation.shape[0]])
        if write_data:
            for i in range(start_images,stop_images):
                q.put([desc, path, out_path_name, files_in_loc, i, image_handle[:,:,i],annotation_handle[:,:,i]])
                # pool.map(write_output, ([desc, path, out_path_name, files_in_loc, i, images[:,:,:,i],annotation[:,:,:,i]] for i in range(start,stop)))
        print((status+1)/total * 100)
        status += 1
        save_obj(os.path.join(path,out_path_name,'descriptions_start_and_stop.pkl'),out_dict)


def write_output(A):
    desc, path, out_path_name, files_in_loc, i, image, annotation = A
    file_name_image = desc + '_' + str(i) + '_image.nii.gz'
    file_name_annotation = file_name_image.replace('_image.nii.gz', '_annotation.nii.gz')
    file_name_image = os.path.join(path, out_path_name, file_name_image)
    file_name_annotation = os.path.join(path, out_path_name, file_name_annotation)
    if file_name_image not in files_in_loc or file_name_annotation not in files_in_loc:
        sitk.WriteImage(image,file_name_image)
        sitk.WriteImage(annotation, file_name_annotation)
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

def main(path= r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments',desired_output_spacing=(None,None,2.5),
         extension=999,write_images=True,re_write_pickle=False, pickle_path=None, resample=False,
         thread_count = int(cpu_count()*.75-1)):
    '''
    :param path: Path to parent folder that has a 'Test','Train', and 'Validation' folder
    :param pickle_file: path to 'patient_info' file
    :param extension: How many images do you want above and below your segmentations
    :param write_images: Write out the images?
    :param re_write_pickle: re-write the pickle file? If true, will require loading images again
    :param desired_output_spacing: desired spacing of output images in mm (dy, dx, dz), None will not change
    :return:
    '''
    patient_info = dict()
    if pickle_path:
        patient_info = load_obj(pickle_path)
    resampler = None
    if resample:
        resampler = Resample_Class_Object()
    print('This is running on ' + str(thread_count) + ' threads')
    q = Queue(maxsize=thread_count)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(q,))
        t.start()
        threads.append(t)
    for added_ext in ['']:
        for ext in ['Train','Test', 'Validation']:
            run(write_data=write_images,path=os.path.join(path,ext+added_ext), extension=extension, q=q, re_write_pickle=re_write_pickle, patient_info=patient_info, resampler=resampler,
                 desired_output_spacing=desired_output_spacing)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
if __name__ == '__main__':
    # path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Pancreas\Pancreas\All_Imaging\UMPC_Patients\Nib_UMPC_Patients'
    # run_main(path=os.path.join(path,'CT'),
    #          pickle_path=os.path.join(path,'patient_info_Ablation_Zones.pkl'),
    #          resample=False, desired_output_spacing=(None,None,2.5))
    xxx = 1