__author__ = 'Brian M Anderson'
# Created on 4/28/2020
import SimpleITK as sitk
import numpy as np
from _collections import OrderedDict


def get_start_stop(annotation, extension=np.inf):
    non_zero_values = np.where(np.max(annotation,axis=(1,2)) > 0)[0]
    start, stop = -1, -1
    if non_zero_values.any():
        start = int(non_zero_values[0])
        stop = int(non_zero_values[-1])
        start = max([start - extension, 0])
        stop = min([stop + extension, annotation.shape[0]])
    return start, stop


def get_bounding_boxes(annotation_handle,value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(annotation_handle,lowerThreshold=value,upperThreshold=value+1)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    stats.Execute(connected_image)
    bounding_boxes = [stats.GetBoundingBox(l) for l in stats.GetLabels()]
    volumes = np.asarray([stats.GetPhysicalSize(l) for l in stats.GetLabels()]).astype('float32')
    return bounding_boxes, volumes


class Image_Processor(object):
    def parse(self, input_features):
        return input_features


class Clip_Images_By_Extension(Image_Processor):
    def __init__(self, extension=np.inf):
        self.extension = extension

    def parse(self, input_features):
        image = input_features['image']
        annotation = input_features['annotation']
        start, stop = get_start_stop(annotation, self.extension)
        if start != -1 and stop != -1:
            image, annotation = image[start:stop, ...], annotation[start:stop, ...]
        input_features['image'] = image
        input_features['annotation'] = annotation
        return input_features


class Distribute_into_3D(Image_Processor):
    def __init__(self, max_z=np.inf, mirror_small_bits=True):
        self.max_z = max_z
        self.mirror_small_bits = mirror_small_bits

    def parse(self, input_features):
        out_features = {}
        start_chop = 0
        image_base = input_features['image']
        annotation_base = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        z_images_base, rows, cols = image_base.shape
        step = min([self.max_z, z_images_base])
        for index in range(z_images_base // step + 1):
            image_features = OrderedDict()
            if start_chop >= z_images_base:
                continue
            image = image_base[start_chop:start_chop + step, ...]
            annotation = annotation_base[start_chop:start_chop + step, ...]
            if image.shape[0] < step:
                if self.mirror_small_bits:
                    while image.shape[0] < step:
                        mirror_image = np.flip(image, axis=0)
                        mirror_annotation = np.flip(annotation, axis=0)
                        image = np.concatenate([image, mirror_image], axis=0)
                        annotation = np.concatenate([annotation, mirror_annotation], axis=0)
                    image = image[:step]
                    annotation = annotation[:step]
            start, stop = get_start_stop(annotation)
            image_features['image_path'] = image_path
            image_features['image'] = image
            image_features['annotation'] = annotation
            image_features['start'] = start
            image_features['stop'] = stop
            image_features['z_images'] = image.shape[0]
            image_features['rows'] = image.shape[1]
            image_features['cols'] = image.shape[2]
            image_features['spacing'] = spacing
            out_features['Image_{}'.format(index)] = image_features
        return out_features


class Distribute_into_2D(Image_Processor):
    def parse(self, input_features):
        out_features = {}
        image = input_features['image']
        annotation = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        z_images_base, rows, cols = annotation.shape
        for index in range(z_images_base):
            image_features = OrderedDict()
            image_features['image_path'] = image_path
            image_features['image'] = image[index]
            image_features['annotation'] = annotation[index]
            image_features['rows'] = rows
            image_features['cols'] = cols
            image_features['spacing'] = spacing[:-1]
            out_features['Image_{}'.format(index)] = image_features
        return out_features


class Get_Images_Annotations(Image_Processor):
    def __init__(self, extension=np.inf, wanted_values_for_bboxes=None,
                 is_3D=True, max_z=np.inf, mirror_small_bits=False):
        self.extension = extension
        self.wanted_values_for_bboxes = wanted_values_for_bboxes
        self.is_3D = is_3D
        self.max_z = max_z
        self.mirror_small_bits = mirror_small_bits

    def parse(self, input_features):
        image = input_features['image']
        annotation = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        features = OrderedDict()
        z_images_base, rows, cols = annotation.shape
        image_base, annotation_base = image, annotation
        if self.is_3D:
            start_chop = 0
            step = min([self.max_z, z_images_base])
            for index in range(z_images_base // step + 1):
                image_features = OrderedDict()
                if start_chop >= z_images_base:
                    continue
                image = image_base[start_chop:start_chop + step, ...]
                annotation = annotation_base[start_chop:start_chop + step, ...]
                if image.shape[0] < step:
                    if self.mirror_small_bits:
                        while image.shape[0] < step:
                            mirror_image = np.flip(image, axis=0)
                            mirror_annotation = np.flip(annotation, axis=0)
                            image = np.concatenate([image, mirror_image], axis=0)
                            annotation = np.concatenate([annotation, mirror_annotation], axis=0)
                        image = image[:step]
                        annotation = annotation[:step]
                    else:
                        continue
                start, stop = get_start_stop(annotation, self.extension)
                if start == -1 and stop == -1:
                    continue  # Nothing found inside anyway
                image_features['image_path'] = image_path
                image_features['image'] = image
                image_features['annotation'] = annotation
                image_features['start'] = start
                image_features['stop'] = stop
                image_features['z_images'] = image.shape[0]
                image_features['rows'] = image.shape[1]
                image_features['cols'] = image.shape[2]
                image_features['spacing'] = spacing
                start_chop += step
                if self.wanted_values_for_bboxes is not None:
                    for val in list(self.wanted_values_for_bboxes):
                        slices = np.where(annotation == val)
                        z_start, z_stop, r_start, r_stop, c_start, c_stop = 0, image.shape[0], 0, image.shape[1], 0, \
                                                                            image.shape[2]
                        volumes = np.zeros(1, dtype='float32')
                        if slices:
                            bounding_boxes, volumes = get_bounding_boxes(sitk.GetImageFromArray(annotation), val)
                            bounding_boxes = bounding_boxes[0]
                            volumes = volumes[0]
                            c_start, r_start, z_start, c_stop, r_stop, z_stop = bounding_boxes
                            z_stop, r_stop, c_stop = z_start + z_stop, r_start + r_stop, c_start + c_stop
                        image_features['bounding_boxes_z_start_{}'.format(val)] = z_start
                        image_features['bounding_boxes_r_start_{}'.format(val)] = r_start
                        image_features['bounding_boxes_c_start_{}'.format(val)] = c_start
                        image_features['bounding_boxes_z_stop_{}'.format(val)] = z_stop
                        image_features['bounding_boxes_r_stop_{}'.format(val)] = r_stop
                        image_features['bounding_boxes_c_stop_{}'.format(val)] = c_stop
                        image_features['volumes_{}'.format(val)] = volumes
                features['Image_{}'.format(index)] = image_features
        return features