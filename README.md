## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
### This builds off of the function found here https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask and into https://github.com/brianmanderson/Data_Generators

## You can now resample the images during this process!

This takes the 3D images created from Dicom_to_Numpy_Arrays and creates a folder called Single_Images3D which identifies which slices have annotations and creates 2D images of the entire process

The reason for this is that you can manipulate how many images you want to load with the data generators, saving time by making it that you don't need to load the entire patient

    from Make_Single_Images import run_main
    path = path_to_split_Images
    pickle_path = path_to_patient_info.pkl
    extension = 50
    run_main(path=path,
             pickle_path=os.path.join(path,'patient_info_Ablation_Zones.pkl'),
             resample=True, desired_output_spacing=(None,None,2.5),extension=extension)
This is saying we want to grab 50 images above and below the slices which have annotations

Also now supporting TFDataset by making records

image_processors do not need to be given, defaults are provided for 2D and 3D creation

    from Make_Single_Images.Make_TFRecord_Class import write_tf_record, os
    from Make_Single_Images.Image_Processors.Image_Processors_TFRecord import *
    path = r'D:\Liver_Work'
    image_processors = [Normalize_to_annotation(1), Clip_Images_By_Extension(16), Distribute_into_3D()]
    write_tf_record(os.path.join(path, 'Train'), is_3D=False, rewrite=True, record_name='bbox_test', image_processors=image_processors)
