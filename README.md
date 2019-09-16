## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
### This builds off of the function found here https://github.com/brianmanderson/Dicom_Data_to_Numpy_Arrays and into https://github.com/brianmanderson/Data_Generators

This takes the 3D images created from Dicom_to_Numpy_Arrays and creates a folder called Single_Images3D which identifies which slices have annotations and creates 2D images of the entire process

The reason for this is that you can manipulate how many images you want to load with the data generators, saving time by making it that you don't need to load the entire patient

    from Make_Single_Images import run_main
    
    run_main(path=r'\Path\To\Images\',pickle_path='\Path\To\Pickle.pkl',extension=50)
This is saying we want to grab 50 images above and below the slices which have annotations
