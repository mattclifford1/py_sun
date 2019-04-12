## UNDER CONTRUSTION
this repository isn't ready for use just yet, I will be writing documentation ASAP and cleaning up all code. If anyone thinks they will find this toolbox useful, send me a message any I'll get things done quicker :)


This repository holds a python tool box for the SUNRGBD dataset.

I made this toolbox box for personal use during an object dection project since I couldn't find any online, so I thought I'd share with anyone interested.
Feel free to use this code as you wish and raise any bugs via issue, or even better submit a pull request.

# Features of the Toolbox

Convert matlab metadata to a python friendly format

Easy extraction of ground truth labels

Point cloud generation from depth images using camera tilt, height and focal lengths

Visualisation of 2D and 3D bounding boxes

Class distribution visualasations 

TensorFlow tf.record seralisation for 2D RGB images and bounding boxes


more to come!!!

# Convert MatLab file to python serialisation
expects SUNRGBD dataset to be in a datasets folder up one directory ie: '../datasets/SUN-RGBD/'

$ python get_data_from_matlab.py

will serialise metadata to python pickle


then use utils.py for features or get_rgbd_points.py for batch training

# Convert python serialisation to TensorFlow serialisation
after converting from MatLab (above)

$ python meta_to_csv.py

will convert SUNRGBD 2D object data to csv format

$ python generate_tfrecord.py

will serialise SUNRGBD 2D object meta data and RGB images to tf.records




Currently only support Mac OS and Linux