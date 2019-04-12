## using SUNRGBD's matlab toolbox to extract data with corresponding metadata
## end result with python list in format:
##              rows: each entry
##              columns:    0: filepath to rgb image .jpg
##                          1: filepath to depth image .png
##                          2: object labels 
##                          3: 2D bounding boxes
##                          4: 3D basis             
##                          5: 3D coefficients
##                          6: 3D centroids

import pickle
import os
import scipy.io as sio

# check if running on blue crystal or not (need to get dataset location to scratch space)
if os.getcwd() == '/mnt/storage/home/mc15445/Technical_project/process_metadata':
    path_to_dataset = '/mnt/storage/scratch/mc15445/datasets/SUN-RGBD/'
else:
    path_to_dataset = '../datasets/SUN-RGBD/'  

# if not os.path.isfile('SUN-RGBD_convert_matlab.pickle'):
matlab_meta = sio.loadmat(path_to_dataset + 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')

dataset = []  # don't know size because might be some missing data
no_data = 0   # count how many are missing annotations
# matlab_meta is a dict -- we want to get metadata from 'SUNRGBDMeta' key
for i in range(len(matlab_meta['SUNRGBDMeta'][0])):
    # get image file paths
    rgb_file_path = matlab_meta['SUNRGBDMeta'][0][i][5][0]
    depth_file_path = matlab_meta['SUNRGBDMeta'][0][i][4][0]
    # slice the path we need
    rgb_file_path = path_to_dataset + rgb_file_path[17:len(rgb_file_path)]
    depth_file_path = path_to_dataset + depth_file_path[17:len(depth_file_path)]
    # camera focal lengths, centers and tilts
    Rtilt = matlab_meta['SUNRGBDMeta'][0][i][2]    # camera tilt
    K = matlab_meta['SUNRGBDMeta'][0][i][3]        # focal length and center of camera
    # set up lists for bounding boxes 
    label = []
    BB_2D = []
    basis_3D = []
    coeff_3D = []
    centroid_3D = []
    # save bounding boxes
    try:
        for j in range(len(matlab_meta['SUNRGBDMeta'][0][i][1][0])):
            label.append(matlab_meta['SUNRGBDMeta'][0][i][1][0][j][3][0])
            BB_2D.append(matlab_meta['SUNRGBDMeta'][0][i][1][0][j][7][0])

            basis_3D.append(matlab_meta['SUNRGBDMeta'][0][i][1][0][j][0])
            coeff_3D.append(matlab_meta['SUNRGBDMeta'][0][i][1][0][j][1][0])
            centroid_3D.append(matlab_meta['SUNRGBDMeta'][0][i][1][0][j][2][0])
    except:
        no_data += 1

    # save to meta dataset 
    dataset.append([rgb_file_path, depth_file_path, Rtilt, K, label, BB_2D, basis_3D, coeff_3D, centroid_3D])
print('Number of objects with missing annotation is {}'.format(no_data))

# save list so we don't have to re-run everytime
with open('SUN-RGBD_convert_matlab.pickle','wb') as pickle_out:
    pickle.dump(dataset, pickle_out)
# else:      # load pickled file if it's already been made
#     with open('SUN-RGBD_convert_matlab.pickle','rb') as pickle_in:
#         dataset = pickle.load(pickle_in)
