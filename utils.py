import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import skimage
if os.getcwd()[:5] != '/mnt/':     # blue crystal can't pip install these for some reason
    import open3d
    import pptk

# bunch of functions to help extract/ manipulate the dataset in a readable format.


def open_depth(meta_data, data_num):
    return cv2.imread(meta_data[data_num][1], -1)  # use -1 to read as 16 bit not 8 bit (need 16 bit for relative depth)


def normalise_depth(depth):
    # use skimage as_uint8
    # return skimage.img_as_float(depth)
    return depth


def open_rgb(meta_data, data_num):
    image = plt.imread(meta_data[data_num][0])
    return image
    # return skimage.img_as_float(image)


def get_2D_bb(meta_data, data_num):
    return meta_data[data_num][5]


def get_3D_bb(meta_data, data_num):
    return (meta_data[data_num][6], meta_data[data_num][7], meta_data[data_num][8])


def get_camera_pos(meta_data, data_num):
    # extract 'K' from meta_data 
    # fx, fy are the focal focal length of x and y
    # cx, cy are the optical centers    of x and y
    [[fx, _, cx], [_, fy, cy], [_, _, _]] = meta_data[data_num][3] # K is [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Rtilt = meta_data[data_num][2]
    return (fx, fy, cx, cy, Rtilt)


def get_label(meta_data, data_num):
    return meta_data[data_num][4]


def load_SUNRGBD_meta(path_to_dataset = '../process_metadata/SUN-RGBD_convert_matlab.pickle'):
    with open(path_to_dataset, 'rb') as pickle_in:
        return pickle.load(pickle_in)
    # meta_data in format:
    # [rgb_file_path, depth_file_path, Rtilt, K, label, BB_2D, basis_3D, coeff_3D, centroid_3D]


def deserialise_pickle(pickle_name):
    with open(pickle_name, 'rb') as pickle_in:
        return pickle.load(pickle_in)


def serialise_pickle(objekt, pickle_name):
    with open(pickle_name, 'wb') as pickle_out:
        pickle.dump(objekt, pickle_out)


def make_point_cloud(depth, meta_data, data_num):   # meta_data needed as holds camera position and tilt
    # convert int16 values to depth (inversely proportional)
    depth_input = ((depth>>3)|(depth<<13))/1000  # taken from sunrgbd matlab toolbox --- fuck knows how this works
    # threshold max depth
    max_depth = 8    # max depth measurement at 8 meters (limit of camera)
    depth_input[np.where(depth_input > 8)] = 8   # max depth
    # get camera parameters to scale and tilt camera
    (fx, fy, cx, cy, Rtilt) = get_camera_pos(meta_data, data_num)

    x, y = np.meshgrid(range(depth_input.shape[1]), range(depth_input.shape[0]))
    # translate from optical center and scale inversely proportional to focal lengths
    x3 = (x-cx) * (depth_input*(1/fx))
    y3 = (y-cy) * (depth_input*(1/fy))
    z3 = depth_input
    # convert to coordinate style list
    points = [x3.reshape(-1), z3.reshape(-1), -y3.reshape(-1)]
    # tilt points so that the floor is flat using camera tilt
    return np.transpose(np.matmul(Rtilt, points))/max_depth   # normalise to the max depth



def down_sample_point_cloud(point_cloud, voxel_size):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_cloud)
    downpcd = open3d.voxel_down_sample(pcd, voxel_size = voxel_size)
    return np.asarray(downpcd.points)


def rand_down_sample(point_cloud, num_of_points):
    rand_to_take = np.random.randint(len(point_cloud), size=(num_of_points))
    return point_cloud[rand_to_take, :]


def plot_point_cloud(point_cloud):  # todo: make better than matplotlib: use pptk
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s=0.1)
    # plt.axis('equal')
    # plt.show()
    v = pptk.viewer(point_cloud)


def plot_image(image):
    if np.max(image) > 1:
        image = image.astype(int)
    plt.imshow(image)
    plt.show()


def save_image(image, file_name):
    plt.imsave(file_name, image)


def plot_images(image_list):   # input list of images TODO: take more than2 images
    _, axarr = plt.subplots(1,2)
    for i in range(len(image_list)):
        if np.max(image_list[i]) > 1:
            image_list[i] = image_list[i].astype(int)
    axarr[0].imshow(image_list[0])
    axarr[1].imshow(image_list[1])
    plt.show()


def load_model(sess, model_dir, train_file):
    if model_dir[len(model_dir) -1] != '/':
        model_dir += '/'
    if train_file[-5:] == '.meta':
        saver = tf.train.import_meta_graph(model_dir + train_file)
        train_file =train_file[:-4]  # get rid of meta for loading wieghts
    else:
        saver = tf.train.import_meta_graph(model_dir + train_file + '.meta')
    current_dir = os.getcwd()
    print(current_dir)
    os.chdir(model_dir)       # bit of a hacky way of doing things... try find better solution
    saver.restore(sess, train_file)
    # saver.restore(sess, tf.train.latest_checkpoint('./'))
    os.chdir(current_dir)


def save_graph(sess, log_dir='logs/'):
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)


def add_text_image(image, text):
    return cv2.putText(image, text,(0,image.shape[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)



