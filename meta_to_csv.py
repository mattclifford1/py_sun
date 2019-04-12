import pickle
import numpy as np
from random import shuffle
import imagesize
import sys
import utils
import pandas as pd
import os


class SUNRGBD:

    def __init__(self, train_num):   #TODO work data_type
        # self.train_num_per_class = train_num_per_class
        # self.test_num_per_class = test_num_per_class
        self.train_num = train_num#  7751   # 75%
        self.test_num = int(train_num/3)#2584    # 25%
        self.label_white_list = ['table','desk','pillow','sofa','bed','box','garbage_bin','shelf','lamp','cabinet']
        self.num_classes = len(self.label_white_list)
        self.load_dataset()


    # define methods
    def load_dataset(self):
        # open the dataset file
        try:
            meta_data = utils.load_SUNRGBD_meta()
        except IOError:
            print("Need to have 'SUN-RGBD_convert_matlab.pickle' file in current directory")

        # load and assign training and test data to class
        self.csv_train_data(meta_data)
        self.csv_test_data(meta_data)


    def csv_train_data(self, meta_data):
        ## set up TRAINING data properties -------------------------------------------------
        class_count = {}
        for single_class in self.label_white_list:
            class_count[single_class] = 0
        # we don't know the size of these:
        self.filename = []     
        self.label = []
        self.label_int = []
        self.width = []
        self.height = []
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []

        im_count = 0
        for entry in range(7751):  # 75%
            if im_count >= self.train_num:
                break
            current_rgb = utils.open_rgb(meta_data, entry)
            bbs_2D = utils.get_2D_bb(meta_data, entry)
            labels = utils.get_label(meta_data, entry)
            checked_im = 0
            for objekt in range(len(bbs_2D)):
                if not labels[objekt] in self.label_white_list:  # only take the objects we care about
                    continue
                if checked_im == 0:
                    im_count += 1
                    checked_im = 1
                # if class_count[labels[objekt]] >= self.train_num_per_class:  # limit examples per class
                #     continue
                # class_count[labels[objekt]] += 1

                bb_2D = bbs_2D[objekt]
                w, h = imagesize.get(meta_data[entry][0])   # get height and width without loading image into RAM

                self.filename.append(meta_data[entry][0])
                self.label.append(labels[objekt])
                self.label_int.append(self.label_white_list.index(labels[objekt])+1)
                # self.label.append(self.label_white_list.index(labels[objekt])+1)
                self.width.append(w)
                self.height.append(h)
                self.xmin.append(int(bb_2D[0]))   # double check
                self.ymin.append(int(bb_2D[1]))   # that theses
                self.xmax.append(int(bb_2D[0])+int(bb_2D[2]))   # are correct
                self.ymax.append(int(bb_2D[1])+int(bb_2D[3]))   # !
        data = {'filename': self.filename,
                'class': self.label,
                'class_int': self.label_int,
                'width': self.width,
                'height': self.height,
                'xmin': self.xmin,
                'ymin': self.ymin,
                'xmax': self.xmax,
                'ymax': self.ymax}
        df = pd.DataFrame.from_dict(data)
        self.train_df = df


    def csv_test_data(self, meta_data):
        ## set up TESTING data properties --------------------------------------------------
        class_count = {}
        for single_class in self.label_white_list:
            class_count[single_class] = 0
        # we don't know the size of these:
        self.filename = []     
        self.label = []
        self.label_int = []
        self.width = []
        self.height = []
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        im_count = 0
        for entry in range(7751, 7751+2584):
            if im_count >= self.test_num:
                break
            current_rgb = utils.open_rgb(meta_data, entry)
            bbs_2D = utils.get_2D_bb(meta_data, entry)
            labels = utils.get_label(meta_data, entry)
            checked_im = 0
            for objekt in range(len(bbs_2D)):
                if not labels[objekt] in self.label_white_list:  # only take the objects we care about
                    continue
                if checked_im == 0:
                    im_count += 1
                    checked_im = 1
                # if class_count[labels[objekt]] >= self.test_num_per_class:  # limit examples per class
                #     continue
                # class_count[labels[objekt]] += 1

                bb_2D = bbs_2D[objekt]
                w, h = imagesize.get(meta_data[entry][0])   # get height and width without loading image into RAM

                self.filename.append(meta_data[entry][0])
                self.label.append(labels[objekt])
                self.label_int.append(self.label_white_list.index(labels[objekt])+1)
                self.width.append(w)
                self.height.append(h)
                self.xmin.append(int(bb_2D[0]))   # double check
                self.ymin.append(int(bb_2D[1]))   # that theses
                self.xmax.append(int(bb_2D[0])+int(bb_2D[2]))   # are correct
                self.ymax.append(int(bb_2D[1])+int(bb_2D[3]))  # !
        data = {'filename': self.filename,
                'class': self.label,
                'class_int': self.label_int,
                'width': self.width,
                'height': self.height,
                'xmin': self.xmin,
                'ymin': self.ymin,
                'xmax': self.xmax,
                'ymax': self.ymax}
        df = pd.DataFrame.from_dict(data)
        self.test_df = df

if __name__ == '__main__':
    if not os.path.isdir('meta_csv'):
        os.mkdir('meta_csv')
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        train_num = 7751    # full dataset
    data = SUNRGBD(train_num)
    data.train_df.to_csv('meta_csv/train'+str(train_num)+'.csv')
    data.test_df.to_csv('meta_csv/test'+str(train_num)+'.csv')
    print('Successfully created 2D meta data into csv format')

