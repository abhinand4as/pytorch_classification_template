import glob
import random
from pandas.core.common import flatten
import pandas as pd


class DataUtils:
    def __init__(self, train_data_path, val_data_path, test_data_path=None):
        self.train_data_path = train_data_path 
        self.val_data_path = val_data_path 
        self.test_data_path = test_data_path
        self.train_image_paths = []
        self.val_image_paths = []
        self.test_image_paths = []
        self.classes = []
        for data_path in glob.glob(self.train_data_path + '/*'):
            self.classes.append(data_path.split('/')[-1])

    def get_train_images(self):
            for data_path in glob.glob(self.train_data_path + '/*'):
                self.train_image_paths.append(glob.glob(data_path + '/*'))
            self.train_image_paths = list(flatten(self.train_image_paths))
            random.shuffle(self.train_image_paths)

            print('train_image_path example: ', self.train_image_paths[0])
            print('class example: ', self.classes[0])

            return self.train_image_paths
    def get_val_images(self):
            # val
            for data_path in glob.glob(self.val_data_path + '/*'):
                self.val_image_paths.append(glob.glob(data_path + '/*'))
            self.val_image_paths = list(flatten(self.val_image_paths))
            
            return self.val_image_paths
    def get_test_images(self):
            for data_path in glob.glob(self.test_data_path + '/*'):
                self.test_image_paths.append(glob.glob(data_path + '/*'))
            self.test_image_paths = list(flatten(self.test_image_paths))
            return self.test_image_paths
    
    def idx_to_class(self):
        return {i:j for i, j in enumerate(self.classes)}
    
    def class_to_idx(self):
        return {value:key for key,value in self.idx_to_class().items()}
    