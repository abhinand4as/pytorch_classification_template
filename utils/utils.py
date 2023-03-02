import glob
import random
from pandas.core.common import flatten

class Utils:
    def __init__(self, train_data_path, val_data_path):
        self.train_data_path = train_data_path 
        self.val_data_path = val_data_path 

        self.train_image_paths = []
        self.val_image_paths = []
        self.classes = []

    def get_image_path(self):
        for data_path in glob.glob(self.train_data_path + '/*'):
            self.classes.append(data_path.split('/')[-1])
            self.train_image_paths.append(glob.glob(data_path + '/*'))

        self.train_image_paths = list(flatten(self.train_image_paths))
        random.shuffle(self.train_image_paths)

        print('train_image_path example: ', self.train_image_paths[0])
        print('class example: ', self.classes[0])

        # val
        for data_path in glob.glob(self.val_data_path + '/*'):
            self.val_image_paths.append(glob.glob(data_path + '/*'))
        self.val_image_paths = list(flatten(self.val_image_paths))
        
        return self.train_image_paths, self.val_image_paths
    def idx_to_class(self):
        return {i:j for i, j in enumerate(self.classes)}
    
    def class_to_idx(self):
        return {value:key for key,value in self.idx_to_class().items()}