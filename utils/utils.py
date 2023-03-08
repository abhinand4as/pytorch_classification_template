import glob
import random
from pandas.core.common import flatten
import pandas as pd

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
        print("length of class: ", len(self.classes))
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
    

    def concat_train_val_csv(self, train_image_paths, val_image_paths):
        label = []
        for train_image_path in train_image_paths:
            _label = train_image_path.split('/')[-2]
            print(_label)
            label.append(self.class_to_idx(_label))
            # print(self.class_to_idx([_label]))
            
        for val_image_path in val_image_paths:
            _label = val_image_path.split('/')[-2]
            label.append(self.class_to_idx(_label))
        path = train_image_paths + val_image_paths
        # dictionary of lists  
        dict = {'cls': label, 'path': path}  
            
        df = pd.DataFrame(dict) 
            
        # saving the dataframe 
        df.to_csv('GFG.csv') 


if __name__ == '__main__':
    utils = Utils(
        'dataset/train',
        'dataset/val'
    )

    train_image_paths, val_image_paths = utils.get_image_path()
    utils.concat_train_val_csv(train_image_paths, val_image_paths)