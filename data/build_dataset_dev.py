from imutils import paths
import numpy as np
import shutil
import os

class BuildDataset:
    def __init__(self, train_path, val_path, dataset_path):
        self.train_path = train_path
        self.val_path = val_path
        self.dataset_path = dataset_path

    def copy_images(self, imagePaths, folder):
        # check if the destination folder exists and if not create it
        if not os.path.exists(folder):
            os.makedirs(folder)
        # loop over the image paths
        for path in imagePaths:
            # grab image name and its label from the path and create
            # a placeholder corresponding to the separate label folder
            imageName = path.split(os.path.sep)[-1]
            label = path.split(os.path.sep)[-2]
            labelFolder = os.path.join(folder, label)
            # check to see if the label folder exists and if not create it
            if not os.path.exists(labelFolder):
                os.makedirs(labelFolder)
            # construct the destination image path and copy the current
            # image to it
            destination = os.path.join(labelFolder, imageName)
            shutil.copy(path, destination)
    
    def build_dataset(self, val_split):
        # load all the image paths and randomly shuffle them
        print("[INFO] loading image paths...")
        imagePaths = list(paths.list_images(self.dataset_path))
        np.random.shuffle(imagePaths)
        # generate training and validation paths
        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        trainPaths = imagePaths[:trainPathsLen]
        valPaths = imagePaths[trainPathsLen:]
        # copy the training and validation images to their respective
        # directories
        print("[INFO] copying training and validation images...")
        self.copy_images(trainPaths, self.train_path)
        self.copy_images(valPaths, self.val_path)


if __name__ == '__main__':
    TRAIN_PATH = "dataset/train"
    VAL_PATH = "dataset/val"
    DATASET_PATH = "flower_photos"
    VAL_SPLIT = 0.1

    BD = BuildDataset(TRAIN_PATH, VAL_PATH, DATASET_PATH)
    BD.build_dataset(VAL_SPLIT)