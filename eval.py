import torch
from data.dataset import FlowerDataset
from utils.data_utils import DataUtils
from model.resnet import ResNetModel
from torch.utils.data import DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose, ShiftScaleRotate,  HueSaturationValue,
    RandomResizedCrop, RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, 
    CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from train import Config


class Test:
    def __init__(self, model, path, test_dataloader):
        self.model = model
        self.path = path
        self.test_dataloader = test_dataloader

    def evaluate(self):
        self.model.load_state_dict(torch.load(self.path)) 

        running_accuracy = 0 
        total = 0 
    
        with torch.no_grad(): 
            for data in self.test_dataloader: 
                inputs, outputs = data 
                outputs = outputs.to(torch.float32) 
                predicted_outputs = self.model(inputs) 
                _, predicted = torch.max(predicted_outputs, 1) 
                total += outputs.size(0) 
                running_accuracy += (predicted == outputs).sum().item() 
            print('Accuracy of the model based on the test is: %d %%' % (100 * running_accuracy / total))    



if __name__=='__main__':

    du = DataUtils(
        train_data_path='dataset/train',
        val_data_path='dataset/val',
        test_data_path='dataset/test'
    )

    test_augments = Compose([
        CenterCrop(Config.CFG['img_size'], Config.CFG['img_size'], p=1.),
        Resize(Config.CFG['img_size'], Config.CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)
    test_image_paths = du.get_test_images()
    test_set = FlowerDataset(test_image_paths, du.class_to_idx(), transform=test_augments)

    test = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=8
    )
    model = ResNetModel()
    test = Test(model, 'output/model.pth', test)
    test.evaluate()