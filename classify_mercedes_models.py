"""
model used: ResNet 18
model pretrained in Kaggle Kernel
dataset path: ./Custom Search API/car_model_images/mercedes_models_images
dataset link: https://www.kaggle.com/datasets/benlaw/mercedes-models-v2
"""

from model import classifier
import torch
import config
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import re
import random


transforms = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ]
)


class MercedesClassifier:
    def __init__(self):
        # inverse map is created in image processing
        self.inverse_map = config.inverse_target_map

    def show_image_by_index(self, index, as_cache=True):
        """
        show image of a mercedes model for its corresponding index,
        index corresponding to the inverse_map
        eg. show_image_by_index(1) -> {1: '...cla-01.png', ...} -> read '...cla-01.png'
        :param index: int, index number corresponding to inverse map
        :param as_cache: bool, if true then function will return PIL.Image object instead of showing the image
        :return:
        """
        model_images_paths = os.listdir(config.dataset_path)
        model_name = self.inverse_map[index]
        selected_model_images_paths = [path for path in model_images_paths if path.startswith(model_name)]
        rand_model_path = random.choice(selected_model_images_paths)
        # img = plt.imread(os.path.join(config.dataset_path, rand_model_path))
        img = Image.open(os.path.join(config.dataset_path, rand_model_path))
        if as_cache:
            return img
        else:
            plt.imshow(img)
            plt.show()

    @staticmethod
    def __image_preprocessing(image_path):
        """
        read image and convert to classifiable tensor size
        :param image_path: str, path of image
        :return:
        """
        # read image
        image = Image.open(image_path)
        img_shape = np.array(image).shape
        if img_shape[-1] != 3:
            raise AttributeError('Image channel != 3')

        # convert PIL image to tensor
        img_tensor = transforms(image)
        return img_tensor

    def classify(self, image_path):
        """
        classify image model name, given image_path
        :param image_path: str, path of the model image
        :return: predicted_model_name, pred_index; returns the predicted model name and its index
        """
        classifier.eval()
        with torch.no_grad():
            # input and model should have same device
            img_tensor = self.__image_preprocessing(image_path).to(config.device)
            # resize input to batch
            img_tensor = img_tensor.unsqueeze(0)

            output = classifier(img_tensor)   # output is a tensor with probs
            _, pred_index = torch.topk(output, 1)   # index is the most possible output for input, is a tensor
            pred_index = pred_index.item()
            predicted_model_name = self.inverse_map.get(pred_index)

            return predicted_model_name, pred_index


if __name__ == '__main__':
    model_classifier = MercedesClassifier()
    # can insert any mercedes model image path, but ensure that image has 3 channels
    # img_path = 'car_model_collection/test_model_images/mercedes-benz/amg-cls4.jpeg'

    # choose random image in dataset
    eval_dataset_path = 'datasets/test_images'
    img_paths = [os.path.join(eval_dataset_path, filename) for filename in os.listdir(eval_dataset_path)]
    img_path = random.choice(img_paths)
    # print(img_path)
    # model prediction results
    pred_model_name, pred_index = model_classifier.classify(img_path)

    # compare input image and predicted model image
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    real_img = plt.imread(img_path)
    real_model_name = re.sub(r'\d\..+', '', img_path.split('/')[-1])
    plt.title(real_model_name)
    plt.imshow(real_img)

    plt.subplot(1, 2, 2)
    pred_img = model_classifier.show_image_by_index(pred_index, as_cache=True)
    plt.imshow(pred_img)
    plt.title(pred_model_name)
    plt.show()

