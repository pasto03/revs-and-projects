import torch
import pickle
import os
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = 'mercedes-model-params.model'

# change to the path where you downloaded the original dataset
dataset_path = 'datasets/train_images'

# obtain the map after data preprocessing(check kaggle kernel)
inverse_target_map = pickle.load(open('inverse_target_map.pkl', 'rb'))

# num_classes for model is the size of target
num_models = len(inverse_target_map)

if __name__ == '__main__':
    # print(torch.__version__)
    # print(device)
    # print(torch.cuda.is_available())
    print(inverse_target_map)

    # select an index of models and show image path
    model_images_paths = os.listdir(dataset_path)
    model_name = inverse_target_map[0]
    selected_model_images_paths = [path for path in model_images_paths if path.startswith(model_name)]
    print(selected_model_images_paths)
