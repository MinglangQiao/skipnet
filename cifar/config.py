import numpy as np


Train_Batch_Size = 10
Validation_batch_size = 10
Test_Batch_Size = 10

Base_path = '/home/ml/ml_saliency/Data/Database/'

Train_maps_path = Base_path + '3-Saliency-TrainSet/All_train_images/'
Resized_train_lable_path = Base_path + '3-Saliency-TrainSet/All_train_hmap/'

Valid_image_path = Base_path + '3-Saliency-TestSet/All_test_images/'
Valid_map_path = Base_path + '3-Saliency-TestSet/All_test_hmap/'

Test_image_path = Base_path + '3-Saliency-TestSet/All_test_images/'
Test_map_path = Base_path + '3-Saliency-TestSet/All_test_hmap/'

MEAN_VALUE = np.array([123.68, 116.779, 103.939])   # RGB, refere to this: https://github.com/tensorflow/models/issues/517
# MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]

# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/4
STD_VALUE = np.array([58.395, 57.120, 50.625]) 
STD_VALUE = STD_VALUE[:,None, None]


Mode_Loss = 'KLDivLoss'
INPUT_SIZE = (1920, 1080)
N_epoch = 100

win_dic = {}
recorder = {
    'line': {},
    'heatmap': {},
    'scatter': {},
    'text': {},
    'image': {}
}

Mode_log_visdom = True