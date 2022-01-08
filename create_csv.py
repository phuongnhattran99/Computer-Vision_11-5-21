import tensorflow as tf
from my_predictors import predict_with_model
import os
import pandas as pd

path_to_image = []
image_class = []
image_prediction = []
results = []


predict_dict = {
    0:0,
    1:1,
    2:10,
    3:11,
    4:12,
    5:13,
    6:14,
    7:15,
    8:16,
    9:17,
    10:18,
    11:19,
    12:2,
    13:20,
    14:21,
    15:22,
    16:23,
    17:24,
    18:25,
    19:26,
    20:27,
    21:28,
    22:29,
    23:3,
    24:30,
    25:31,
    26:32,
    27:33,
    28:34,
    29:35,
    30:36,
    31:37,
    32:38,
    33:39,
    34:4,
    35:40,
    36:41,
    37:42,
    38:5,
    39:6,
    40:7,
    41:8,
    42:9
}

img_path = "C:/Users/PC/Desktop/miai/ComputerVision/Test"

#img_path = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Test/0/00807.png"
for i, folders in enumerate(os.listdir(img_path)):
    path_folders = os.path.join(img_path,folders)

    for j, folder in enumerate(os.listdir(path_folders)):
        path_folder = os.path.join(path_folders,folder)

        model = tf.keras.models.load_model('C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Model')
        prediction = predict_with_model(model, path_folder)
        prediction = predict_dict[prediction]
        print(f'{i}:{folders}:{folder}:{prediction}')
        path_to_image.append(path_folder)
        image_class.append(folders)
        image_prediction.append(prediction)
        if int(folders) == int(prediction):
            results.append(True)
        else:
            results.append(False) 



dict = {'path': path_to_image, 'classes': image_class, 'predictions': image_prediction, 'results': results}  
       
df = pd.DataFrame(dict) 

# saving the dataframe 
df.to_csv('C:/Users/PC/Desktop/miai/ComputerVision/GT-final_test.csv')    



    

  
