import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from deep_learning_model import streesigns_model
from my_utils import split_data, order_test_set, create_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == "__main__":

    if False:
        path_to_data = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Train"
        path_to_save_train = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Trainingset/train"
        path_to_save_val = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Trainingset/val"
        split_data(path_to_data, path_to_save_train, path_to_save_val)

    if False:
        path_to_images = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Test"
        path_to_csv = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Test.csv"
        order_test_set(path_to_images, path_to_csv)

    path_to_train = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Trainingset/train"
    path_to_val = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Trainingset/val"
    path_to_test = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Test"
    
    batch_size = 64
    epochs = 15
    train_generator, val_generator, test_generator =  create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    
    nbr_classes = train_generator.num_classes

    TRAIN = True
    TEST = False

    if TRAIN:
        path_to_save_model = "C:/Users/PC/Desktop/miai/ComputerVision/Dataset/Model"
        
        ckpt_saver = ModelCheckpoint(path_to_save_model, monitor='val_accuracy', mode='max', save_best_only=True, save_freq='epoch', verbose=1)
        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = streesigns_model(nbr_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(train_generator, epochs = epochs, batch_size= batch_size,validation_data=val_generator, callbacks=[ckpt_saver, early_stop])


    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set : ")
        model.evaluate(test_generator)