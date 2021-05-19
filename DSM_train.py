"""
@author: Sophia Bano

This is the implmentation of the Deep Sequential Mosaicking [1][2] which received 
the MICCAI2019-IJCARS best paper award at MICCAI2020 award cermenoy. 

Note: If you use this code, consider citing [1][2]. 

[1] Bano, S., Vasconcelos, F., Amo, M.T., Dwyer, G., Gruijthuijsen, C., Deprest, J.,
Ourselin, S., Vander Poorten, E., Vercauteren, T., Stoyanov, D.: Deep sequential
mosaicking of fetoscopic videos. In: International Conference on Medical Image
Computing and Computer-Assisted Intervention. pp. 311–319. Springer (2019)

[2] Bano, S., Vasconcelos, F., Amo, M.T., Dwyer, G., Gruijthuijsen, C., Deprest, J.,
Ourselin, S., Vander Poorten, E., Vercauteren, T., Stoyanov, D.: Deep learning-based 
fetoscopic mosaicking for field-of-view expansion. In: International journal of 
computer assisted radiology and surgery, 15(11), 1807-1816, Springer (2020)


This script trains the Homography model using patches generated through controlled 
data augmentation. 

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot as plt
from keras import backend as K
from glob import glob
import numpy as np
import cv2
import random

# help from: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from Utils import data_augmentation_pert


def euclidean_distance(y_true, y_pred):
    # Euclidean loss function
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))

def data_generator(path, batch_size):  
    """ 
    Input:
        path -  path containing the training data (images either png or jpeg). 
        Fetoscopy images already cropped to square
        batch_size - batch size for training

    Output:
        X (input patch pairs) 
        Y (GT 4-points homography) 
        
        needed for training the homography regression model
    """
    while 1:
        # hyperparameters
        rho = 32    # 32
        patch_size = 128
        height = 256 
        width = 256 
        angle = 5
        dim = 2    # for  grayscale = 2 color = 6
        scale = 0

        # list all png and jpg images in the train folder
        loc_list = glob(path+"*.jpg")
        loc_list2 = glob(path+"*.png")
        loc_list = loc_list + loc_list2
        X = np.zeros((batch_size,patch_size, patch_size, dim))  # images
        Y = np.zeros((batch_size,8))
        for i in range(batch_size):
            # select random image from training set
            index = random.randint(0, len(loc_list)-1)
            img_file_location = loc_list[index]
            
            # Read image, resize and convert to gray
            color_image = cv2.imread(img_file_location)
            color_image = cv2.resize(color_image, (width, height))
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # create random point P within appropriate bounds
            y = random.randint(rho, height - rho - patch_size)  # row
            x = random.randint(rho, width - rho - patch_size)   # col
            
            four_points, perturbed_four_points = data_augmentation_pert(x, y, patch_size, width, height, rho, scale, angle)

            # compute H
            H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            H_inverse = np.linalg.inv(H)
            inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
            #warped_image = cv2.warpPerspective(gray_image, H, (width, height))

            # grab image patches
            original_patch = gray_image[y:y + patch_size, x:x + patch_size]
            warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
            
            # make into dataset
            training_image = np.dstack((original_patch, warped_patch))
            H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
            X[i, :, :] = training_image
            Y[i, :] = H_four_points.reshape(-1)        
        yield (X,Y)
        
# Construct Deep image homography model  
def homography_model():
    drop_prob_1 = 0.5
    input_shape=(128, 128,2)
    
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1', activation="relu"))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
    

    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3', activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv4', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
   
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv5', activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv6', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))
    
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv7', activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv8', activation='relu'))    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, name='FC1'))
    model.add(layers.Dropout(drop_prob_1))
    model.add(layers.Dense(8, name='loss'))

    plot_model(model, to_file='homography_model.png', show_shapes=True)
    
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to folder containing training images (already cropped to inscribed square of the fetoscope circle)", default="data/seq/images_crop/")

    args = parser.parse_args()
    
    EPOCHS = 10        # Set EPOCHS and BS intuitively. Typically train for 1000 to 2000 epochs
    BS = 16
    data_path = args.data_path
    
    # Load the model
    model = homography_model()
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=euclidean_distance)
   
    # Setting the checkpoint
    checkpoint = ModelCheckpoint("checkpoints/homography_model_weights.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='min') 
    callbacks_list = [checkpoint]
    
    loc_list = glob(data_path+"*.jpg")
    loc_list2 = glob(data_path+"*.png")
    num_samples = len(loc_list + loc_list2)
    
    print("training ......")
    hist = model.fit_generator(data_generator(path = data_path,batch_size=BS),epochs=EPOCHS, steps_per_epoch=round(num_samples/BS), callbacks = callbacks_list)
    
    plt.plot(hist.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


