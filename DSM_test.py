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


This script creates pair of patches from a single image for testing where the 
ground-truth is already available due to data augmentation. It performs
- Testing on pair of images
- Visualising the results
- Evaluation using mean corner error and root mean square error

"""

from keras.models import load_model
from keras import backend as K
import cv2
from glob import glob
import numpy as np
import random
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error


from Utils import data_augmentation_pert

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))

# test data generation 
def get_test(path, iformat):
    rho = 16  
    patch_size = 128
    height = 256
    width =  256
    angle = 10
    scale = 0
       
    dim = 2    # for  grayscale = 2 
        
    #random read image
    loc_list = glob(path + "*." + iformat)
    index = random.randint(0, len(loc_list)-1)
    img_file_location = loc_list[index]
    color_image = cv2.imread(img_file_location)
    color_image = cv2.resize(color_image, (width, height))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    #gray_image = color_image
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    #points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho,  width - rho - patch_size)  # col
    
    # Controlled Data Augmentation
    four_points, perturbed_four_points = data_augmentation_pert(x, y, patch_size, width, height, rho, scale, angle)
    four_points_array = np.array(four_points)
    
    #compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = np.linalg.inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
    
    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    val_image = training_image.reshape((1,patch_size,patch_size,dim))
    
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    four_points_delta_array = H_four_points.reshape(1,8)        
    
    #compute H  - for color
    inv_warped_image = cv2.warpPerspective(color_image, H_inverse, (width, height))
    
    # grab image patches
    P_Ac = color_image[y:y + patch_size, x:x + patch_size]
    P_Bc = inv_warped_image[y:y + patch_size, x:x + patch_size]
    
    return color_image, P_Ac, P_Bc, H_inverse,val_image,four_points_delta_array, four_points_array


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to fetoscopy sequences images cropped to inscribed square of the fetoscope", default="../../datasets/Data_homo/Invivo/anon001/seq2/images_crop/")
    parser.add_argument("--iformat", help="Input image file format png or jpg", default="png")
    parser.add_argument("--Print_results", help = "plot orginal patch, perturbed patch and wraped image ", default=False)
    
    args = parser.parse_args()

    # load model
    chkpt = "checkpoints/DSM_weights.hdf5"
    model = load_model(chkpt, custom_objects={'euclidean_distance': euclidean_distance})
    
    # Input params
    data_path = args.data_path
    iformat = args.iformat
    Print_results = args.Print_results 
    
    num_samples  = len(glob(data_path+"*.png"))
    
    error = np.empty(num_samples)
    rmse = np.empty(num_samples)
    for j in range(num_samples):
            
        color_image, P_Ac, P_Bc, H_inverse,val_image,four_points_d, four_points1 = get_test(data_path, iformat)
    
        siz_im = np.shape(color_image)
        
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        #predict
        predicted_four_points = model.predict(val_image)
        
        if Print_results:
            # original patch
            color_image_c =color_image.copy()
            cv2.polylines(color_image_c, np.int32([four_points1]), 1, (0,0,255),thickness=3)
            plt.imshow(color_image_c)
            plt.show()
            
            # perturbed patch
            color_image_cB = color_image.copy()
            four_points2 = four_points1.reshape(1,8) + four_points_d
            four_points2 = four_points2.reshape(4,2) 
            cv2.polylines(color_image_cB, np.int32([four_points2]), 1, (0,255,0),thickness=3)
            plt.imshow(color_image_cB)
            plt.show()  
        
            # warped image
            inv_warped_image = cv2.warpPerspective(color_image.copy(), H_inverse, (siz_im[0], siz_im[1]))  #(269, 269)
            cv2.polylines(inv_warped_image, np.int32([four_points1]), 1, (0,255,0),thickness=3)
            plt.imshow(inv_warped_image)
            plt.show()
    
        patchA = np.squeeze(val_image[0,:,:,0])
        patchB = np.squeeze(val_image[0,:,:,1])
            
        x_1 = np.sqrt((four_points_d[:, 0] - predicted_four_points[:, 0]) ** 2 + (four_points_d[:, 1] - predicted_four_points[:, 1]) ** 2)
        x_2 = np.sqrt((four_points_d[:, 2] - predicted_four_points[:, 2]) ** 2 + (four_points_d[:, 3] - predicted_four_points[:, 3]) ** 2)
        x_3 = np.sqrt((four_points_d[:, 4] - predicted_four_points[:, 4]) ** 2 + (four_points_d[:, 5] - predicted_four_points[:, 5]) ** 2)
        x_4 = np.sqrt((four_points_d[:, 6] - predicted_four_points[:, 6]) ** 2 + (four_points_d[:, 6] - predicted_four_points[:, 7]) ** 2)
        error[j] = np.average(x_1 + x_2 + x_3 + x_4)/4
        print('Mean Corner Error: ', error[j])
        rmse[j] = math.sqrt(mean_squared_error(four_points_d, predicted_four_points))
        print('Root Mean Square Error: ', rmse[j])
    print('Mean Average Corner Error: ', np.average(error))
    print('Mean Standard Deviation Corner Error: ', np.std(error))
    print('Mean Root Mean Square Error: ', np.average(rmse))
    print('Mean Standard Root Mean Square Error: ', np.std(rmse))
            