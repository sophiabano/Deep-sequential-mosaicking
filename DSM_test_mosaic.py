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

This script performs the testing of the sequence data (frames extracted from a video clip) and 
implements the outlier rejection stage for pruning the most consistent homography estimate

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
import os
from PIL import Image



def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))

# get paired data for testing from two consecutive frames
def get_pair_data(im1, im2):
    rho = 16  
    patch_size = 128
    height = 256 
    width = 256 
    
    dim = 2
    
    color_image1 = cv2.imread(im1)
    color_image1 = cv2.resize(color_image1, (width, height))
    gray_image1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)
    
    color_image2 = cv2.imread(im2)
    color_image2 = cv2.resize(color_image2, (width, height))
    gray_image2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)
    
    # points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho, width - rho - patch_size) # col
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = np.array([top_left_point, bottom_left_point, bottom_right_point, top_right_point])
           
    # grab image patches
    patch1 = gray_image1[y:y + patch_size, x:x + patch_size]
    patch2 = gray_image2[y:y + patch_size, x:x + patch_size]
        
    # make into dataset
    training_image = np.dstack((patch1, patch2))
    val_image = training_image.reshape((1,patch_size,patch_size,dim))     

    return color_image1, color_image2, patch1, patch2,val_image, four_points

"""
Singular value decomposition for pruning homoraphy estimates
The function decomposes the homograpy matrix using Singular Value Decomposition (SVD).
"""
def SVD(H_prune):
    svd = np.zeros([6,len(H_prune)])
    sel_svd = np.zeros([6,1])
    for j in range(len(H_prune)):
        H1 = H_prune[j]
    
        E = (H1[0,0] + H1[1,1])/2
        F = (H1[0,0] - H1[1,1])/2
        G = (H1[1,0] + H1[0,1])/2
        H_val = (H1[1,0] - H1[0,1])/2
        
        Q = np.sqrt(np.power(E,2) + np.power(H_val,2))   
        R = np.sqrt(np.power(F,2) + np.power(G,2))

        a1 = np.arctan2(G, F)
        a2 = np.arctan2(H_val, E)
        
        svd[0,j] = Q + R                # sx sy
        svd[1,j] = Q - R
        
        svd[2,j] = (a2 - a1)/2# theta gamma
        svd[3,j] = (a2 + a1)/2
        
        svd[4,j] = H1[0,2]    #tx, ty
        svd[5,j] = H1[1,2]
    
    idx = 2
    #finding the index of the median value
    median_idx = np.argsort(svd[idx,:])[len(svd[idx,:])//2]
    
    H_ret = np.zeros([3,3])
    H_ret[0,2] = svd[4,median_idx]
    H_ret[1,2] = svd[5,median_idx]
    H_ret[2,2] = 1;
    
    U_d = np.array([[np.cos(svd[2,median_idx]), np.sin(svd[2,median_idx])], [-np.sin(svd[2,median_idx]), np.cos(svd[2,median_idx])]])
    S_d = np.array([[svd[0,median_idx], 0], [0, svd[1,median_idx]]])
    V_d = np.array([[np.cos(svd[3,median_idx]), np.sin(svd[3,median_idx])], [-np.sin(svd[3,median_idx]), np.cos(svd[3,median_idx])]])
    
    H_ret[0:2,0:2] = np.transpose(np.matmul(np.matmul(U_d,S_d), V_d))

    return H_ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to fetoscopy sequences images cropped to inscribed square of the fetoscope", default="data/seq/images_crop/")
    parser.add_argument("--save_path", help="Path to output folder to save relative homographies as text and resulting mosaic as images", default="results_mosaic/" )
    parser.add_argument("--iformat", help="Input image file format png or jpgOutput path to save plot", default="png")
    parser.add_argument("--seq_start_cnt", type=int, help="count ID of the first frame inthe sequence", default=1)
    parser.add_argument("--seq_length", type=int, help="Total number of frames (for which homograhy to be estimated) in a sequence starting from seq_start_cnt", default=10)
    
    args = parser.parse_args()
    
    # load model
    chkpt = "checkpoints/DSM_weights.hdf5"   
    model = load_model(chkpt, custom_objects={'euclidean_distance': euclidean_distance})
    
    
    # Input params
    data_path = args.data_path
    save_path = args.save_path 
    iformat = args.iformat
    nn = args.seq_start_cnt - 1
    no_samples = args.seq_length
    
        
    # height and width for resizing the input frame
    height = 256
    width = 256
            
    if not os.path.exists(save_path):
        os.makedirs(save_path)
       
    
    loc_list = sorted(glob(data_path+"*."+iformat))
    
    # Predict homography between image pairs 
    H = [np.array([[1, 0, 0],[0, 1, 0],[0,0,1]])]
    fname = (loc_list[nn].split('/'))[-1]
    print(fname)
    fname = fname.replace(iformat,'txt')
    # for saving the relative homography as txt
    np.savetxt(save_path+fname, H[0]) 
        
    
    """
    Homography computed N times between consecutive frame are decomposed using SVD
    Median value over theta is computed
    Consistent homography matrix is obtained using the index of the median value
    """
    # no_samples = 100 #len(loc_list)
    for j in range(no_samples-1):   
        H_prune = []
        for i in range(49):  #199
    
            im1 = loc_list[nn+j]
            im2 = loc_list[nn+j+1]
        
            color_imageA, color_imageB, patchA, patchB, val_image, four_points = get_pair_data(im1, im2)
            
            predicted_four_points_d = model.predict(val_image)
            predicted_four_points_d = predicted_four_points_d.reshape(4,2)
            predicted_four_points = four_points+predicted_four_points_d
            
            H1 = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(predicted_four_points))
            H_prune.append(H1)
        Hret = SVD(H_prune)
        fname = (loc_list[nn+j+1].split('/'))[-1]
        print(fname)
        fname = fname.replace(iformat,'txt')
        
        # for saving the relative homography as txt
        np.savetxt(save_path+fname, Hret) 
        H.append(Hret)
            
    # Get range of Homography needed to translate the mosaic image to positive xy axis
    # Get relative homography Hrel wrt the first frame
    Hrel = [np.array([[1, 0, 0],[0, 1, 0],[0,0,1]])]
    img_corner_1 = np.array([[0,0],[height,0], [0,width],[height,width]], dtype='float')
    img_corner = np.array([img_corner_1])
    for j in range(no_samples-1):   
                
        H_inverse = H[nn+j+1]
        Hrel.append(np.matmul(H_inverse,Hrel[-1]))
        
        img_corner_out = cv2.perspectiveTransform(img_corner,Hrel[-1])
        img_corner_1 = np.concatenate((img_corner_1,np.squeeze(img_corner_out)))
            
    minx = np.min(img_corner_1[:,0])
    maxx = np.max(img_corner_1[:,0])
    miny = np.min(img_corner_1[:,1])
    maxy = np.max(img_corner_1[:,1])
    
    H_offset = np.array([[1, 0, 0],[0, 1, 0],[0,0,1]], dtype='float')
    if minx < 0:
        offsetx = -(minx)
        H_offset[0,2] = offsetx
    else:
        offsetx = 0
        
    if miny < 0:
        offsety = -(miny)
        H_offset[1,2] = offsety
    else: 
        offsety = 0
        
    # Add offset to range of reltaive homography
    img_corner_2 = np.array([[0,0],[height,0], [0,width],[height,width]], dtype='float')
    img_corner0 = np.array([img_corner_2])
    H_scale = np.array([[1, 0, 0],[0, 1, 0],[0,0,1]], dtype='float')
    for j in range(len(Hrel)):
        Hrel[j] =  np.matmul(H_offset,Hrel[j])
        #Hrel[j] =  np.matmul(H_scale,Hrel[j])   # add scale to make it 512 x 512 
        
        img_corner_out2 = cv2.perspectiveTransform(img_corner0,Hrel[j])
        img_corner_2 = np.concatenate((img_corner_2,np.squeeze(img_corner_out2)))   
    
    minxx = np.min(img_corner_2[:,0])
    maxxx = np.max(img_corner_2[:,0])
    minyy = np.min(img_corner_2[:,1])
    maxyy = np.max(img_corner_2[:,1])
    
    
    # For ploting the mosaic
    mosaic = np.zeros([int(round(maxxx)),int(round(maxyy)),3]).astype(int)
    dst = np.zeros([int(round(maxxx)),int(round(maxyy)),3]).astype(np.uint8)
    for j in range(no_samples):    
        color_image = cv2.imread(loc_list[nn+j],cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (256, 256))
            
        warped_image = cv2.warpPerspective(color_image,Hrel[j],(int(round(maxyy)),int(round(maxxx))), flags=cv2.INTER_CUBIC,    
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Create masked composite
        (ret,data_map) = cv2.threshold(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY),0, 255, cv2.THRESH_BINARY)
        
        # Creating kernel 
        kernel = np.ones((3,3), np.uint8) 
          
        # Using cv2.erode() method  
        data_map = cv2.erode(data_map, kernel) 
    
        
        dst = cv2.add(mosaic, dst, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
        
        # Now add the warped image
        mosaic = cv2.add(dst, warped_image, dtype=cv2.CV_8U)
        
        cv2.imwrite('results_mosaic/output_'+str("{:04d}".format(j))+'.jpg',mosaic)
        plt.imshow(mosaic)
        plt.show()
        dst = np.copy(mosaic)
        mosaic = np.zeros([int(round(maxxx)),int(round(maxyy)),3])