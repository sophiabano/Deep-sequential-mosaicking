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


This script performs Controlled Data Augmentation.

"""


import numpy as np
#import cv2, scipy.io
#from glob import glob
#from numpy.linalg import inv
import random
import cv2
from matplotlib import pyplot as plt

def min_max(four_points):
    minx = np.min([four_points[0][0], four_points[1][0], four_points[2][0], four_points[3][0]])
    maxx = np.max([four_points[0][0], four_points[1][0], four_points[2][0], four_points[3][0]])
    miny = np.min([four_points[0][1], four_points[1][1], four_points[2][1], four_points[3][1]]) 
    maxy = np.max([four_points[0][1], four_points[1][1], four_points[2][1], four_points[3][1]]) 
    return minx, maxx, miny, maxy


def data_augmentation_pert(x, y, patch_size, width, height, rho, scale, angle):
    """ 
    Input:
        x and y location of top left corner of P_A
        patch_size - size of the patch which is 128
        width - of the image which is 256
        height - of the image which is 256
        rho - maximium displacement value which is set to 16
        angle - maximum rotation angle value which is set to 5

    Output:
        four_points - Four corners of P_A
        perturbed_four_points - Four corners of P_B
    """
    
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    perb_fix1 = random.randint(-rho, rho)
    perb_fix2 = random.randint(-rho, rho)
    
    perb_fix3 = random.randint(-rho/2, rho/2)
    perb_fix4 = random.randint(-rho/2, rho/2)
    
    perturbed_four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        
    # Translation
    if random.randint(0,1):
        top_left_point_pert = (top_left_point[0] + perb_fix1, top_left_point[1] +  perb_fix2)
        bottom_left_point_pert = (bottom_left_point[0] + perb_fix1, bottom_left_point[1] +  perb_fix2)
        bottom_right_point_pert = (bottom_right_point[0] + perb_fix1, bottom_right_point[1] +  perb_fix2)
        top_right_point_pert = (top_right_point[0] + perb_fix1, top_right_point[1] +  perb_fix2)
        perturbed_four_points = [top_left_point_pert, bottom_left_point_pert, bottom_right_point_pert, top_right_point_pert]
    else:
    # Rotation
    #if random.randint(0,1):
        minx, maxx, miny, maxy = min_max(perturbed_four_points)
    
        theta = random.randint(-angle*10, angle*10)/10 * np.pi/180
        trans_rot = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        top_left_point_pert = (perturbed_four_points[0][0] - ((maxx - minx)/2),perturbed_four_points[0][1] - ((maxy - miny)/2))
        bottom_left_point_pert = (perturbed_four_points[1][0] - ((maxx - minx)/2),perturbed_four_points[1][1] - ((maxy - miny)/2))
        bottom_right_point_pert = (perturbed_four_points[2][0] - ((maxx - minx)/2),perturbed_four_points[2][1] - ((maxy - miny)/2))
        top_right_point_pert = (perturbed_four_points[3][0] - ((maxx - minx)/2),perturbed_four_points[3][1] - ((maxy - miny)/2))
        
        top_left_point_pert = np.matmul(trans_rot,np.transpose(top_left_point_pert))
        bottom_left_point_pert = np.matmul(trans_rot,np.transpose(bottom_left_point_pert))
        bottom_right_point_pert = np.matmul(trans_rot,np.transpose(bottom_right_point_pert))
        top_right_point_pert = np.matmul(trans_rot,np.transpose(top_right_point_pert))
    
        top_left_point_pert = (top_left_point_pert[0] + ((maxx - minx)/2), top_left_point_pert[1] + ((maxy - miny)/2))
        bottom_left_point_pert = (bottom_left_point_pert[0] + ((maxx - minx)/2), bottom_left_point_pert[1] + ((maxy - miny)/2))
        bottom_right_point_pert = (bottom_right_point_pert[0] + ((maxx - minx)/2), bottom_right_point_pert[1] + ((maxy - miny)/2))
        top_right_point_pert = (top_right_point_pert[0] + ((maxx - minx)/2), top_right_point_pert[1] + ((maxy - miny)/2))    
        
        minx, maxx, miny, maxy = min_max([top_left_point_pert, bottom_left_point_pert, bottom_right_point_pert, top_right_point_pert])
        
        if minx > 0 and miny > 0 and maxx < width and maxy <width:    
            perturbed_four_points = [top_left_point_pert, bottom_left_point_pert, bottom_right_point_pert, top_right_point_pert]
            
        if random.randint(0,1):
            top_left_point_pert = (perturbed_four_points[0][0] + perb_fix1, perturbed_four_points[0][1] +  perb_fix2)
            bottom_left_point_pert = (perturbed_four_points[1][0] + perb_fix1, perturbed_four_points[1][1] +  perb_fix2)
            bottom_right_point_pert = (perturbed_four_points[2][0] + perb_fix1, perturbed_four_points[2][1] +  perb_fix2)
            top_right_point_pert = (perturbed_four_points[3][0] + perb_fix1, perturbed_four_points[3][1] +  perb_fix2)
            
            if minx > 0 and miny > 0 and maxx < width and maxy <width:  
                perturbed_four_points = [top_left_point_pert, bottom_left_point_pert, bottom_right_point_pert, top_right_point_pert]
  
    perturbed_four_points = ((round(perturbed_four_points[0][0]), round(perturbed_four_points[0][1])), (round(perturbed_four_points[1][0]), round(perturbed_four_points[1][1])), (round(perturbed_four_points[2][0]), round(perturbed_four_points[2][1])), (round(perturbed_four_points[3][0]), round(perturbed_four_points[3][1])))        
    return four_points, perturbed_four_points
 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--Img_name", help="Path to a test image with image name")
    
    args = parser.parse_args()
    
    Img_name = args.Img_name
    
    rho = 16    # 32
    patch_size = 256
    height = 300
    width = 300
    angle = 3
    scale = 0.05
            
    color_image = cv2.imread(Img_name)
    gray_image = cv2.resize(color_image, (width, height))
    
    y = 22#random.randint(rho, height - rho - patch_size)
    x = 22#random.randint(rho, width - rho - patch_size)
    
    four_points, perturbed_four_points = data_augmentation_pert(x, y, patch_size, width, height, rho, scale, angle)
    cv2.polylines(gray_image, np.int32([four_points]), 1, (0,255,0))
    cv2.polylines(gray_image, np.int32([perturbed_four_points]), 1, (0,255,255))
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    
    plt.axis("off")
    
    plt.show()


