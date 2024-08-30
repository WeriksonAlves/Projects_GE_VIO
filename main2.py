import os
import glob
import cv2

from modules import *
from pyparrot.Bebop import Bebop

import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import CENSURE

# Initialize the main directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
list_mode = ['camera_calibration', 'correspondences','feature_matching', 'pose_estimation']
mode = 1

if list_mode[mode] == 'camera_calibration':
    B6 = Bebop()
    camera = Camera(B6)
    
    dataset_path = os.path.join(BASE_DIR, 'datasets/calibration/board_B6_99')
    result_file = os.path.join(BASE_DIR, 'results/calibration/B6_99.npz')

    camera_calibration(
        camera=camera, 
        base_dir=BASE_DIR, 
        dataset_path=dataset_path, 
        result_file=result_file,
        attempts=100,
        save=False,
        num_images=30,
        display=True)

        
elif list_mode[mode] == 'correspondences':    
    image_files = glob.glob(os.path.join(BASE_DIR, 'datasets/test/images/*.png'))

    gray_image_files = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_files]
    
    i1 = 52 #ik-1
    i2 = 57 #ik

    # AKAZE method
    akaze_feature_matching = FeatureMatching(cv2.AKAZE_create())
    akaze_keypoints_1, akaze_descriptors_1 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i1])
    akaze_keypoints_2, akaze_descriptors_2 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i2])
    akaze_matches = akaze_feature_matching.matchingKeypoints(akaze_descriptors_1, akaze_descriptors_2)
    akaze_matched_image = cv2.drawMatches(gray_image_files[i1], akaze_keypoints_1, gray_image_files[i2], akaze_keypoints_2, akaze_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('AKAZE Matches', akaze_matched_image)
    cv2.waitKey(10)

    #FAST method
    fast_feature_matching = FastFeatureMatching(cv2.ORB_create())
    fast_keypoints_1, fast_descriptors_1 = fast_feature_matching.my_detectAndCompute(gray_image_files[i1])
    fast_keypoints_2, fast_descriptors_2 = fast_feature_matching.my_detectAndCompute(gray_image_files[i2])
    fast_matches = fast_feature_matching.matchingKeypoints(fast_descriptors_1, fast_descriptors_2)
    fast_matched_image = cv2.drawMatches(gray_image_files[i1], fast_keypoints_1, gray_image_files[i2], fast_keypoints_2, fast_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('FAST Matches', fast_matched_image)
    cv2.waitKey(10)





















    # Press 'q' to close the window
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

