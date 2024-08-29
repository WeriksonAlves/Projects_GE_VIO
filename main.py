import os
import glob
import cv2

import numpy as np

from modules import *
from typing import Tuple
from pyparrot.Bebop import Bebop

import cv2
print(cv2.__version__)


# Initialize the main directory path and the camera calibration object.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(BASE_DIR, 'results/calibration')
datasets_path = os.path.join(BASE_DIR, 'datasets/calibration')

def capture_images(B: Camera, save: bool = False, dataset_name: str = 'uav_B6_?') -> None:
    """
    Capture images using the Bebop2CameraCalibration object.
    
    Parameters:
        B (Bebop2CameraCalibration): The Bebop2CameraCalibration object used to capture images.
        save (bool, optional): Flag indicating whether to save the captured images. Defaults to False.
        dataset_name (str, optional): The name of the dataset to save the images. Defaults to 'uav_B6_?'.
    Returns:
        None
    """
    dataset_path = os.path.join(datasets_path, dataset_name)
    success = B.capture_images(attempts=100, save=save, path=dataset_path)
    
    if success:
        print("Images captured successfully!")
    else:
        print("Failed to capture images.")

def intrinsic_calibration(B: Camera, resolution_image: Tuple[int,int],  name_file: str = 'B6_?.npz') -> None:
    """
    Perform intrinsic calibration of the camera using the provided Bebop2CameraCalibration object.
    
    Parameters:
        B (Bebop2CameraCalibration): The Bebop2CameraCalibration object used for calibration.
        name_file (str): The name of the file to save the calibration results. Default is 'B6_?.npz'.
    Returns:
        None
    """
    _, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = B.calibrate_camera(
        object_points=object_points,
        image_points=image_points,
        image_size=resolution_image
    )
    
    print(f'\nIntrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'\nDistortion coefficients:\n{distortion_coeffs.ravel()}\n')

    
    result_file = os.path.join(results_path, name_file)
    B.save_calibration(result_file, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs)

def extrinsic_calibration(B: Camera, name_file: str = 'B6_?.npz') -> None:
    """
    Perform extrinsic calibration using the given Bebop2CameraCalibration object and the specified calibration file.
    
    Parameters:
        B (Bebop2CameraCalibration): The Bebop2CameraCalibration object used for calibration.
        name_file (str): The name of the calibration file to load. Default is 'B6_?.npz'.
    Returns:
        None
    """
    
    result_file = os.path.join(results_path, name_file)
    _, _, rotation_vectors, translation_vectors = B.load_calibration(result_file)

    display = DisplayPostureCamera()
    display.display_extrinsic_parameters(
        np.hstack(rotation_vectors),
        np.hstack(translation_vectors),
        object_pattern
    )

def validate_calibration(B: Camera, name_file:str = 'B6_?.npz') -> None:
    """
    Validates the calibration of the Bebop2 camera.
    
    Parameters:
        B (Bebop2CameraCalibration): The Bebop2CameraCalibration object.
        name_file (str): The name of the calibration file to load. Default is 'B6_?.npz'.
    Returns:
        None
    """
    result_file = os.path.join(results_path, name_file)
    intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors = B.load_calibration(result_file)

    B.validate_calibration(
        object_points,
        rotation_vectors,
        translation_vectors,
        intrinsic_matrix,
        distortion_coeffs,
        image_points
    )


if __name__ == '__main__':
    list_mode = ['calibration', 'feature_matching', 'pose_estimation']
    mode = 1

    if list_mode[mode] == 'calibration':
        B6 = Bebop()
        camera = Camera(B6)
        
        # Uncomment the line below to capture images and save them.
        # capture_images(camera,save=False)

        image_files = glob.glob(os.path.join(datasets_path, 'imgs/*.jpg')) # Raquel
        
        print(f"\nImages found: {len(image_files)}\n")
        if len(image_files) > 1:
            object_points, image_points, object_pattern = camera.process_images(image_files=image_files, num_images=30, display=False)

            intrinsic_calibration(camera, resolution_image=(640,480))
            validate_calibration(camera)
            extrinsic_calibration(camera)
            
    elif list_mode[mode] == 'feature_matching':    
        image_files = glob.glob(os.path.join(BASE_DIR, 'datasets/test/images/*.png'))
        gray_image_files = [cv2.imread(image) for image in image_files]
        
        i1 = 52
        i2 = 57

        sift_feature_matching = FeatureMatching(cv2.SIFT_create())
        sift_keypoints_1, sift_descriptors_1 = sift_feature_matching.my_detectAndCompute(gray_image_files[i1])
        sift_keypoints_2, sift_descriptors_2 = sift_feature_matching.my_detectAndCompute(gray_image_files[i2])
        sift_matches = sift_feature_matching.matchingKeypoints(sift_descriptors_1, sift_descriptors_2)
        sift_matched_image = cv2.drawMatches(gray_image_files[i1], sift_keypoints_1, gray_image_files[i2], sift_keypoints_2, sift_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('SIFT Matches', sift_matched_image)
        cv2.waitKey(10)

        orb_feature_matching = FeatureMatching(cv2.ORB_create())
        orb_keypoints_1, orb_descriptors_1 = orb_feature_matching.my_detectAndCompute(gray_image_files[i1])
        orb_keypoints_2, orb_descriptors_2 = orb_feature_matching.my_detectAndCompute(gray_image_files[i2])
        orb_matches = orb_feature_matching.matchingKeypoints(orb_descriptors_1, orb_descriptors_2)
        orb_matched_image = cv2.drawMatches(gray_image_files[i1], orb_keypoints_1, gray_image_files[i2], orb_keypoints_2, orb_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('ORB Matches', orb_matched_image)
        cv2.waitKey(10)

        akaze_feature_matching = FeatureMatching(cv2.AKAZE_create())
        akaze_keypoints_1, akaze_descriptors_1 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i1])
        akaze_keypoints_2, akaze_descriptors_2 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i2])
        akaze_matches = akaze_feature_matching.matchingKeypoints(akaze_descriptors_1, akaze_descriptors_2)
        akaze_matched_image = cv2.drawMatches(gray_image_files[i1], akaze_keypoints_1, gray_image_files[i2], akaze_keypoints_2, akaze_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('AKAZE Matches', akaze_matched_image)
        cv2.waitKey(10)

        brisk_feature_matching = FeatureMatching(cv2.BRISK_create())
        brisk_keypoints_1, brisk_descriptors_1 = brisk_feature_matching.my_detectAndCompute(gray_image_files[i1])
        brisk_keypoints_2, brisk_descriptors_2 = brisk_feature_matching.my_detectAndCompute(gray_image_files[i2])
        brisk_matches = brisk_feature_matching.matchingKeypoints(brisk_descriptors_1, brisk_descriptors_2)
        brisk_matched_image = cv2.drawMatches(gray_image_files[i1], brisk_keypoints_1, gray_image_files[i2], brisk_keypoints_2, brisk_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('BRISK Matches', brisk_matched_image)
        cv2.waitKey(10)

        fast_feature_matching = FastFeatureMatching(cv2.FastFeatureDetector_create())
        fast_keypoints_1 = fast_feature_matching.my_detectAndCompute(gray_image_files[i1])
        # fast_feature_matching.drawKeyPoints(gray_image_files[i1], fast_keypoints_1, imageName='FAST Keypoints')


        # Press 'q' to close the window
        if (cv2.waitKey(0) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
    
