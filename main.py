import os
import glob
import cv2

import numpy as np

from modules import *
from typing import Tuple
from pyparrot.Bebop import Bebop

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

    display = DisplayPosture()
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
    B6 = Bebop()
    camera = Camera(B6)
    feature_matching = MySIFT()
    list_mode = ['calibration', 'feature_matching', 'pose_estimation']
    mode = 1

    if list_mode[mode] == 'calibration':
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
        image_files = glob.glob(os.path.join(datasets_path, 'imgs/*.jpg')) # Raquel
        gray_image_files = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in image_files]

        keypoints, descriptors = feature_matching.my_detectAndCompute(gray_image_files[0])
        feature_matching.drawKeyPoints(gray_image_files[0], keypoints)
        feature_matching.saveKeypoints('keypoints.npz', keypoints, descriptors)
        matches = feature_matching.matchingKeypoints(descriptors, descriptors)
    
