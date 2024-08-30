# This file contains the main routines to capture images, calibrate the camera, and validate the calibration.
import numpy as np
import cv2
import os
import glob

from modules.calibration import Camera
from modules.auxiliary import DisplayPostureCamera

from typing import Tuple, List


def camera_calibration(
        camera: Camera, 
        base_dir: str, 
        dataset_path: str, 
        result_file: str,
        attempts: int = 100,
        save: bool = False,
        num_images: int = 30,
        display: bool = False
        ):
    """
    Perform camera calibration using a set of images.
    
    Args:
        camera (Camera): The camera object used for calibration.
        base_dir (str): The base directory path.
        dataset_path (str): The path to the dataset containing calibration images.
        result_file (str): The file path to save the calibration results.
        attempts (int, optional): The number of attempts to capture images. Defaults to 100.
        save (bool, optional): Whether to save the captured images. Defaults to False.
        num_images (int, optional): The number of images to use for calibration. Defaults to 30. Maximum is 150.
        display (bool, optional): Whether to display the calibration process. Defaults to False.
    
    Returns:
        None
    """

    if save:
        success = camera.capture_images(attempts=attempts, save=save, path=dataset_path)
        if success:
            print("Images captured successfully!")
        else:
            print("Failed to capture images.")
            return

    image_files = glob.glob(os.path.join(os.path.join(base_dir, dataset_path),'*.png'))
    print(f"\n\nImages found: {len(image_files)}\n")
    
    if len(image_files) < 2: return
    
    object_points, image_points, object_pattern = camera.process_images(
                                                    image_files=image_files, 
                                                    num_images=num_images, 
                                                    display=display)
    _, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = camera.calibrate_camera(
                                                                                object_points=object_points,
                                                                                image_points=image_points,
                                                                                image_size=cv2.imread(image_files[0]).shape[:2]
                                                                                )
    
    print(f'\nIntrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'\nDistortion coefficients:\n{distortion_coeffs.ravel()}\n')

    camera.save_calibration(
        result_file, 
        intrinsic_matrix, 
        distortion_coeffs, 
        rotation_vecs, 
        translation_vecs)

    camera.validate_calibration(
            object_points,
            rotation_vecs,
            translation_vecs,
            intrinsic_matrix,
            distortion_coeffs,
            image_points
        )

    display_posture = DisplayPostureCamera()
    display_posture.display_extrinsic_parameters(
        np.hstack(rotation_vecs),
        np.hstack(translation_vecs),
        object_pattern
        )


