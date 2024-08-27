import os
import glob
import numpy as np

from modules.calibration_system import Bebop2CameraCalibration
from typing import Tuple

# Initialize the main directory path and the camera calibration object.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(BASE_DIR, 'results')
datasets_path = os.path.join(BASE_DIR, 'datasets')

def capture_images(B: Bebop2CameraCalibration, save: bool = False, dataset_name: str = 'uav_B6_?') -> None:
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
    success = B.capture_images(save=save, path=dataset_path)
    
    if success:
        print("Images captured successfully!")
    else:
        print("Failed to capture images.")
        
def intrinsic_calibration(B: Bebop2CameraCalibration, resolution_image: Tuple[int,int],  name_file: str = 'B6_?.npz') -> None:
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

def extrinsic_calibration(B: Bebop2CameraCalibration, name_file: str = 'B6_?.npz') -> None:
    """
    Perform extrinsic calibration using the given Bebop2CameraCalibration object and the specified calibration file.
    
    Parameters:
        B (Bebop2CameraCalibration): The Bebop2CameraCalibration object used for calibration.
        name_file (str): The name of the calibration file to load. Default is 'B6_?.npz'.
    Returns:
        None
    """
    
    result_file = os.path.join(results_path, name_file)
    intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors = B.load_calibration(result_file)
    
    print(f'Intrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'Distortion coefficients:\n{distortion_coeffs.ravel()}\n')

    B.display_extrinsic_parameters(
        np.hstack(rotation_vectors),
        np.hstack(translation_vectors),
        object_pattern
    )

def validate_calibration(B: Bebop2CameraCalibration, name_file:str = 'B6_?.npz') -> None:
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
    
    print(f'Intrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'Distortion coefficients:\n{distortion_coeffs.ravel()}\n')

    B.validate_calibration(
        object_points,
        rotation_vectors,
        translation_vectors,
        intrinsic_matrix,
        distortion_coeffs,
        image_points
    )


if __name__ == '__main__':
    B = [None]*1
    for i in range(len(B)): 
        B[i] = Bebop2CameraCalibration(axis_length=80)

    # Uncomment the line below to capture images and save them.
    # capture_images(B[0],save=True, dataset_name='datasets/uav_B6_3')

    image_files = glob.glob(os.path.join(datasets_path, 'board_B6_1/*.png'))
    
    print(f"\nImages found: {len(image_files)}\n")
    if len(image_files) > 1:
        object_points, image_points, object_pattern = B[0].process_images(image_files=image_files, num_images=20, display=False)

        intrinsic_calibration(B[0], resolution_image=(480,856))
        extrinsic_calibration(B[0])
        validate_calibration(B[0])
    