import os
import glob
import numpy as np

from modules.calibration_system import Bebop2CameraCalibration

# Initialize the main directory path and the camera calibration object.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
B6 = Bebop2CameraCalibration(axis_length=80)

def capture_images(save: bool = False, dataset_name: str = 'uav_B6_?') -> None:
    """
    Capture images for calibration and optionally save them to a specified directory.

    Args:
        save (bool): If True, the captured images will be saved to the specified directory.
        dataset_name (str): Name of the directory to save the captured images.
    """
    dataset_path = os.path.join(BASE_DIR, dataset_name)
    success = B6.capture_images(save=save, path=dataset_path)
    
    if success:
        print("Images captured successfully!")
    else:
        print("Failed to capture images.")
        
def intrinsic_calibration() -> None:
    """
    Perform intrinsic calibration of the camera using captured images.
    """
    
    _, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = B6.calibrate_camera(
        object_points=object_points,
        image_points=image_points,
        image_size=(1280, 720)
    )
    
    print(f'\nIntrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'\nDistortion coefficients:\n{distortion_coeffs.ravel()}\n')

    results_path = os.path.join(BASE_DIR, 'results')
    result_file = os.path.join(results_path, 'B6_?.npz')
    B6.save_calibration(result_file, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs)

def extrinsic_calibration() -> None:
    """
    Perform extrinsic calibration of the camera using captured images.
    """
    
    results_path = os.path.join(BASE_DIR, 'results')
    result_file = os.path.join(results_path, 'B6_calibration.npz')
    intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors = B6.load_calibration(result_file)
    
    print(f'Intrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'Distortion coefficients:\n{distortion_coeffs.ravel()}\n')

    B6.display_extrinsic_parameters(
        np.hstack(rotation_vectors),
        np.hstack(translation_vectors),
        object_pattern
    )

def validate_calibration() -> None:
    """
    Validate the camera calibration by comparing the projected points with the captured image points.
    """

    results_path = os.path.join(BASE_DIR, 'results')
    result_file = os.path.join(results_path, 'B6_calibration.npz')
    intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors = B6.load_calibration(result_file)
    
    print(f'Intrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'Distortion coefficients:\n{distortion_coeffs.ravel()}\n')

    B6.validate_calibration(
        object_points,
        rotation_vectors,
        translation_vectors,
        intrinsic_matrix,
        distortion_coeffs,
        image_points
    )


if __name__ == '__main__':
    # Uncomment the line below to capture images and save them.
    # capture_images(save=True, dataset_name='datasets/uav_B6_1')

    datasets_path = os.path.join(BASE_DIR, 'datasets')
    dataset_path = os.path.join(datasets_path, 'uav_B6_1')
    image_files = glob.glob(os.path.join(dataset_path, '*.png'))
    
    print(f"\nImages found: {len(image_files)}\n")
    if len(image_files) > 1:
        object_points, image_points, object_pattern = B6.process_images(image_files=image_files, num_images=15, display=True)

        intrinsic_calibration()
        extrinsic_calibration()
        validate_calibration()
    

    