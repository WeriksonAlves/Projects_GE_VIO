import cv2
import glob
import os

import kornia as K
import numpy as np

from modules.auxiliary import DisplayPostureCamera
from modules.calibration import Camera
from modules.feature.FeatureMatcher import LoFTRMatcher
from modules.feature.interfaces import InterfaceFeatureExtractor, InterfaceFeatureMatcher

from typing import Tuple, List


def run_camera_calibration(
        camera: Camera,
        dataset_path: str,
        result_file: str,
        attempts: int = 100,
        save: bool = False,
        num_images: int = 30,
        display: bool = False
        ):
    """
    Perform camera calibration using a set of images.
    
    :param camera: The camera object used for calibration.
    :param dataset_path: The path to the dataset containing calibration images.
    :param result_file: The file path to save the calibration results.
    :param attempts: The number of attempts to capture images. Defaults to 100.
    :param save: Whether to save the captured images. Defaults to False.
    :param num_images: The number of images to use for calibration. Defaults to 30. Maximum is 150.
    :param display: Whether to display the calibration process. Defaults to False.
    """
    if save:
        success = camera.capture_images(attempts=attempts, save=save, path=dataset_path)
        if success:
            print("Images captured successfully!")
        else:
            print("Failed to capture images.")
            return

    image_files = sorted(glob.glob(os.path.join(dataset_path, '*.png')))
    print(f"\n\nImages found: {len(image_files)}\n")
    
    if len(image_files) < 2: return
    
    object_points, image_points, object_pattern = camera.process_images(
        image_files=image_files, num_images=num_images, display=display
    )
    _, intrinsic_matrix, distortion_coeffs, rvecs, tvecs = camera.calibrate_camera(
        object_points=object_points, 
        image_points=image_points, 
        image_size=cv2.imread(image_files[0]).shape[:2],
    )
    
    print(f'\nIntrinsic matrix:\n{intrinsic_matrix}\n')
    print(f'\nDistortion coefficients:\n{distortion_coeffs.ravel()}\n')

    camera.save_calibration(
        result_file, 
        intrinsic_matrix, 
        distortion_coeffs, 
        rvecs, 
        tvecs)

    camera.validate_calibration(
            object_points,
            rvecs,
            tvecs,
            intrinsic_matrix,
            distortion_coeffs,
            image_points
        )

    display_posture = DisplayPostureCamera()
    display_posture.display_extrinsic_parameters(
        np.hstack(rvecs),
        np.hstack(tvecs),
        object_pattern
        )

def run_feature_matching(
    img1: str,
    img2: str,
):
    """
    Matches features between two images.
    
    :param img1: The first image file path.
    :param img2: The second image file path.
    """
    # Método LoFTR para correspondência de features
    img1_tensor = K.io.load_image(img1, K.io.ImageLoadType.RGB32)[None, ...]
    img2_tensor = K.io.load_image(img2, K.io.ImageLoadType.RGB32)[None, ...]
    
    loftr_matcher = LoFTRMatcher(parameters={"pretrained": "indoor"})
    mkpts0, mkpts1, inliers = loftr_matcher.match_features(img1_tensor, img2_tensor)
    loftr_matcher.show_matches(img1_tensor, img2_tensor, mkpts0, mkpts1, inliers)

def run_visual_odometry(
    feature_extractor: InterfaceFeatureExtractor,
    feature_matcher: InterfaceFeatureMatcher,
    img1: str,
    img2: str,
    params: dict = None
):
    """
    Perform pose estimation between images.
    """
    img1_tensor = K.io.load_image(img1, K.io.ImageLoadType.RGB32)[None, ...]
    img2_tensor = K.io.load_image(img2, K.io.ImageLoadType.RGB32)[None, ...]
    
    loftr_matcher = LoFTRMatcher(parameters={"pretrained": "indoor"})
    mkpts0, mkpts1, inliers = loftr_matcher.match_features(img1_tensor, img2_tensor)
    loftr_matcher.show_matches(img1_tensor, img2_tensor, mkpts0, mkpts1, inliers)
    