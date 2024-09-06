import cv2
import glob
import os

import kornia as K
import numpy as np

from pyparrot.Bebop import Bebop

from modules.auxiliary import DisplayPostureCamera
from modules.calibration.Camera import Camera
from modules.feature.FeatureExtractor import LoFTRExtractor
from modules.feature.FeatureMatcher import LoFTRMatcher
from modules.feature.interfaces import InterfaceFeatureExtractor, InterfaceFeatureMatcher, InterfaceModelFitter
from modules.pose_estimation import ModelFitter, VisualOdometry

from typing import Optional

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to save the calibration dataset
DATASET_PATH = os.path.join(BASE_DIR, "datasets/calibration/board_B6_11")

# Path to save and load the calibration results
RESULT_FILE = os.path.join(BASE_DIR, "results/calibration/B6_1.npz")

# Path to the images for feature matching
IMAGE_FILES = sorted(glob.glob(os.path.join(BASE_DIR, "datasets/matching/images/*.png")))

# Operating modes
MODES = ["camera_calibration", "feature_matching", "pose_estimation"]
mode = 2


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

def run_feature_matching_LoFTR(
    img1: str,
    img2: str,
    feature_extractor: InterfaceFeatureExtractor,
    feature_matcher: InterfaceFeatureMatcher,
):
    """
    Matches features between two images.
    
    :param img1: The first image file path.
    :param img2: The second image file path.
    """

    # Pretrained model
    img1_tensor = K.io.load_image(img1, K.io.ImageLoadType.RGB32)[None, ...]
    img2_tensor = K.io.load_image(img2, K.io.ImageLoadType.RGB32)[None, ...]
    
    correspondences, _ = feature_matcher.match_features(
        feature_extractor.extract_features(img1_tensor, img2_tensor)
    )

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    a = mkpts1[inliers.squeeze()]
    b = mkpts0[inliers.squeeze()]
    idx = np.random.choice(np.arange(len(a-b)), int(0.05 * len(a-b)))
    diff = (a-b).mean(axis=0) * 0.42735042735042733
    print(f" Diff: {diff}", end='')
    
    feature_matcher.show_matches(img1_tensor, img2_tensor, mkpts0, mkpts1, inliers)

def run_visual_odometry(
    img1: str,
    img2: str,
    feature_extractor: InterfaceFeatureExtractor,
    feature_matcher: InterfaceFeatureMatcher,
    calibration: Camera,
    model_fitter: InterfaceModelFitter,
    params: Optional[dict] = None
):
    """
    Perform pose estimation between images.
    """
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter, params)
    
    # step 1 -> Read a new Ik frame
    img1_tensor = K.io.load_image(img1, K.io.ImageLoadType.RGB32)[None, ...]
    img2_tensor = K.io.load_image(img2, K.io.ImageLoadType.RGB32)[None, ...]

    intrinsic_matrix, _, _, _ = calibration.load_calibration(RESULT_FILE)

    M = vo.process_frames(img1_tensor, img2_tensor, intrinsic_matrix)
    print(f"Matrix: \n{M}")

# Escolha do modo de operação
if MODES[mode] == "camera_calibration":
    bebop_drone = Bebop()
    camera = Camera(
        uav=bebop_drone,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    )

    run_camera_calibration(
        camera=camera,
        dataset_path=DATASET_PATH,
        result_file=RESULT_FILE,
        attempts=100,
        save=False,
        num_images=50,
        display=False
    )

elif MODES[mode] == "feature_matching":
    # Image files for feature matching
    img1 = IMAGE_FILES[4]
    img2 = IMAGE_FILES[5]
    feature_extractor = LoFTRExtractor(parameters={"pretrained": "indoor"})
    feature_matcher = LoFTRMatcher(feature_extractor)
    run_feature_matching_LoFTR(img1, img2, feature_extractor, feature_matcher)

elif MODES[mode] == "pose_estimation":
    img1 = IMAGE_FILES[49]
    img2 = IMAGE_FILES[51]
    feature_extractor = LoFTRExtractor(parameters={"pretrained": "indoor"})
    feature_matcher = LoFTRMatcher(feature_extractor)
    calibration = Camera(uav=Bebop())
    model_fitter = ModelFitter(prob=0.999, reproj_thresh=0.4)
    params = {
        "num_points": 8,
        "epsilon": 0.5,
        "prob": 0.999,
        "display": True
    }
    run_visual_odometry(img1, img2, feature_extractor, feature_matcher, calibration, model_fitter, params)

else:
    print("Invalid mode of operation.")
    exit(1)