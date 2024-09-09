import cv2
import glob
import os
import time

import kornia as K
import numpy as np

from modules.auxiliary import DisplayPostureCamera
from modules.calibration.Camera import Camera
from modules.feature.FeatureExtractor import LoFTRExtractor
from modules.feature.FeatureMatcher import LoFTRMatcher
from modules.feature.interfaces import InterfaceFeatureExtractor, InterfaceFeatureMatcher, InterfaceModelFitter
from modules.pose_estimation import VisualOdometry
from pyparrot.Bebop import Bebop
from typing import Optional

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to save the calibration dataset
DATASET_PATH = os.path.join(BASE_DIR, "datasets/calibration/board_B6_11")

# Path to save and load the calibration results
RESULT_FILE = os.path.join(BASE_DIR, "results/calibration/B6_1.npz")

# Path to the images for feature matching
IMAGE_FILES = sorted(glob.glob(os.path.join(BASE_DIR, "datasets/matching/images_500/*.png")))

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

def run_visual_odometry():
    """
    Perform pose estimation between images.
    """
    # Load the feature extractor and matcher
    feature_extractor = LoFTRExtractor(parameters={"pretrained": "indoor"})
    feature_matcher = LoFTRMatcher(feature_extractor)
    params = {
        "num_points": 8,
        "epsilon": 0.5,
        "prob": 0.999,
        "scale": 0.5*100,
        "display": False
    }
    vo = VisualOdometry(feature_extractor, feature_matcher, params)
    
    # Load the camera calibration
    calibration = Camera(uav=Bebop())
    intrinsic_matrix, _, _, _ = calibration.load_calibration(RESULT_FILE)

    # Variables to store the differences
    diff_list = []
    diff_acum = np.zeros((4,4))
    
    M = np.eye(4)

    # Rotation in Z
    angle = -np.pi/2
    # M[0,0] = np.cos(angle)
    # M[0,1] = -np.sin(angle)
    # M[1,0] = np.sin(angle)
    # M[1,1] = np.cos(angle)
    
    
    start_time = time.time()
    # Process the images
    for i in range(0, len(IMAGE_FILES)-1):
        img1 = IMAGE_FILES[i]
        img2 = IMAGE_FILES[i+1]

        # step 1 -> Read a new Ik frame
        img1_tensor = K.io.load_image(img1, K.io.ImageLoadType.RGB32)[None, ...]
        img2_tensor = K.io.load_image(img2, K.io.ImageLoadType.RGB32)[None, ...]

        M = vo.process_frames(img1_tensor, img2_tensor, intrinsic_matrix, M)

        diff_acum += M
        diff_list.append(M)

        # Formatando o print para números alinhados
        print(f"{'Img':<5} [{i:>3d} {i+1:3d}] {'Difference in relation:':<25} {'Frames:':<10} [{M[0,3]:>8.4f} {M[1,3]:>8.4f} {M[2,3]:>8.4f}] {'Accumulated:':<15} [{diff_acum[0,3]:>8.4f} {diff_acum[1,3]:>8.4f} {diff_acum[2,3]:>8.4f}] {'Time (s):':<5} [{time.time()-start_time:>8.1f}]")

    np.save(os.path.join(BASE_DIR, "results/pose_estimation/pose_Felipe1_500_1.npy"), diff_list)

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
    run_visual_odometry()

else:
    print("Invalid mode of operation.")
    exit(1)