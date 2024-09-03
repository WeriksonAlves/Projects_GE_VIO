import os
import glob
import cv2

import numpy as np

# from modules import *
from modules.calibration.Camera import Camera
from modules.auxiliary.DisplayPostureCamera import DisplayPostureCamera
from modules.feature.FeatureMatching import (
    FeatureMatcher,
    FeatureExtractor,
    ModelFitter,
    VisualOdometry,
)
from pyparrot.Bebop import Bebop

# Initialize the main directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "datasets/calibration/board_B6_1")
result_file = os.path.join(BASE_DIR, "results/calibration/B6_1.npz")

list_mode = [
    "camera_calibration",
    "correspondences",
    "feature_matching",
    "pose_estimation",
]
mode = 1

B6 = Bebop()
camera = Camera(
    uav=B6,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
)

if list_mode[mode] == "camera_calibration":
    # if camera.capture_images(attempts=100, path=dataset_path):
    #     print("Images captured successfully!")
    # else:
    #     print("No capture images.")

    image_files = sorted(
        glob.glob(os.path.join(os.path.join(BASE_DIR, dataset_path), "*.png"))
    )
    print(f"\n\nImages found: {len(image_files)}\n")
    if len(image_files) > 1:
        object_points, image_points, object_pattern = camera.process_images(
            image_files=image_files, num_images=50, display=True
        )
        (
            _,
            intrinsic_matrix,
            distortion_coeffs,
            rotation_vecs,
            translation_vecs,
        ) = camera.calibrate_camera(
            object_points=object_points,
            image_points=image_points,
            image_size=cv2.imread(image_files[0]).shape[:2],
        )

        print(f"\nIntrinsic matrix:\n{intrinsic_matrix}\n")
        print(f"\nDistortion coefficients:\n{distortion_coeffs.ravel()}\n")

        camera.save_calibration(
            result_file,
            intrinsic_matrix,
            distortion_coeffs,
            rotation_vecs,
            translation_vecs,
        )

        camera.validate_calibration(
            object_points,
            rotation_vecs,
            translation_vecs,
            intrinsic_matrix,
            distortion_coeffs,
            image_points,
        )

        display_posture = DisplayPostureCamera()
        display_posture.display_extrinsic_parameters(
            np.hstack(rotation_vecs),
            np.hstack(translation_vecs),
            object_pattern,
        )


elif list_mode[mode] == "correspondences":
    image_files = sorted(
        glob.glob(os.path.join(BASE_DIR, "datasets/matching/images/*.png"))
    )
    gray_image_files = [
        cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_files
    ]

    # Load images
    img1 = "/home/ubuntu/Documentos/Werikson/GitHub/env_GE-VIO/Projects_GE_VIO/references/notre_dame_1.jpg"#image_files[49]
    img2 = "/home/ubuntu/Documentos/Werikson/GitHub/env_GE-VIO/Projects_GE_VIO/references/notre_dame_2.jpg"#image_files[50]

    # Initialize components
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)

    search_params = dict(checks=50)
    feature_matcher = FeatureMatcher("FLANN", "DEFAULT", {"index_params": index_params, "search_params": search_params}, {"k": 2})
    model_fitter = ModelFitter()

    
    feature_extractor = FeatureExtractor("SIFT")
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    gray_img1 = vo.read_frame(img1)
    gray_img2 = vo.read_frame(img2)
    vo.extract_and_match_features(gray_img1, gray_img2, True)

    feature_extractor = FeatureExtractor("ORB")
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    gray_img1 = vo.read_frame(img1)
    gray_img2 = vo.read_frame(img2)
    vo.extract_and_match_features(gray_img1, gray_img2, True)

    feature_extractor = FeatureExtractor("AKAZE")
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    gray_img1 = vo.read_frame(img1)
    gray_img2 = vo.read_frame(img2)
    vo.extract_and_match_features(gray_img1, gray_img2, True)

    feature_extractor = FeatureExtractor("FAST", {"suppression": True, "threshold": 125})
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    gray_img1 = vo.read_frame(img1)
    gray_img2 = vo.read_frame(img2)
    vo.extract_and_match_features(gray_img1, gray_img2, True)
    

    
    
    
    
    
    

    # intrinsic_matrix, _, _, _ = camera.load_calibration(result_file)

    # # Process frames and estimate transformation matrix
    # C = vo.process_frames(
    #     img1=img1, 
    #     img2=img2, 
    #     intrinsic_matrix=intrinsic_matrix,
    #     display=True
    # )

    # print(f"Estimated Transformation Matrix: \n{C}")

    # Press 'q' to close the window
    if cv2.waitKey(0) and 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(100)
