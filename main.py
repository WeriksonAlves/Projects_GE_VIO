import os
import glob
import cv2

import numpy as np

from modules.calibration.Camera import Camera
from modules.auxiliary.DisplayPostureCamera import DisplayPostureCamera

from modules.feature.FeatureExtractor import *
from modules.feature.FeatureMatcher import *
from modules.feature.ModelFitter import ModelFitter
from modules.feature.VisualOdometry import VisualOdometry

from pyparrot.Bebop import Bebop

# Initialize the main directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "datasets/calibration/board_B6_1")
result_file = os.path.join(BASE_DIR, "results/calibration/B6_1.npz")

list_mode = [
    "camera_calibration",
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


elif list_mode[mode] == "feature_matching":
    image_files = sorted(glob.glob(os.path.join(BASE_DIR, "datasets/matching/images/*.png")))
    gray_image_files = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_files]

    # Load images
    img1 = image_files[4]
    img2 = image_files[5]

    #%% Initialize components: Classic method 

    # Feature extractor
    feature_extractor = FAST()

    # Feature matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    parammeters = {"index_params": index_params, "search_params": search_params}
    feature_matcher = FlannMatcher(parammeters)
    
    # Model fitter
    model_fitter = ModelFitter()

    # Visual odometry
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    gray_img1 = vo.read_frame(img1)
    gray_img2 = vo.read_frame(img2)
    vo.extract_and_match_features(gray_img1, gray_img2, True)
    cv2.waitKey(100)

    #%% Initialize components: LoFTR method

    # Load images
    img1 = K.io.load_image(image_files[4], K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(image_files[5], K.io.ImageLoadType.RGB32)[None, ...]

    feature_matcher = LoFTRMatcher(parameters={"pretrained": "indoor"})

    mkpts0, mkpts1, inliers = feature_matcher.match_features(img1, img2)
    feature_matcher.show_matches(img1, img2, mkpts0, mkpts1, inliers)

    
    #%% Quit program

    #Press 'q' to close the window
    if cv2.waitKey(0) and 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(100)
    
    
    
elif list_mode[mode] == "pose_estimation":
    # intrinsic_matrix, _, _, _ = camera.load_calibration(result_file)

    # # Process frames and estimate transformation matrix
    # C = vo.process_frames(
    #     img1=img1, 
    #     img2=img2, 
    #     intrinsic_matrix=intrinsic_matrix,
    #     display=True
    # )

    # print(f"Estimated Transformation Matrix: \n{C}")

    pass
    
