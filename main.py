import os
import glob
import cv2

from modules import *
from pyparrot.Bebop import Bebop

import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import CENSURE

# Initialize the main directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'datasets/calibration/board_B6_99')
result_file = os.path.join(BASE_DIR, 'results/calibration/B6_1.npz')

list_mode = ['camera_calibration', 'correspondences','feature_matching', 'pose_estimation']
mode = 1

B6 = Bebop()
camera = Camera(B6)

if list_mode[mode] == 'camera_calibration':
    if camera.capture_images(attempts=100, save=True, path=dataset_path):
        print("Images captured successfully!")
    else:
        print("Failed to capture images.")

    image_files = sorted(glob.glob(os.path.join(os.path.join(BASE_DIR, dataset_path),'*.png')))
    print(f"\n\nImages found: {len(image_files)}\n")
    if len(image_files) > 1:
        object_points, image_points, object_pattern = camera.process_images(
                                                        image_files=image_files, 
                                                        num_images=30, 
                                                        display=True)
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

        
elif list_mode[mode] == 'correspondences':    
    image_files = sorted(glob.glob(os.path.join(BASE_DIR, 'datasets/matching/images/*.png')))
    gray_image_files = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in image_files]

    # Load images
    img1 = gray_image_files[0]
    img2 = gray_image_files[1]

    # Initialize components
    parammeter = {'suppression': True, 'threshold': 10}
    feature_extractor = FeatureExtractor(method="FAST", parammeters=parammeter)
    feature_matcher = FeatureMatcher()
    model_fitter = ModelFitter()

    # Create Visual Odometry instance
    vo = VisualOdometry(feature_extractor, feature_matcher, model_fitter)
    
    intrinsic_matrix, _, _, _ = camera.load_calibration(result_file)

    # Process frames and estimate transformation matrix
    F = vo.process_frames(img1, img2, intrinsic_matrix)

    print(f"Estimated Matrix: \n{F}")

    # Visualization of the matches (for debugging/verification)
    kp1, des1 = feature_extractor.extract_features(img1)
    kp2, des2 = feature_extractor.extract_features(img2)
    matches = feature_matcher.match_features(des1, des2)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # Press 'q' to close the window
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

