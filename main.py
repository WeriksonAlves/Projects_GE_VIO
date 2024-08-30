import os
import glob
import cv2

from modules import *
from pyparrot.Bebop import Bebop


# Initialize the main directory path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
list_mode = ['camera calibration', 'feature_matching', 'pose_estimation']
mode = 0

if list_mode[mode] == 'camera calibration':
    B6 = Bebop()
    camera = Camera(B6)
    
    dataset_path = os.path.join(BASE_DIR, 'datasets/calibration/board_B6_99')
    result_file = os.path.join(BASE_DIR, 'results/calibration/B6_99.npz')

    if camera.capture_images(attempts=100, save=True, path=dataset_path):
        print("Images captured successfully!")
    else:
        print("Failed to capture images.")

    image_files = glob.glob(os.path.join(os.path.join(BASE_DIR, dataset_path),'*.png')) # Raquel
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

        
elif list_mode[mode] == 'feature_matching':    
    image_files = glob.glob(os.path.join(BASE_DIR, 'datasets/test/images/*.png'))

    gray_image_files = [cv2.imread(image) for image in image_files]
    
    i1 = 52
    i2 = 57

    sift_feature_matching = FeatureMatching(cv2.SIFT_create())
    sift_keypoints_1, sift_descriptors_1 = sift_feature_matching.my_detectAndCompute(gray_image_files[i1])
    sift_feature_matching.drawKeyPoints(gray_image_files[i1], sift_keypoints_1, imageName='SIFT Keypoints 1')
    sift_keypoints_2, sift_descriptors_2 = sift_feature_matching.my_detectAndCompute(gray_image_files[i2])
    sift_matches = sift_feature_matching.matchingKeypoints(sift_descriptors_1, sift_descriptors_2)
    sift_matched_image = cv2.drawMatches(gray_image_files[i1], sift_keypoints_1, gray_image_files[i2], sift_keypoints_2, sift_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('SIFT Matches', sift_matched_image)
    cv2.waitKey(10)

    orb_feature_matching = FeatureMatching(cv2.ORB_create())
    orb_keypoints_1, orb_descriptors_1 = orb_feature_matching.my_detectAndCompute(gray_image_files[i1])
    orb_feature_matching.drawKeyPoints(gray_image_files[i1], orb_keypoints_1, imageName='ORB Keypoints 1')
    orb_keypoints_2, orb_descriptors_2 = orb_feature_matching.my_detectAndCompute(gray_image_files[i2])
    orb_matches = orb_feature_matching.matchingKeypoints(orb_descriptors_1, orb_descriptors_2)
    orb_matched_image = cv2.drawMatches(gray_image_files[i1], orb_keypoints_1, gray_image_files[i2], orb_keypoints_2, orb_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('ORB Matches', orb_matched_image)
    cv2.waitKey(10)

    akaze_feature_matching = FeatureMatching(cv2.AKAZE_create())
    akaze_keypoints_1, akaze_descriptors_1 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i1])
    akaze_feature_matching.drawKeyPoints(gray_image_files[i1], akaze_keypoints_1, imageName='AKAZE Keypoints 1')
    akaze_keypoints_2, akaze_descriptors_2 = akaze_feature_matching.my_detectAndCompute(gray_image_files[i2])
    akaze_matches = akaze_feature_matching.matchingKeypoints(akaze_descriptors_1, akaze_descriptors_2)
    akaze_matched_image = cv2.drawMatches(gray_image_files[i1], akaze_keypoints_1, gray_image_files[i2], akaze_keypoints_2, akaze_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('AKAZE Matches', akaze_matched_image)
    cv2.waitKey(10)

    brisk_feature_matching = FeatureMatching(cv2.BRISK_create())
    brisk_keypoints_1, brisk_descriptors_1 = brisk_feature_matching.my_detectAndCompute(gray_image_files[i1])
    brisk_feature_matching.drawKeyPoints(gray_image_files[i1], brisk_keypoints_1, imageName='BRISK Keypoints 1')
    brisk_keypoints_2, brisk_descriptors_2 = brisk_feature_matching.my_detectAndCompute(gray_image_files[i2])
    brisk_matches = brisk_feature_matching.matchingKeypoints(brisk_descriptors_1, brisk_descriptors_2)
    brisk_matched_image = cv2.drawMatches(gray_image_files[i1], brisk_keypoints_1, gray_image_files[i2], brisk_keypoints_2, brisk_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('BRISK Matches', brisk_matched_image)
    cv2.waitKey(10)

    fast_feature_matching = FastFeatureMatching(cv2.FastFeatureDetector_create(), cv2.ORB_create())
    fast_keypoints_1, fast_descriptors_1 = fast_feature_matching.my_detectAndCompute(gray_image_files[i1])
    fast_feature_matching.drawKeyPoints(gray_image_files[i1], fast_keypoints_1, imageName='FAST Keypoints 1')
    fast_keypoints_2, fast_descriptors_2 = fast_feature_matching.my_detectAndCompute(gray_image_files[i2])
    fast_matches = fast_feature_matching.matchingKeypoints(fast_descriptors_1, fast_descriptors_2)
    fast_matched_image = cv2.drawMatches(gray_image_files[i1], fast_keypoints_1, gray_image_files[i2], fast_keypoints_2, fast_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('FAST Matches', fast_matched_image)
    cv2.waitKey(10)


    # Press 'q' to close the window
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

