import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, Sequence

from .interfaces import InterfaceFeatureExtractor, InterfaceFeatureMatcher, InterfaceModelFitter

class VisualOdometry:
    def __init__(
        self,
        feature_extractor: InterfaceFeatureExtractor,
        feature_matcher: InterfaceFeatureMatcher,
        model_fitter: InterfaceModelFitter,
        num_points: int = 8,
        epsilon: float = 0.5,
        prob: float = 0.99,
    ) -> None:
        """
        Initialize the Visual Odometry pipeline.

        :param feature_extractor: Feature extractor instance.
        :param feature_matcher: Feature matcher instance.
        :param model_fitter: Model fitter instance.
        :param num_points: Number of points required for RANSAC.
        :param epsilon: RANSAC parameter for outlier rejection.
        :param prob: RANSAC parameter for probability.
        """
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.model_fitter = model_fitter
        self.num_points = num_points
        self.epsilon = epsilon
        self.prob = prob

    def read_frame(
        self, 
        image_path: str
    ) -> np.ndarray:
        """
        Read an image from the file path.

        :param image_path: The image file path.
        :return: The image as a NumPy array.
        """
        return cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (640, 480))

    def extract_and_match_features(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray, 
        display: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Extract and match features between two images.

        :param img1: The first image.
        :param img2: The second image.
        :param display: Whether to display the matches.
        :return: A tuple containing the keypoints, descriptors, and matches.
        """
        kp1, des1 = self.feature_extractor.extract_features(img1)
        kp2, des2 = self.feature_extractor.extract_features(img2)

        matches, matchesMask = self.feature_matcher.match_features(des1, des2)
        if display: self.feature_matcher.show_matches(img1, kp1, img2, kp2, matches, matchesMask)

        # good_matches, matchesMask = self.feature_matcher.filter_matches(matches)
        # pts1, pts2 = self.feature_matcher.extract_keypoints(kp1, kp2, good_matches)

        

        

        return None#pts1, pts2, good_matches
    
    def _display_matches(
        self, 
        img1: np.ndarray, 
        kp1: List[cv2.KeyPoint], 
        img2: np.ndarray, 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch], 
        good_matches: List[cv2.DMatch],
        matchesMask: List[List[int]]
    ) -> None:
        """
        Display the feature matches.

        :param img1: The first image.
        :param kp1: Keypoints in the first image.
        :param img2: The second image.
        :param kp2: Keypoints in the second image.
        :param matches: Matches between the keypoints.
        :param matchesMask: Mask for the matches.
        """
        img_matches = cv2.drawMatchesKnn(
            img1=img1,
            keypoints1=kp1, 
            img2=img2, 
            keypoints2=kp2, 
            matches1to2=matches, 
            outImg=None, 
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.imshow("Matches KNN", img_matches)
        cv2.waitKey(100)

        # img_matches = cv2.drawMatches(
        #     img1=img1,
        #     keypoints1=kp1, 
        #     img2=img2, 
        #     keypoints2=kp2, 
        #     matches1to2=good_matches, 
        #     outImg=None, 
        #     matchColor=(0, 255, 0),
        #     singlePointColor=(0, 0, 255),
        #     flags=cv2.DrawMatchesFlags_DEFAULT)
        # cv2.imshow("Matches", img_matches)
        # cv2.waitKey(100)

    def decompose_essential_matrix(
        self, 
        E: np.ndarray, 
        pts1: np.ndarray, 
        pts2: np.ndarray, 
        intrinsic_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the essential matrix into rotation and translation matrices.

        :param E: The essential matrix.
        :param pts1: The keypoints of the first image.
        :param pts2: The keypoints of the second image.
        :param intrinsic_matrix: The camera intrinsic matrix.
        :return: A tuple containing the rotation matrix and translation matrix.
        """
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, intrinsic_matrix)
        return R, t

    def rescale_translation(
        self, 
        t: np.ndarray, 
        scale: float
    ) -> np.ndarray:
        """
        Rescale the translation vector.

        :param t: The translation vector.
        :param scale: The scale factor.
        :return: The rescaled translation vector.
        """
        return t * scale

    def concatenate_transformation(
        self, C_prev: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate the transformation matrices.

        :param C_prev: The previous transformation matrix.
        :param R: The rotation matrix.
        :param t: The translation matrix.
        :return: The concatenated transformation matrix.
        """
        T = np.hstack((R, t))
        T = np.vstack((T, [0, 0, 0, 1]))
        return C_prev @ T

    def process_frames(
        self,
        img1: str,
        img2: str,
        C_prev: np.ndarray = np.eye(4),
        intrinsic_matrix: Optional[np.ndarray] = None,
        display: bool = False
    ) -> Optional[np.ndarray]:
        """
        Process two frames and estimate the transformation matrix.

        :param img1: The first image file path.
        :param img2: The second image file path.
        :param C_prev: The previous transformation matrix (optional).
        :param intrinsic_matrix: The camera intrinsic matrix (optional).
        :param display: Whether to display the matches.
        :return: The estimated transformation matrix (either F or E).
        """
        
        # step 1 -> Capture new Ik frame
        gray_img1 = self.read_frame(img1)
        gray_img2 = self.read_frame(img2)

        # Step 2 -> Extract and combine features between Ik-1 and Ik
        pts1, pts2, matches = self.extract_and_match_features(gray_img1, gray_img2, display)

        max_iter = self.model_fitter.compute_num_iterations(self.prob, self.epsilon, self.num_points, matches)

        if intrinsic_matrix is not None:
            # Step3 -> Calculate the essential matrix for the image pair I_{k-1}, I_k
            E, inliers1, inliers2 = self.model_fitter.fit_essential_matrix(pts1, pts2, intrinsic_matrix, max_iter)
            if E is None:
                print("No valid essential matrix found, skipping frame.")
                return None

            # Step 4 -> Decompose essential matrix into R_k and t_k , and form T_k
            R, t = self.decompose_essential_matrix(E, pts1, pts2, intrinsic_matrix)

            # Step 5 -> Compute relative scale and rescale t_k accordingly
            t_rescaled = self.rescale_translation(t, scale=0.5)

            # step 6 -> Concatenate transformation by computing C_k = C_{k-1} T_k
            C = self.concatenate_transformation(C_prev, R, t_rescaled)
            return C

        else:
            F, inliers1, inliers2 = self.model_fitter.fit_fundamental_matrix(pts1, pts2, max_iter)
            if F is None:
                print("No valid fundamental matrix found, skipping frame.")
                return None
            return F
