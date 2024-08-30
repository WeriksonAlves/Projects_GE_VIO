
import cv2
import numpy as np

from typing import Tuple, List, Optional, Union

class FeatureExtractor:
    def __init__(self, method: str = "AKAZE", parammeters: Union[dict,None] = None)-> None:
        """
        Initialize the feature extractor.

        :param method: Feature extraction method, either "AKAZE" or "FAST".
        """
        self.method = method
        self.parammeters = parammeters
        self.detector, self.descriptor = self._initialize_detector()

    def _initialize_detector(self) -> Tuple[cv2.Feature2D, cv2.Feature2D]:
        if self.method == "AKAZE":
            detector = cv2.AKAZE_create()
            descriptor = detector  # AKAZE combines detector and descriptor
        elif self.method == "FAST":
            detector = cv2.FastFeatureDetector_create()
            detector.setNonmaxSuppression(self.parammeters['suppression'])
            detector.setThreshold(self.parammeters['threshold'])
            descriptor = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        return detector, descriptor

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors


class FeatureMatcher:
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check: bool = True):
        """
        Initialize the feature matcher.

        :param norm_type: Norm type for matching, default is NORM_HAMMING.
        :param cross_check: If true, use cross-checking in matching.
        """
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)

    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two sets.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of matches.
        """
        matches = self.matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)


class ModelFitter:
    def __init__(self, reproj_thresh: float = 3.0, max_iter: int = 1000):
        """
        Initialize the model fitter with RANSAC parameters.

        Args:
            reproj_thresh (float): The maximum allowed reprojection error threshold.
            max_iter (int): The maximum number of iterations for RANSAC.

        Returns:
            None
        """
        self.prob = 0.999
        self.reproj_thresh = reproj_thresh
        self.max_iter = max_iter

    def fit_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray, max_iter) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the fundamental matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :return: Fundamental matrix, inliers from pts1, inliers from pts2.
        """
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, self.reproj_thresh, max_iter)
        
        # Check if a valid fundamental matrix was found
        if F is None or mask is None:
            print("Warning: No valid fundamental matrix found.")
            return None, np.array([]), np.array([])

        # Select inliers
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        return F, inliers1, inliers2

    def fit_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, max_iter) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the essential matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :param K: Camera intrinsic matrix.
        :return: Essential matrix, inliers from pts1, inliers from pts2.
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=self.prob, threshold=self.reproj_thresh, maxIters=max_iter)
        
        # Check if a valid essential matrix was found
        if E is None or mask is None:
            print("Warning: No valid essential matrix found.")
            return None, np.array([]), np.array([])

        # Select inliers
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        return E, inliers1, inliers2


class VisualOdometry:
    def __init__(self, feature_extractor: FeatureExtractor, feature_matcher: FeatureMatcher, model_fitter: ModelFitter):
        """
        Initialize the Visual Odometry pipeline.

        :param feature_extractor: Instance of FeatureExtractor.
        :param feature_matcher: Instance of FeatureMatcher.
        :param model_fitter: Instance of ModelFitter.
        """
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.model_fitter = model_fitter

    def process_frames(self, img1: np.ndarray, img2: np.ndarray, intrinsic_matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Process two consecutive frames and estimate the motion.

        :param img1: First image (previous frame).
        :param img2: Second image (current frame).
        :param intrinsic_matrix: Optional camera intrinsic matrix.
        :return: Estimated transformation matrix (Fundamental or Essential matrix).
        """
        kp1, des1 = self.feature_extractor.extract_features(img1)
        kp2, des2 = self.feature_extractor.extract_features(img2)

        matches = self.feature_matcher.match_features(des1, des2)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        if intrinsic_matrix is not None:
            E, inliers1, inliers2 = self.model_fitter.fit_essential_matrix(pts1, pts2, intrinsic_matrix, max_iter=1000)
            if E is None:
                print("No valid essential matrix found, skipping frame.")
                return None
            return E
        else:
            F, inliers1, inliers2 = self.model_fitter.fit_fundamental_matrix(pts1, pts2, max_iter=1000)
            if F is None:
                print("No valid fundamental matrix found, skipping frame.")
                return None
            return F
