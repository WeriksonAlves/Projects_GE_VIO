import cv2
import numpy as np
from typing import Tuple, List, Optional, Union


class FeatureExtractor:
    def __init__(self, method: str, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param method: Feature extraction method, either "SIFT", "ORB", "AKAZE" 
        or "FAST".
        :param parameters: Parameters for the feature extraction method.
        """
        self.method = method
        self.parameters = parameters or {}
        self.detector, self.descriptor = self._initialize_detector()

    def _initialize_detector(self) -> Tuple[cv2.Feature2D, cv2.Feature2D]:
        """
        Initialize the feature detector and descriptor.

        :return: Detector and descriptor.
        """
        if self.method == "SIFT":
            detector = cv2.SIFT_create()
            descriptor = detector
        elif self.method == "ORB":
            detector = cv2.ORB_create()
            descriptor = detector
        elif self.method == "AKAZE":
            detector = cv2.AKAZE_create()
            descriptor = detector  # AKAZE combines detector and descriptor
        elif self.method == "FAST":
            detector = cv2.FastFeatureDetector_create()
            detector.setNonmaxSuppression(self.parameters.get("suppression", True))
            detector.setThreshold(self.parameters.get("threshold", 10))
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
    def __init__(self, method: str, matches: str, parameters_method: Optional[dict] = None, parameters_matches: dict = {"k": 2}) -> None:
        """
        Initialize the feature matcher.

        :param method: Feature matching method, either "BF" or "FLANN".
        :param matches: Matching method, either "DEFAULT" or "KNN".
        :param parameters_method: Parameters for the feature matching method.
        :param parameters_matches: Parameters for the matching method.
        """
        self.method = method
        self.matches = matches
        self.parameters1 = parameters_method or {}
        self.parameters2 = parameters_matches or {}
        self.matcher = self._initialize_matcher()

    def _initialize_matcher(self) -> cv2.DescriptorMatcher:
        """
        Initialize the feature matcher.

        :return: Matcher object.
        """
        if self.method == "BF":
            return cv2.BFMatcher(self.parameters1.get("norm_type", cv2.NORM_HAMMING),
                                self.parameters1.get("crossCheck", True))
        elif self.method == "FLANN":
            return cv2.FlannBasedMatcher(self.parameters1["index_params"],
                                        self.parameters1["search_params"])
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _change_type(self, des1: np.ndarray, des2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Change the type of the descriptors to float32 if needed for FLANN matcher.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: Descriptors with the correct type.
        """
        if self.method == "FLANN":
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
        elif self.method == "BF":
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)
        return des1, des2
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two sets.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of matches.
        """
        des1, des2 = self._change_type(des1, des2)

        if self.matches == "DEFAULT":
            matches = self.matcher.match(des1, des2)
            return sorted(matches, key=lambda x: x.distance)
        elif self.matches == "KNN":
            matches = self.matcher.knnMatch(des1, des2, k=self.parameters2.get("k", 2))
            return self._filter_matches(matches)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _filter_matches(self, matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
        """
        Filter matches using the ratio test.

        :param matches: Raw matches from knnMatch.
        :return: Filtered list of matches.
        """
        # matchesMask = [[0, 0] for i in range(len(matches))]
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.7 * n.distance:
        #         matchesMask[i] = [1, 0]
        
        
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def extract_keypoints(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
            matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Extract keypoints and create DMatch objects.

        :param kp1: Keypoints from the first image.
        :param kp2: Keypoints from the second image.
        :param matches: List of matches.
        :return: Keypoint coordinates and good matches.
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2, matches


class ModelFitter:
    def __init__(self, prob: float = 0.999, reproj_thresh: float = 0.4) -> None:
        """
        Initialize the model fitter with RANSAC parameters.

        :param prob: Probability of success for RANSAC.
        :param reproj_thresh: Reprojection error threshold for RANSAC.
        """
        self.prob = prob
        self.reproj_thresh = reproj_thresh

    def fit_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray, max_iter: int) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the fundamental matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :param max_iter: Maximum number of iterations for RANSAC.
        :return: Fundamental matrix, inliers from pts1, inliers from pts2.
        """
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, self.reproj_thresh, maxIters=max_iter)
        if F is None or mask is None:
            print("Warning: No valid fundamental matrix found.")
            return None, np.array([]), np.array([])

        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        return F, inliers1, inliers2

    def fit_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, max_iter: int) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the essential matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :param K: Camera intrinsic matrix.
        :param max_iter: Maximum number of iterations for RANSAC.
        :return: Essential matrix, inliers from pts1, inliers from pts2.
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=self.prob, threshold=self.reproj_thresh, maxIters=max_iter)
        if E is None or mask is None:
            print("Warning: No valid essential matrix found.")
            return None, np.array([]), np.array([])

        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        return E, inliers1, inliers2

    def compute_num_iterations(self, prob: float, epsilon: float, num_points: int, matches) -> int:
        """
        Compute the number of iterations for RANSAC.

        :param prob: Probability of success.
        :param epsilon: Inlier ratio.
        :param num_points: Number of points required for RANSAC.
        :param matches: Number of matches.
        :return: The number of iterations.
        """
        if len(matches) < num_points: 
            num_points = len(matches)
        max_iter = int(np.log(1 - prob) / np.log(1 - (1 - epsilon) ** num_points))
        return max_iter


class VisualOdometry:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        feature_matcher: FeatureMatcher,
        model_fitter: ModelFitter,
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

        matches = self.feature_matcher.match_features(des1, des2)
        pts1, pts2, good_matches = self.feature_matcher.extract_keypoints(kp1, kp2, matches)

        if display:
            self._display_matches(img1, kp1, img2, kp2, good_matches)

        return pts1, pts2, good_matches
    
    def _display_matches(
        self, 
        img1: np.ndarray, 
        kp1: List[cv2.KeyPoint], 
        img2: np.ndarray, 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> None:
        """
        Display the feature matches.

        :param img1: The first image.
        :param kp1: Keypoints in the first image.
        :param img2: The second image.
        :param kp2: Keypoints in the second image.
        :param matches: Matches between the keypoints.
        """
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(100)

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
