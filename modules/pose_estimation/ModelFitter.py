import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, Sequence


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
