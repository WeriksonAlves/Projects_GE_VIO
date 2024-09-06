import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, Sequence

from ..feature.interfaces import InterfaceFeatureExtractor, InterfaceFeatureMatcher, InterfaceModelFitter
from .interfaces import PoseEstimator


class VisualOdometry(PoseEstimator):
    def __init__(
        self,
        feature_extractor: InterfaceFeatureExtractor,
        feature_matcher: InterfaceFeatureMatcher,
        model_fitter: InterfaceModelFitter,
        params: Optional[dict] = None
    ) -> None:
        """
        Initialize the Visual Odometry pipeline.

        :param feature_extractor: Feature extractor instance.
        :param feature_matcher: Feature matcher instance.
        :param model_fitter: Model fitter instance.
        :param params: Optional parameters for the pipeline.
        """
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.model_fitter = model_fitter
        self.params = params or {}

        self._define_parameters()
    
    def _define_parameters(self) -> None:
        """
        Define the parameters for the Visual Odometry pipeline.
        """
        self.num_points = self.params.get("num_points", 8)
        self.epsilon = self.params.get("epsilon", 0.5)
        self.prob = self.params.get("prob", 0.99)
        self.display = self.params.get("display", False)

    def _initialize_pose_estimator(self, *args, **kwargs):
        pass

    def extract_and_match_features(
        self, img1_tensor: np.ndarray, img2_tensor: np.ndarray
    ):
        """
        Extract and match features between two images.
        
        :param img1_tensor: The first image tensor.
        :param img2_tensor: The second image tensor.
        
        """
        correspondences, _ = self.feature_matcher.match_features(
            self.feature_extractor.extract_features(img1_tensor, img2_tensor)
        )
        return correspondences

    def extract_keypoints(self, correspondences):
        """
        Extract keypoints from the correspondences.

        :param correspondences: The feature correspondences.
        :return: The keypoints from the correspondences
        """
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        return mkpts0, mkpts1

    def compute_num_iterations(self, prob, epsilon, num_points, matches):
        """
        Compute the number of iterations for the RANSAC algorithm.

        :param prob: The probability of success.
        :param epsilon: The outlier ratio.
        :param num_points: The number of points.
        :param matches: The number of matches.
        :return: The number of iterations.
        """
        if len(matches) < num_points: 
            num_points = len(matches)
        max_iter = int(np.log(1 - prob) / np.log(1 - (1 - epsilon) ** num_points))
        return max_iter

    def find_essential_matrix(self, mkpts0, mkpts1, intrinsic_matrix, max_iter):
        """
        Find the essential matrix between two images.
        
        :param mkpts0: The keypoints of the first image.
        :param mkpts1: The keypoints of the second image.
        :param intrinsic_matrix: The camera intrinsic matrix.
        :param max_iter: The maximum number of iterations.
        :return: The essential matrix and the inliers.
        """
        Em, inliers = cv2.findEssentialMat(
            points1=mkpts0,
            points2=mkpts1,
            cameraMatrix=intrinsic_matrix,
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.epsilon,
            maxIters=100000
        )
        inliers = inliers > 0
        return Em, inliers

    def find_fundamental_matrix(self, mkpts1, mkpts2, max_iter):
        """
        Find the fundamental matrix between two images.
        
        :param mkpts1: The keypoints of the first image.
        :param mkpts2: The keypoints of the second image.
        :param max_iter: The maximum number of iterations.
        :return: The fundamental matrix and the inliers.
        """
        Fm, inliers = cv2.findFundamentalMat(
            points1=mkpts1,
            points2=mkpts2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.epsilon,
            confidence=self.prob,
            maxIters=100000
        )
        inliers = inliers > 0
        return Fm, inliers

    def decompose_essential_matrix(self, Em, mkpts0, mkpts1, intrinsic_matrix):
        """
        Recover the pose from the essential matrix.

        :param Em: The essential matrix.
        :param mkpts0: The keypoints of the first image.
        :param mkpts1: The keypoints of the second image.
        :param intrinsic_matrix: The camera intrinsic matrix.
        :return: The rotation matrix and translation vector.
        """
        _, R, t, _ = cv2.recoverPose(Em, mkpts0, mkpts1, intrinsic_matrix)
        return R, t

    def decompose_fundamental_matrix(self, pts1, pts2, percent):
        """
        Recover the pose from the fundamental matrix.


        """
        diff = pts2 - pts1
        len_pts = len(diff)
        idx = np.random.choice(np.arange(len_pts), int(percent * len_pts))
        t = diff.mean(axis=0)
        return t
    
    def rescale_translation(self, t: np.ndarray, scale: float) -> np.ndarray:
        """
        Rescale the translation vector.

        :param t: The translation vector.
        :param scale: The scale factor.
        :return: The rescaled translation vector.
        """
        return t * scale

    def essential_homogeneous_matrix(
        self, H_previous: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Compute the current homogeneous matrix.

        :param H_previous: The previous homogeneous matrix.
        :param R: The rotation matrix.
        :param t: The translation matrix.
        :return: The current homogeneous matrix.
        """
        H_aux = np.hstack((R, t))
        H = np.vstack((H_aux, [0, 0, 0, 1]))
        return H_previous @ H
    
    def fundamental_homogeneous_matrix(
        self, H_previous: np.ndarray, Fm: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Compute the current homogeneous matrix.

        :param H_previous: The previous homogeneous matrix.
        :param R: The rotation matrix.
        :param t: The translation matrix.
        :return: The current homogeneous matrix.
        """
        t = np.hstack((t,np.array([1])))
        H_aux = np.hstack((Fm, np.expand_dims(t, axis=1)))
        H = np.vstack((H_aux, [0, 0, 0, 1]))
        return H_previous @ H

    def process_frames(
        self,
        img1: str,
        img2: str,
        intrinsic_matrix: Optional[np.ndarray] = None,
        H_previous: np.ndarray = np.eye(4),
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
        # Step 2 -> Extract and combine features between Ik-1 and Ik
        correspondences = self.extract_and_match_features(img1, img2)

        mkpts1, mkpts2 = self.extract_keypoints(correspondences)

        max_iter = self.compute_num_iterations(self.prob, self.epsilon, self.num_points, correspondences)

        if intrinsic_matrix is not None:
            # Step3 -> Calculate the essential matrix for the image pair I_{k-1}, I_k
            Em, inliers = self.find_essential_matrix(mkpts1, mkpts2, intrinsic_matrix, max_iter)
            if Em is None:
                print("No valid essential matrix found, skipping frame.")
                return None
            
            # Step 4 -> Decompose essential matrix into R_k and t_k , and form T_k
            R, t = self.decompose_essential_matrix(Em, mkpts1, mkpts2, intrinsic_matrix)

            # Step 5 -> Compute relative scale and rescale t_k accordingly
            t_rescaled = self.rescale_translation(t, scale=0.42*100)

            # step 6 -> Concatenate transformation by computing C_k = C_{k-1} T_k
            current_pose = self.essential_homogeneous_matrix(H_previous, R, t_rescaled)
        
        else:
            print("No intrinsic matrix found.")
            # Step3 -> Calculate the fundamental matrix for the image pair I_{k-1}, I_k
            Fm, inliers = self.find_fundamental_matrix(mkpts1, mkpts2, max_iter)
            if Fm is None:
                print("No valid fundamental matrix found, skipping frame.")
                return None
            
            # Step 4 -> Decompose fundamental matrix into R_k and t_k , and form T_k
            t = self.decompose_fundamental_matrix(
                mkpts1[inliers.squeeze()], mkpts2[inliers.squeeze()], 0.05
            )

            # Step 5 -> Compute relative scale and rescale t_k accordingly
            t_rescaled = self.rescale_translation(t, scale=0.42735042735042733)

            # step 6 -> Concatenate transformation by computing C_k = C_{k-1} T_k
            current_pose = self.fundamental_homogeneous_matrix(H_previous, Fm, t_rescaled)
                    
        
        if self.display: 
            self.feature_matcher.show_matches(img1, img2, mkpts1, mkpts2, inliers)
        
        return current_pose

