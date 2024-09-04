import cv2
import numpy as np
from typing import Tuple, List, Optional
from .interfaces import InterfaceFeatureMatcher


class BF(InterfaceFeatureMatcher):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature matcher.

        :param parameters: Parameters for the feature matching method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.matcher = self._initialize_matcher()

    def _define_parameters(self) -> None:
        """
        Define the parameters for the BF feature matcher.
        """
        self.norm_type = self.parameters.get("norm_type", cv2.NORM_HAMMING)
        self.crossCheck = self.parameters.get("crossCheck", True)

    def _initialize_matcher(self) -> cv2.DescriptorMatcher:
        """
        Initialize the feature matcher.

        :return: Matcher object.
        """
        return cv2.BFMatcher(self.norm_type, self.crossCheck)

    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two sets.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of matches.
        """
        matches = self.matcher.match(des1.astype(np.uint8), des2.astype(np.uint8))
        return sorted(matches, key=lambda x: x.distance), None

    def show_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint], img2: np.ndarray, kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], matchesMask = None) -> None:
        """
        Show the matches between two images.

        :param img1: First image.
        :param kp1: Keypoints from the first image.
        :param img2: Second image.
        :param kp2: Keypoints from the second image.
        :param matches: List of matches.
        """
        img_matches = cv2.drawMatches(img1=img1, keypoints1=kp1, img2=img2, keypoints2=kp2, matches1to2=matches, outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(100)


class FLANN(InterfaceFeatureMatcher):
    def __init__(self, parameters: dict) -> None:
        """
        Initialize the feature matcher.

        :param parameters: Parameters for the feature matching method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.matcher = self._initialize_matcher()

    def _define_parameters(self) -> None:
        """
        Define the parameters for the FLANN feature matcher.
        """
        self.index_params = self.parameters["index_params"]
        self.search_params = self.parameters["search_params"]

    def _initialize_matcher(self) -> cv2.DescriptorMatcher:
        """
        Initialize the feature matcher.

        :return: Matcher object.
        """
        return cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def _filter_matches(self, matches: List[List[cv2.DMatch]], k: float = 0.7) -> Tuple[List[cv2.DMatch], List[List[int]]]:
        """
        Filter matches based on the Lowe's ratio test.
        
        :param matches: List of matches.
        :param k: Threshold for the ratio test.
        :return: List of good matches and mask.
        """
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < k * n.distance:
                matchesMask[i] = [1, 0]
        return matchesMask
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two sets.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of matches.
        """
        matches = self.matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        matchesMask = self._filter_matches(matches)
        return matches, matchesMask
    
    def show_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint], img2: np.ndarray, kp2: List[cv2.KeyPoint], matches: List[List[cv2.DMatch]], matchesMask: List[List[int]]) -> None:
        """
        Show the matches between two images.

        :param img1: First image.
        :param kp1: Keypoints from the first image.
        :param img2: Second image.
        :param kp2: Keypoints from the second image.
        :param matches: List of matches.
        :param matchesMask: Mask for the matches.
        """
        img_matches = cv2.drawMatchesKnn(img1=img1, keypoints1=kp1, img2=img2, keypoints2=kp2, matches1to2=matches, outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.imshow("Matches KNN", img_matches)
        cv2.waitKey(100)