from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np


class InterfaceFeatureExtractor(ABC):
    @abstractmethod
    def _define_parameters(self, parameters: dict) -> None:
        pass

    @abstractmethod
    def _initialize_extractor(self) -> cv2.Feature2D:
        pass

    @abstractmethod
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pass


class InterfaceFeatureMatcher(ABC):
    @abstractmethod
    def match_features(
        self,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        descriptors2: np.ndarray,
    ) -> List[cv2.DMatch]:
        pass

    @abstractmethod
    def show_matches(*args, **kwargs) -> None:
        pass


class InterfaceModelFitter(ABC):
    @abstractmethod
    def fit_model(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        intrinsic_parameters: np.ndarray,
        object_pattern: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


