from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np


class InterfaceFeatureExtractor(ABC):
    @abstractmethod
    def __init__(self, parameters: dict) -> None:
        pass

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
    def __init__(self, parameters: dict) -> None:
        pass

    @abstractmethod
    def _define_parameters(self) -> None:
        pass

    @abstractmethod
    def _initialize_matcher(self):
        pass
    
    @abstractmethod
    def match_features(*args, **kwargs):
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


