import cv2
import torch

import kornia as K
import kornia.feature as KF
import numpy as np

from .interfaces import InterfaceFeatureExtractor
from kornia.feature import LoFTR
from typing import List, Tuple, Optional


class SIFT(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()
        self.descriptor = self.detector

    def _define_parameters(self) -> None:
        """
        Define the parameters for the SIFT feature detector.
        """
        self.nfeatures = 0
        self.nOctaveLayers = 3
        self.contrastThreshold = 0.04
        self.edgeThreshold = 10
        self.sigma = 1.6

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.nOctaveLayers,
            contrastThreshold=self.contrastThreshold,
            edgeThreshold=self.edgeThreshold,
            sigma=self.sigma,
        )
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.detector.compute(image, keypoints)
        return keypoints, descriptors


class ORB(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()
        self.descriptor = self.detector

    def _define_parameters(self) -> None:
        """
        Define the parameters for the ORB feature detector.
        """
        self.nfeatures = 500
        self.scaleFactor = 1.2
        self.nlevels = 8
        self.edgeThreshold = 31
        self.firstLevel = 0
        self.WTA_K = 2
        self.scoreType = cv2.ORB_HARRIS_SCORE
        self.patchSize = 31
        self.fastThreshold = 20

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=self.scaleFactor,
            nlevels=self.nlevels,
            edgeThreshold=self.edgeThreshold,
            firstLevel=self.firstLevel,
            WTA_K=self.WTA_K,
            scoreType=self.scoreType,
            patchSize=self.patchSize,
            fastThreshold=self.fastThreshold,
        )
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.detector.compute(image, keypoints)
        return keypoints, descriptors


class AKAZE(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()

    def _define_parameters(self) -> None:
        """
        Define the parameters for the AKAZE feature detector.
        """
        self.descriptor_type = cv2.AKAZE_DESCRIPTOR_MLDB
        self.descriptor_size = 0
        self.descriptor_channels = 3
        self.threshold = 0.001
        self.nOctaves = 4
        self.nOctaveLayers = 4
        self.diffusivity = cv2.KAZE_DIFF_PM_G2

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.AKAZE_create(
            descriptor_type=self.descriptor_type,
            descriptor_size=self.descriptor_size,
            descriptor_channels=self.descriptor_channels,
            threshold=self.threshold,
            nOctaves=self.nOctaves,
            nOctaveLayers=self.nOctaveLayers,
            diffusivity=self.diffusivity,
        )
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detectAndCompute(image, None)
        return keypoints


class FAST(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()
        self.descriptor = cv2.ORB_create()

    def _define_parameters(self) -> None:
        """
        Define the parameters for the FAST feature detector.
        """
        self.suppression = True
        self.threshold = 10

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.FastFeatureDetector_create()
        detector.setNonmaxSuppression(self.suppression)
        detector.setThreshold(self.threshold)
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors


class BRISK(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()
        self.descriptor = self.detector

    def _define_parameters(self) -> None:
        """
        Define the parameters for the BRISK feature detector.
        """
        self.thresh = 30
        self.octaves = 3
        self.patternScale = 1.0

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.BRISK_create(
            thresh=self.thresh, octaves=self.octaves, patternScale=self.patternScale
        )
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.detector.compute(image, keypoints)
        return keypoints, descriptors


class KAZE(InterfaceFeatureExtractor):
    def __init__(self, parameters: Optional[dict] = None) -> None:
        """
        Initialize the feature extractor.

        :param parameters: Parameters for the feature extraction method.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.detector = self._initialize_extractor()
        self.descriptor = self.detector

    def _define_parameters(self) -> None:
        """
        Define the parameters for the KAZE feature detector.
        """
        self.extended = False
        self.upright = False
        self.threshold = 0.001
        self.nOctaves = 4
        self.nOctaveLayers = 4
        self.diffusivity = cv2.KAZE_DIFF_PM_G2

    def _initialize_extractor(self) -> cv2.Feature2D:
        """
        Initialize the feature detector.

        :return: Detector.
        """
        detector = cv2.KAZE_create(
            extended=self.extended,
            upright=self.upright,
            threshold=self.threshold,
            nOctaves=self.nOctaves,
            nOctaveLayers=self.nOctaveLayers,
            diffusivity=self.diffusivity,
        )
        return detector

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.detector.compute(image, keypoints)
        return keypoints, descriptors


class LoFTRExtractor(InterfaceFeatureExtractor):
    def __init__(self, parameters: dict = None) -> None:
        """
        Inicializa o matcher LoFTR.
        """
        self.parameters = parameters or {}
        self._define_parameters()
        self.matcher = self._initialize_extractor()

    def _define_parameters(self) -> None:
        """
        Define os parâmetros para o matcher LoFTR.
        """
        self.model_type = self.parameters.get("model_type", "indoor")  # 'indoor' ou 'outdoor'
        self.gpu = self.parameters.get("gpu", False)

    def _initialize_extractor(self) -> LoFTR:
        """
        Inicializa o matcher LoFTR.
        """
        matcher = LoFTR(pretrained=self.model_type)
        if self.gpu: matcher = matcher.to(torch.device("cuda"))
        return matcher

    def extract_features(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        img1_gray = K.color.rgb_to_grayscale(img1)
        img2_gray = K.color.rgb_to_grayscale(img2)

        input_dict = {
            "image0": img1_gray,
            "image1": img2_gray
        }
        if self.gpu: input_dict = {k: v.to(torch.device("cuda")) for k, v in input_dict.items()}

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
        
        return correspondences