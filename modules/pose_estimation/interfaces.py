from abc import ABC, abstractmethod

class PoseEstimator(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def _define_parameters(self, *args, **kwargs):
        """
        Define the parameters for the Visual Odometry pipeline.
        """
        pass

    @abstractmethod
    def _initialize_pose_estimator(self, *args, **kwargs):
        """
        Initialize the pose estimator.
        """
        pass

    @abstractmethod
    def extract_and_match_features(self, *args, **kwargs):
        """
        Extract and match features between two images.
        
        :param img1: The first image tensor.
        :param img2: The second image tensor.
        :param display: Whether to display the matches.
        :return: A tuple containing the keypoints, descriptors, and matches.
        """
        pass

    @abstractmethod
    def extract_keypoints(self, *args, **kwargs):
        """
        Extract keypoints from the correspondences.

        :param correspondences: The feature correspondences.
        :return: A tuple containing the keypoints and descriptors.
        """
        pass

    @abstractmethod
    def rescale_translation(self, *args, **kwargs):
        pass

    @abstractmethod
    def essential_homogeneous_matrix(self, *args, **kwargs):
        pass