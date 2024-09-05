from abc import ABC, abstractmethod

class PoseEstimator(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def _define_parameters(self, *args, **kwargs):
        pass

    @abstractmethod
    def _initialize_pose_estimator(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract_features(self, *args, **kwargs):
        pass

    @abstractmethod
    def match_features(self, *args, **kwargs):
        pass

    @abstractmethod
    def rescale_translation(self, *args, **kwargs):
        pass

    @abstractmethod
    def transformation_matrix(self, *args, **kwargs):
        pass