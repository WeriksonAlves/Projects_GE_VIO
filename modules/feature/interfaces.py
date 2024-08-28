from abc import ABC, abstractmethod
import numpy as np

class FeatureMatchingInterface(ABC):
    """
    Abstract base class for tracking processors.
    """
    @abstractmethod
    def my_detectAndCompute(self, detector, frame: np.ndarray) -> None:
        """
        Perform feature detection and computation on the given frame using the specified detector.
        Args:
            detector: The feature detector to use for detection and computation.
            frame: The frame on which to perform feature detection and computation.
        Returns:
            None
        """
        
        pass