from abc import ABC, abstractmethod
import numpy as np

class FeatureMatchingInterface(ABC):
    """
    Abstract base class for tracking processors.
    """
    @abstractmethod
    def my_detectAndCompute(self, *args):
        """
        Detects keypoints and computes descriptors using the specified detector on the given gray image.
        """
        pass
    
    @abstractmethod
    def drawKeyPoints(self, *args):
        """
        Draws keypoints on the given image and displays the image with keypoints.
        
        Args:
            image (np.ndarray): The input image on which keypoints will be drawn.
            keypoints (list): List of keypoints to be drawn on the image.
            outImage (np.ndarray, optional): Output image where the keypoints will be drawn. If not provided, a new image will be created.
            imagename (str, optional): Name of the window to display the image with keypoints. Default is 'SIFT Keypoints'.
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def saveKeypoints(self, *args):
        """
        Save keypoints and descriptors to a file.

        Args:
            filename (str): The name of the file to save the keypoints and descriptors.
            keypoints: The keypoints to be saved.
            descriptors: The descriptors to be saved.

        Returns:
            None
        """
        pass

    @abstractmethod
    def matchingKeypoints(self, *args):
        """
        Matches keypoints between two sets of descriptors.
        """
        pass