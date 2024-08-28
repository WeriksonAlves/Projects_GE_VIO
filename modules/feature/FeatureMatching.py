from .interfaces import FeatureMatchingInterface
import numpy as np
import cv2

class MySIFT(FeatureMatchingInterface):
    def __init__(self):
        """
        Initializes the FeatureMatching object.
        """
        self.model = cv2.SIFT_create() # Create a SIFT detector object

    def my_detectAndCompute(self, gray_image: np.ndarray, mask: np.ndarray = None) -> tuple:
        """
        Detects keypoints and computes descriptors using the specified detector on the given gray image.

        Args:
            gray_image (np.ndarray): The input gray image on which keypoints and descriptors are computed.
            mask (np.ndarray, optional): An optional mask specifying where to detect keypoints. Defaults to None.

        Returns:
            tuple: A tuple containing the detected keypoints and computed descriptors.
        """
        keypoints, descriptors = self.model.detectAndCompute(gray_image, mask)
        return keypoints, descriptors

    def drawKeyPoints(self, image: np.ndarray, keypoints: list, outImage: np.ndarray = None, imageName: str = 'SIFT Keypoints' ) -> None:
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
        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the image with keypoints
        cv2.imshow(imageName, image_with_keypoints)

    def saveKeypoints(self, filename: str, keypoints, descriptors) -> None:
        """
        Save keypoints and descriptors to a file.

        Args:
            filename (str): The name of the file to save the keypoints and descriptors.
            keypoints: The keypoints to be saved.
            descriptors: The descriptors to be saved.

        Returns:
            None
        """
        np.savez(filename, keypoints=keypoints, descriptors=descriptors)

    def matchingKeypoints(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        """
        Matches keypoints based on their descriptors using the Brute-Force Matcher.

        Args:
            descriptors1 (np.ndarray): Descriptors of keypoints from the first image.
            descriptors2 (np.ndarray): Descriptors of keypoints from the second image.

        Returns:
            list: List of matches sorted by distance.
        """

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

class MySURF(FeatureMatchingInterface):
    def __init__(self, threshold: int = 400, orientation: bool = False, extended: bool = False):
        """
        Initializes the FeatureMatching object.
        Parameters:
            threshold (int): The Hessian threshold for SURF feature detection. Default is 400.
            orientation (bool): Flag to compute orientation of keypoints. Default is False.
            extended (bool): Flag to get 128-dim descriptors instead of 64-dim. Default is False (64-dim).
        """
        
        self.model = cv2.xfeatures2d.SURF_create(hessianThreshold=threshold)
        self.model.setUpright(orientation)  # To not compute orientation
        self.model.setExtended(extended)  # To get 128-dim descriptors instead of 64-dim

    def my_detectAndCompute(self, gray_image: np.ndarray, mask: np.ndarray = None) -> tuple:
        """
        Detects keypoints and computes descriptors for a given gray image.

        Args:
            gray_image (np.ndarray): The gray image to detect keypoints and compute descriptors on.
            mask (np.ndarray, optional): The optional mask to apply on the image. Defaults to None.

        Returns:
            tuple: A tuple containing the keypoints and descriptors.

        """
        keypoints, descriptors = self.model.detectAndCompute(gray_image, mask)
        return keypoints, descriptors

    def drawKeyPoints(self, image: np.ndarray, keypoints: list, outImage: np.ndarray = None, imageName: str = 'SURF Keypoints' ) -> np.ndarray:
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
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(imageName, image_with_keypoints)

    def saveKeypoints(self, filename: str, keypoints, descriptors) -> None:
        """
        Save keypoints and descriptors to a file.

        Args:
            filename (str): The name of the file to save the keypoints and descriptors.
            keypoints: The keypoints to be saved.
            descriptors: The descriptors to be saved.

        Returns:
            None
        """
        np.savez(filename, keypoints=keypoints, descriptors=descriptors)

    def matchingKeypoints(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
    
class MyORB(FeatureMatchingInterface):
    def __init__(self, nfeatures: int = 500, scaleFactor: float = 1.2, nlevels: int = 8, edgeThreshold: int = 31, firstLevel: int = 0, WTA_K: int = 2, scoreType: int = cv2.ORB_HARRIS_SCORE, patchSize: int = 31, fastThreshold: int = 20):
        """
        Initializes the FeatureMatching object.
        Parameters:
            nfeatures (int): The maximum number of features to retain. Default is 500.
            scaleFactor (float): Pyramid decimation ratio. Default is 1.2.
            nlevels (int): The number of pyramid levels. Default is 8.
            edgeThreshold (int): The size of the border where the features are not detected. Default is 31.
            firstLevel (int): The level of pyramid to put source image to. Default is 0.
            WTA_K (int): The number of points that produce each element of the oriented BRIEF descriptor. Default is 2.
            scoreType (int): The HARRIS_SCORE or FAST_SCORE. Default is cv2.ORB_HARRIS_SCORE.
            patchSize (int): The size of the patch used by the oriented BRIEF descriptor. Default is 31.
            fastThreshold (int): The fast threshold. Default is 20.
        """
        self.model = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, edgeThreshold=edgeThreshold, firstLevel=firstLevel, WTA_K=WTA_K, scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold)

    def my_detectAndCompute(self, gray_image: np.ndarray, mask: np.ndarray = None) -> tuple:
        """
        Detects keypoints and computes descriptors for a given gray image.

        Args:
            gray_image (np.ndarray): The gray image to detect keypoints and compute descriptors on.
            mask (np.ndarray, optional): The optional mask to apply on the image. Defaults to None.

        Returns:
            tuple: A tuple containing the keypoints and descriptors.

        """
        keypoints, descriptors = self.model.detectAndCompute(gray_image, mask)
        return keypoints, descriptors
