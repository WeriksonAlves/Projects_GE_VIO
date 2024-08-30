from .interfaces import FeatureMatchingInterface
import numpy as np
import cv2

class FeatureMatching(FeatureMatchingInterface):
    def __init__(self, model) -> None:
        """
        Initializes a FeatureMatching object.

        Args:
            model: The model used for feature detection.

        Returns:
            None
        """
        self.model = model # Create the detector object

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

    def drawKeyPoints(self, image: np.ndarray, keypoints: list, imageName: str, outImage: np.ndarray = None ) -> None:
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

class FastFeatureMatching(FeatureMatchingInterface):
    def __init__(self, model: cv2.FastFeatureDetector, aux_model, suppression: bool = True, threshold: int = 10) -> None:
        """
        Initializes a FeatureMatching object.

        Args:
            model (cv2.FastFeatureDetector): The feature detector model.
            suppression (bool, optional): Flag indicating whether to use non-maximum suppression. Default is True.
            threshold (int, optional): The threshold for detection. Default is 10.
        
        Returns:
            None
        """
        self.model = model # Create the detector object
        # Optionally, you can set some parameters
        self.model.setNonmaxSuppression(suppression)  # Use non-maximum suppression
        self.model.setThreshold(threshold)  # Threshold for detection
        self.aux_model = aux_model

    def my_detectAndCompute(self, gray_image: np.ndarray, mask: np.ndarray = None) -> list:
        """
        Detects keypoints and computes descriptors using the specified detector on the given gray image.

        Args:
            gray_image (np.ndarray): The input gray image on which keypoints and descriptors are computed.
            mask (np.ndarray, optional): An optional mask specifying where to detect keypoints. Defaults to None.

        Returns:
            list: A list containing the detected keypoints.
        """
        keypoints = self.model.detect(gray_image, mask)
        keypoints, descriptors = self.aux_model.compute(gray_image, keypoints)
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
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage, color=(255,0,0))

        # Display the image with keypoints
        cv2.imshow(imageName, image_with_keypoints)
        cv2.waitKey(100)

    def saveKeypoints(self, filename: str, img_keypoints: np.ndarray) -> None:
        """
        Save the image keypoints to a file.

        Args:
            filename (str): The name of the file to save the keypoints.
            img_keypoints (np.ndarray): The image keypoints to be saved.

        Returns:
            None
        """
        cv2.imwrite(filename, img_keypoints)

    def matchingKeypoints(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches based on the distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
        