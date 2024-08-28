from .interfaces import FeatureMatchingInterface
import numpy as np
import cv2

class MySIFT(FeatureMatchingInterface):
    def __init__(self):
        # Create a SIFT detector object
        self.sift = cv2.SIFT_create()


    def my_detectAndCompute(self, gray_image: np.ndarray, mask: np.ndarray = None) -> tuple:
        """
        Detects keypoints and computes descriptors using the specified detector on the given gray image.

        Args:
            detector (cv2.SIFT): The detector to be used for keypoint detection and descriptor computation.
            gray_image (np.ndarray): The input gray image on which keypoints and descriptors are computed.
            mask (np.ndarray, optional): An optional mask specifying where to detect keypoints. Defaults to None.

        Returns:
            tuple: A tuple containing the detected keypoints and computed descriptors.
        """
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, mask)
        return keypoints, descriptors

    def drawKeyPoints(self, image: np.ndarray, keypoints: list, outImage: np.ndarray = None ) -> np.ndarray:
        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the image with keypoints
        cv2.imshow('SIFT Keypoints', image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveKeypoints(self, filename: str, keypoints: list, descriptors: np.ndarray) -> None:
        # Save keypoints and descriptors to a file
        np.savez(filename, keypoints=keypoints, descriptors=descriptors)

    def matchingKeypoints(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
