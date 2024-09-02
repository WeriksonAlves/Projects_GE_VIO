import cv2

import numpy as np

from typing import Tuple, List, Optional, Union


class FeatureExtractor:
    def __init__(
        self, method: str = "AKAZE", parammeters: Union[dict, None] = None
    ) -> None:
        """
        Initialize the feature extractor.

        :param method: Feature extraction method, either "AKAZE" or "FAST".
        :param parammeters: Parameters for the feature extraction method.
        """
        self.method = method
        self.parammeters = parammeters
        self.detector, self.descriptor = self._initialize_detector()

    def _initialize_detector(self) -> Tuple[cv2.Feature2D, cv2.Feature2D]:
        """
        Initialize the feature detector and descriptor.

        :return: Detector and descriptor.
        """
        if self.method == "AKAZE":
            detector = cv2.AKAZE_create()
            descriptor = detector  # AKAZE combines detector and descriptor
        elif self.method == "FAST":
            detector = cv2.FastFeatureDetector_create()
            detector.setNonmaxSuppression(self.parammeters["suppression"])
            detector.setThreshold(self.parammeters["threshold"])
            descriptor = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        return detector, descriptor

    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract keypoints and descriptors from an image.

        :param image: Input image.
        :return: Keypoints and descriptors.
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors


class FeatureMatcher:
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check: bool = True):
        """
        Initialize the feature matcher.

        :param norm_type: Norm type for matching, default is NORM_HAMMING.
        :param cross_check: If true, use cross-checking in matching.
        """
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)

    def match_features(
        self, des1: np.ndarray, des2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors between two sets.

        :param des1: Descriptors from the first image.
        :param des2: Descriptors from the second image.
        :return: List of matches.
        """
        matches = self.matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)


class ModelFitter:
    def __init__(
        self, prob: float = 0.999, reproj_thresh: float = 3.0
    ) -> None:
        """
        Initialize the model fitter with RANSAC parameters.

        :param prob: Probability of success for RANSAC.
        :param reproj_thresh: Reprojection error threshold for RANSAC.
        """
        self.prob = prob
        self.reproj_thresh = reproj_thresh

    def fit_fundamental_matrix(
        self, pts1: np.ndarray, pts2: np.ndarray, max_iter: int
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the fundamental matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :param max_iter: Maximum number of iterations for RANSAC.
        :return: Fundamental matrix, inliers from pts1, inliers from pts2.
        """
        # 2) Repetir
        # 2.1) Selecionar aleatoriamente um conjunto de s pontos de A
        # 2.2) Ajustar um modelo a esses pontos
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, self.reproj_thresh, maxIters=max_iter
        )

        # Verificar se uma matriz fundamental válida foi encontrada
        if F is None or mask is None:
            print("Warning: No valid fundamental matrix found.")
            return None, np.array([]), np.array([])

        # 2.4) Construir o conjunto de inliers (i.e., contar o número de
        # pontos cuja distância do modelo < d)
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]

        # 2.5) Armazenar esses inliers (já feito internamente pelo OpenCV)
        return F, inliers1, inliers2

    def fit_essential_matrix(
        self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, max_iter: int
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Fit the essential matrix using RANSAC.

        :param pts1: Points from the first image.
        :param pts2: Corresponding points from the second image.
        :param K: Camera intrinsic matrix.
        :return: Essential matrix, inliers from pts1, inliers from pts2.
        """
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.reproj_thresh,
            maxIters=max_iter,
        )

        # Check if a valid essential matrix was found
        if E is None or mask is None:
            print("Warning: No valid essential matrix found.")
            return None, np.array([]), np.array([])

        # Select inliers
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]

        return E, inliers1, inliers2


class VisualOdometry:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        feature_matcher: FeatureMatcher,
        model_fitter: ModelFitter,
    ) -> None:
        """
        Initialize the Visual Odometry pipeline.

        :param feature_extractor: Feature extractor instance.
        :param feature_matcher: Feature matcher instance.
        :param model_fitter: Model fitter instance.
        """
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.model_fitter = model_fitter

    # 1) Capturar novo frame Ik
    def capture_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Capture a new frame.

        :param image: The input image.
        :return: The captured frame.
        """
        return image

    # 2) Extrair e combinar características entre Ik-1 e Ik
    def extract_and_match_features(
        self, img1: np.ndarray, img2: np.ndarray, display: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Extract and match features between two images.

        :param img1: The first image.
        :param img2: The second image.
        :return: A tuple containing the keypoints, descriptors, and matches.
        """
        kp1, des1 = self.feature_extractor.extract_features(img1)
        kp2, des2 = self.feature_extractor.extract_features(img2)
        matches = self.feature_matcher.match_features(des1, des2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        if display:
            # Visualize the matches
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
            cv2.imshow("Matches", img_matches)
            cv2.waitKey(1)

        return pts1, pts2, matches

    # 3) Calcular a matriz essencial para o par de imagens Ik-1, Ik
    def compute_essential_matrix(
        self, pts1: np.ndarray, pts2: np.ndarray, intrinsic_matrix: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Compute the essential matrix for a pair of images.

        :param pts1: The keypoints of the first image.
        :param pts2: The keypoints of the second image.
        :param intrinsic_matrix: The camera intrinsic matrix.
        :return: A tuple containing the essential matrix, inliers of the first
        image, and inliers of the second image.
        """
        return self.model_fitter.fit_essential_matrix(
            pts1, pts2, intrinsic_matrix, max_iter=5000
        )

    # 4) Decompor a matriz essencial em Rk e tk e formar Tk
    def decompose_essential_matrix(
        self, E: np.ndarray, pts1, pts2, intrinsic_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the essential matrix into rotation and translation matrices.

        :param E: The essential matrix.
        :param pts1: The keypoints of the first image.
        :param pts2: The keypoints of the second image.
        :param intrinsic_matrix: The camera intrinsic matrix.
        :return: A tuple containing the rotation matrix and translation matrix.
        """
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, intrinsic_matrix)
        return R, t

    # 5) Calcular a escala relativa e reescalar tk de acordo
    def rescale_translation(self, t: np.ndarray, scale: float) -> np.ndarray:
        """
        Rescale the translation vector.

        :param t: The translation vector.
        :param scale: The scale factor.
        :return: The rescaled translation vector.
        """
        return t * scale

    # 6) Concatenar a transformação computando Ck = Ck-1 Tk
    def concatenate_transformation(
        self, C_prev: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate the transformation matrices.

        :param C_prev: The previous transformation matrix.
        :param R: The rotation matrix.
        :param t: The translation matrix.
        :return: The concatenated transformation matrix.
        """
        T = np.hstack((R, t))
        T = np.vstack((T, [0, 0, 0, 1]))
        C = np.dot(C_prev, T)
        return C

    # 7) Repetir desde 1).
    def process_frames(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        intrinsic_matrix: Optional[np.ndarray] = None,
        display: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Process two frames and estimate the transformation matrix.

        :param img1: The first image.
        :param img2: The second image.
        :param intrinsic_matrix: The camera intrinsic matrix (optional).
        :return: The estimated transformation matrix (either F or E).
        """
        # 1) Capturar novo frame Ik (Neste caso, os frames já foram capturados
        # e fornecidos)
        # self.capture_frame(img1)
        # self.capture_frame(img2)

        # 2) Extrair e combinar características entre Ik-1 e Ik
        pts1, pts2, matches = self.extract_and_match_features(
            img1, img2, display
        )

        if intrinsic_matrix is not None:
            # 3) Calcular a matriz essencial para o par de imagens Ik-1, Ik
            E, inliers1, inliers2 = self.compute_essential_matrix(
                pts1, pts2, intrinsic_matrix
            )
            if E is None:
                print("No valid essential matrix found, skipping frame.")
                return None

            # 4) Decompor a matriz essencial em Rk e tk e formar Tk
            R, t = self.decompose_essential_matrix(
                E, pts1, pts2, intrinsic_matrix
            )

            # 5) Calcular a escala relativa e reescalar tk de acordo
            t_rescaled = self.rescale_translation(
                t, scale=0.5
            )  # Aqui, a escala é arbitrariamente 1.0

            # 6) Concatenar a transformação computando Ck = Ck-1 Tk
            # Exemplo: se houvesse um Ck-1 previamente conhecido
            C_prev = np.eye(4)  # Supondo Ck-1 = identidade como exemplo
            C = self.concatenate_transformation(C_prev, R, t_rescaled)

            # 7) Repetir desde 1) (Aqui, está retornando a transformação atual)
            return C

        else:
            F, inliers1, inliers2 = self.model_fitter.fit_fundamental_matrix(
                pts1, pts2, max_iter=1000
            )
            if F is None:
                print("No valid fundamental matrix found, skipping frame.")
                return None
            return F
