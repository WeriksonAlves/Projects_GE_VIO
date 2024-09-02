import cv2
import os
import numpy as np


from ..auxiliary.Model import Model

from typing import List, Tuple
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision


class Camera:
    def __init__(
        self,
        uav: Bebop,
        chessboard_size: Tuple[int, int] = (7, 10),
        square_size: int = 22,
        criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        ),
    ):
        """
        Initializes the CalibrationSystem object.

        :param uav: The Bebop drone object.
        :param chessboard_size: The size of the chessboard pattern.
        :param square_size: The size of the squares in the chessboard pattern.
        :param criteria: The criteria for corner detection.
        """

        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = criteria
        self.bebop = uav

    def capture_images(
        self,
        attempts: int = 50,
        buffer_size: int = 10,
        save: bool = False,
        path: str = None,
        prep_time: int = 10,
        capture_time: int = 30,
    ) -> bool:
        """
        Captures images for calibration.

        :param attempts: Number of connection attempts to the Bebop drone.
        :param buffer_size: Size of the image buffer. Default is 10.
        :param save: Flag indicating whether to save the images or not.
        :param path: The path where the images should be saved.
        :param prep_time: Time in seconds to prepare before capturing images.
        :param capture_time: Time in seconds to capture images.
        :return: True if images were captured successfully, False otherwise.
        """
        success = self.bebop.connect(attempts)
        if not success:
            print("Error connecting to Bebop. Please retry.")
            return False

        bebop_vision = DroneVision(
            self.bebop, Model.BEBOP, buffer_size=buffer_size
        )
        user_vision = UserVision(bebop_vision)
        bebop_vision.set_user_callback_function(
            user_vision.save_image, user_callback_args=(save, path)
        )

        success = bebop_vision.open_video()
        if not success:
            print("Error starting video stream.")
            return False

        print(
            f"\nGet ready to capture images for calibration. You have \
                {prep_time} seconds to prepare.\n"
        )
        self.bebop.smart_sleep(prep_time)

        print(
            f"Move the drone around and hold the pattern in front of the \
                camera for {capture_time} seconds."
        )
        self.bebop.smart_sleep(capture_time)

        print("Finishing and stopping vision")
        bebop_vision.close_video()
        self.bebop.disconnect()
        return True

    def process_images(
        self,
        image_files: List[str],
        num_images: int,
        display: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Processes images to detect chessboard corners and return the object
        points, image points, and object pattern.

        :param image_files: List of file paths to the input images.
        :param num_images: Number of images to process.
        :param display: Whether to display the images with detected corners.
        :return: Object points, image points, and object pattern.
        """
        obj_pattern = np.zeros(
            (self.chessboard_size[1] * self.chessboard_size[0], 3), np.float32
        )
        obj_pattern[:, :2] = (
            np.mgrid[
                0:self.chessboard_size[0], 0:self.chessboard_size[1]
            ].T.reshape(-1, 2)
            * self.square_size
        )

        object_points = []
        image_points = []

        if num_images >= 150:
            print(
                "\n\nToo many images to process. Limiting to 150 images.\n\n"
            )
            num_images = 150

        print("Processed images: ", end="")
        count = 0
        for idx, image_file in enumerate(image_files):
            if idx % (len(image_files) // num_images) == 0:
                print(f"{idx} ", end="", flush=True)
                count += 1
                image = cv2.imread(image_file)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(
                    gray_image, self.chessboard_size, None
                )
                if ret:
                    object_points.append(obj_pattern)
                    refined_corners = cv2.cornerSubPix(
                        gray_image, corners, (11, 11), (-1, -1), self.criteria
                    )
                    image_points.append(refined_corners)

                    if display:
                        image = cv2.drawChessboardCorners(
                            image, self.chessboard_size, refined_corners, ret
                        )
                        cv2.imshow("Image", image)
                        cv2.waitKey(100)

        cv2.destroyAllWindows()
        print(f"\nTotal of images processed {count}")
        print(
            f"Number of images where corners were detected: \
                {len(object_points)}"
        )

        return object_points, image_points, obj_pattern

    def calibrate_camera(
        self,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        image_size: Tuple[int, int],
        use_extended_model: bool = False
    ) -> Tuple[
        bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]
    ]:
        """
        Calibrates the camera using object points, image points, and image
        size.

        :param object_points: List of arrays containing 3D object points.
        :param image_points: List of arrays containing 2D image points.
        :param image_size: Tuple containing the width and height of the image.
        :param use_extended_model: Flag indicating whether to use the extended
        :return: Success flag, intrinsic matrix, distortion coefficients,
            rotation vectors, and translation vectors.
        """
        if use_extended_model:
            dist_coeffs = np.zeros((14, 1))  # Extended model
        else:
            dist_coeffs = None  # Default 5-coefficient model

        (
            ret,
            intrinsic_matrix,
            distortion_coeffs,
            rotation_vecs,
            translation_vecs,
        ) = cv2.calibrateCamera(
            object_points, image_points, image_size[::-1], None, dist_coeffs
        )
        return (
            ret,
            intrinsic_matrix,
            distortion_coeffs,
            rotation_vecs,
            translation_vecs,
        )

    def validate_calibration(
        self,
        object_points: List[np.ndarray],
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray],
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        image_points: List[np.ndarray],
        error_threshold: float = 0.5
    ) -> bool:
        """
        Validates the calibration by calculating the mean reprojection error.

        :param object_points: List of object points.
        :param rvecs: List of rotation vectors.
        :param tvecs: List of translation vectors.
        :param intrinsic_matrix: Intrinsic camera matrix.
        :param distortion_coeffs: Distortion coefficients.
        :param image_points: List of image points.
        """
        total_error = 0
        for i in range(len(object_points)):
            projected_image_points, _ = cv2.projectPoints(
                object_points[i],
                rvecs[i],
                tvecs[i],
                intrinsic_matrix,
                distortion_coeffs,
            )
            error = cv2.norm(
                image_points[i], projected_image_points, cv2.NORM_L2
            ) / len(projected_image_points)
            total_error += error

        mean_error = total_error / len(object_points)
        print(f"Mean reprojection error: {mean_error}")

        # Check if the reprojection error is within the acceptable threshold
        if mean_error > error_threshold:
            print(
                "Calibration error exceeds the acceptable threshold. \
                Consider redoing the calibration."
            )
            return False
        else:
            print("Calibration is within acceptable error limits.")
            return True

    def save_calibration(
        self,
        file_name: str,
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        rotation_vecs: List[np.ndarray],
        translation_vecs: List[np.ndarray],
    ) -> None:
        """
        Saves the calibration parameters to a file.

        :param file_name: Name of the file to save the calibration parameters.
        :param intrinsic_matrix: Intrinsic matrix of the camera.
        :param distortion_coeffs: Distortion coefficients of the camera.
        :param rotation_vecs: Rotation vectors of the camera.
        :param translation_vecs: Translation vectors of the camera.
        """
        np.savez(
            file_name,
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            rotation_vecs=rotation_vecs,
            translation_vecs=translation_vecs,
        )

    def load_calibration(
        self,
        file_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Loads the calibration parameters from a file.

        :param file_name: Name of the file to load the calibration parameters.
        :return: Intrinsic matrix, distortion coefficients, rotation vectors,
            and translation vectors.
        """
        with np.load(file_name) as data:
            intrinsic_matrix = data["intrinsic_matrix"]
            distortion_coeffs = data["distortion_coeffs"]
            rotation_matrix = data["rotation_vecs"]
            translation_vector = data["translation_vecs"]

        return (
            intrinsic_matrix,
            distortion_coeffs,
            rotation_matrix,
            translation_vector,
        )


class UserVision:
    def __init__(
        self,
        vision: DroneVision
    ) -> None:
        """
        Initialize the UserVision class with a vision object.

        :param vision: The DroneVision object responsible for image capture.
        """
        self.image_index = 1
        self.vision = vision

    def save_image(
        self,
        args: bool = False,
        dataset_path: str = None
    ) -> None:
        """
        Saves the latest valid picture captured by the vision system.

        :param args: Flag indicating whether to save the image or not.
        :param dataset_path: The path where the images should be saved.
        """
        image = self.vision.get_latest_valid_picture()
        cv2.imshow("Captured Image", image)
        cv2.waitKey(1)

        if image is not None:
            if args[0]:
                if not os.path.exists(args[1]):
                    os.makedirs(args[1])

                filename = os.path.join(
                    args[1], f"image_{self.image_index:04d}.png"
                )
                cv2.imwrite(filename, image)
            self.image_index += 1
