import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
from enum import Enum, auto
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from mpl_toolkits.mplot3d import Axes3D

class Bebop2CameraCalibration:

    def __init__(
            self,
            chessboard_size: Tuple[int, int] = (7, 10),        
            square_size: int = 22,        
            criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),        
            axis_length: float = 1.5
        ):
        """
        Initializes the CalibrationSystem object.
        Args:
            - chessboard_size (Tuple[int, int]): The size of the chessboard as a tuple of integers (rows, columns).
            - square_size (int): The size of each square in the chessboard.
            - criteria (Tuple[int, int, float]): The criteria used for calibration.
            - axis_length (float): The length of the axis used for visualization.
        Returns:
            None
        """
        
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = criteria
        self.axis_length = axis_length
        self.bebop = Bebop()
    
    def capture_images(
            self, 
            attempts: int = 50, 
            buffer_size: int = 10,
            save: bool = False,
            path: str = None,
            prep_time: int = 10, 
            capture_time: int = 30
        ) -> bool:
        """
        Captures images for calibration.
        Args:
            - attempts (int): Number of connection attempts to the Bebop drone. Default is 50.
            - buffer_size (int): Size of the image buffer. Default is 10.
            - save (bool): Flag indicating whether to save the images or not. Default is False.
            - path (str): The path where the images should be saved. Default is None.
            - prep_time (int): Time in seconds to prepare before capturing images. Default is 10.
            - capture_time (int): Time in seconds to capture images. Default is 30.
        Returns:
            - bool: True if images were captured successfully, False otherwise.
        """
        success = self.bebop.connect(attempts)
        if not success:
            print("Error connecting to Bebop. Please retry.")
            return False
        
        bebop_vision = DroneVision(self.bebop, Model.BEBOP, buffer_size=buffer_size)
        user_vision = UserVision(bebop_vision)
        bebop_vision.set_user_callback_function(user_vision.save_image(save=save, path=path), user_callback_args=None)
        
        success = bebop_vision.open_video()
        if not success:
            print("Error starting video stream.")
            return False
        
        print(f"\nGet ready to capture images for calibration. You have {prep_time} seconds to prepare.\n")
        self.bebop.smart_sleep(prep_time)
        
        print(f"Move the drone around and hold the pattern in front of the camera for {capture_time} seconds.")
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
        Processes images to detect chessboard corners and return the object points, image points, and object pattern.
        
        Args:
            - image_files (List[str]): List of file paths to the input images.
            - num_images (int): Number of images to process.
            - display (bool): Whether to display the images with detected corners. Default is False.
        
        Returns:
            - Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]: Object points, image points, and object pattern.
        """
        obj_pattern = np.zeros((self.chessboard_size[1] * self.chessboard_size[0], 3), np.float32)
        obj_pattern[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size
        
        object_points = []
        image_points = []
        
        print(f"Processing {num_images} images: ", end='')
        for idx, image_file in enumerate(image_files):
            if idx % (len(image_files) // num_images) == 0:
                print(f"{idx} ", end='', flush=True)
                image = cv2.imread(image_file)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                ret, corners = cv2.findChessboardCorners(gray_image, self.chessboard_size, None)
                if ret:
                    object_points.append(obj_pattern)
                    refined_corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), self.criteria)
                    image_points.append(refined_corners)
                    
                    if display:
                        image = cv2.drawChessboardCorners(image, self.chessboard_size, refined_corners, ret)
                        cv2.imshow('Image', image)
                        cv2.waitKey(100)
        
        cv2.destroyAllWindows()
        print(f'\n\nNumber of images where corners were detected: {len(object_points)}')
        
        return object_points, image_points, obj_pattern
    
    def calibrate_camera(
        self, 
        object_points: List[np.ndarray], 
        image_points: List[np.ndarray], 
        image_size: Tuple[int, int]
    ) -> Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Calibrates the camera using object points, image points, and image size.
        
        Args:
            - object_points (List[np.ndarray]): List of arrays containing 3D object points.
            - image_points (List[np.ndarray]): List of arrays containing 2D image points.
            - image_size (Tuple[int, int]): Tuple containing the width and height of the image.
        
        Returns:
            - Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
                - Success flag.
                - Intrinsic matrix of the camera.
                - Distortion coefficients of the camera.
                - Rotation vectors.
                - Translation vectors.
        """
        ret, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = cv2.calibrateCamera(
            object_points, image_points, image_size[::-1], None, None
        )
        return ret, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs
    
    def validate_calibration(
        self, 
        object_points: List[np.ndarray], 
        rvecs: List[np.ndarray], 
        tvecs: List[np.ndarray], 
        intrinsic_matrix: np.ndarray, 
        distortion_coeffs: np.ndarray, 
        image_points: List[np.ndarray]
    ):
        """
        Validates the calibration by calculating the mean reprojection error.
        
        Args:
            - object_points (List[np.ndarray]): List of object points.
            - rvecs (List[np.ndarray]): List of rotation vectors.
            - tvecs (List[np.ndarray]): List of translation vectors.
            - intrinsic_matrix (np.ndarray): Intrinsic camera matrix.
            - distortion_coeffs (np.ndarray): Distortion coefficients.
            - image_points (List[np.ndarray]): List of image points.
        
        Returns:
            None
        Prints:
            - Mean reprojection error.
        """
        total_error = 0
        for i in range(len(object_points)):
            projected_image_points, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], intrinsic_matrix, distortion_coeffs
            )
            error = cv2.norm(image_points[i], projected_image_points, cv2.NORM_L2) / len(projected_image_points)
            total_error += error
        
        mean_error = total_error / len(object_points)
        print(f"Mean reprojection error: {mean_error}")
    
    def save_calibration(
        self, 
        file_name: str, 
        intrinsic_matrix: np.ndarray, 
        distortion_coeffs: np.ndarray, 
        rotation_vecs: List[np.ndarray], 
        translation_vecs: List[np.ndarray]
    ) -> None:
        """
        Saves the calibration parameters to a file.
        
        Args:
            - file_name (str): The name of the file to save the calibration parameters.
            - intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera.
            - distortion_coeffs (np.ndarray): The distortion coefficients of the camera.
            - rotation_vecs (List[np.ndarray]): The rotation vectors of the camera.
            - translation_vecs (List[np.ndarray]): The translation vectors of the camera.
        
        Returns:
            - None
        """
        np.savez(
            file_name, 
            intrinsic_matrix=intrinsic_matrix, 
            distortion_coeffs=distortion_coeffs, 
            rotation_vecs=rotation_vecs, 
            translation_vecs=translation_vecs
        )
    
    def load_calibration(self, file_name: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Loads the calibration parameters from a file.
        
        Args:
            - file_name (str): The name of the file containing the calibration parameters.
        
        Returns:
            - Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
                - Intrinsic matrix of the camera.
                - Distortion coefficients of the camera.
                - Rotation vectors of the camera.
                - Translation vectors of the camera.
        """
        with np.load(file_name) as data:
            intrinsic_matrix = data['intrinsic_matrix']
            distortion_coeffs = data['distortion_coeffs']
            rotation_matrix = data['rotation_vecs']
            translation_vector = data['translation_vecs']
            
        return intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_vector
    
    def setup_3d_plot(
        self,
        ax: Optional[Axes3D] = None,
        figsize: Tuple[int, int] = (9, 8),
        projection: str = '3d',
        x_lim: Tuple[int, int] = (-2, 2),
        y_lim: Tuple[int, int] = (-2, 2),
        z_lim: Tuple[int, int] = (-2, 2)
    ) -> Axes3D:
        """
        Set up a 3D plot with specified axis limits and labels.
        
        Args:
            ax (Axes3D, optional): The 3D axes to plot on. If None, a new one will be created.
            figsize (tuple, optional): The size of the figure in inches. Default is (9, 8).
            x_lim (tuple, optional): The limits of the x-axis. Default is (-2, 2).
            y_lim (tuple, optional): The limits of the y-axis. Default is (-2, 2).
            z_lim (tuple, optional): The limits of the z-axis. Default is (-2, 2).
        
        Returns:
            Axes3D: The 3D axes object.
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("Camera Calibration")
        
        return ax
    
    def plot_camera_axes(
        self,
        origin: Tuple[float, float, float],
        rotation_matrix: np.ndarray,
        ax: Axes3D
    ) -> Axes3D:
        """
        Plot the camera axes on the given 3D axis.
        
        Args:
            origin (tuple): The origin of the camera axes.
            rotation_matrix (np.ndarray): The rotation matrix for the camera.
            ax (Axes3D): The 3D axis to plot on.
        
        Returns:
            Axes3D: The modified 3D axis.
        """
        axis_colors = ['red', 'green', 'blue']
        
        for i in range(3):
            ax.quiver(
                origin[0], origin[1], origin[2],
                rotation_matrix[0, i], rotation_matrix[1, i], rotation_matrix[2, i],
                color=axis_colors[i], pivot='tail', length=self.axis_length
            )
        
        return ax
    
    def plot_camera_movement(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray
    ) -> Axes3D:
        """
        Plot the camera movement in a 3D plot.
        
        Args:
            rotation_vecs (np.ndarray): Array of rotation vectors.
            translation_vecs (np.ndarray): Array of translation vectors.
            object_points (np.ndarray): Array of object points.
        
        Returns:
            Axes3D: The 3D plot with the camera movement.
        """
        ax = self.setup_3d_plot(x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[-1000, 0])
        unit_vectors = np.eye(3).T
        camera_positions = np.zeros(translation_vecs.shape)
        
        for i in range(rotation_vecs.shape[1]):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
            camera_positions[:, i] = -rotation_matrix.T @ translation_vecs[:, i]
            rotated_axes = rotation_matrix.T @ unit_vectors
            ax = self.plot_camera_axes(camera_positions[:, i], rotated_axes, ax)
        
        ax.plot_wireframe(object_points[0], object_points[1], np.zeros_like(object_points[0]))
        
        return ax

    def plot_moving_pattern(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray
    ) -> Axes3D:
        """
        Plot the moving pattern of a camera calibration.
        
        Args:
            rotation_vecs (np.ndarray): Array of rotation vectors.
            translation_vecs (np.ndarray): Array of translation vectors.
            object_points (np.ndarray): Array of object points.
        
        Returns:
            Axes3D: The 3D plot axes.
        """
        ax = self.setup_3d_plot(x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[0, 1000])
        unit_vectors = np.eye(3)
        ax = self.plot_camera_axes([0, 0, 0], unit_vectors, ax)
        
        for i in range(rotation_vecs.shape[1]):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
            rotated_objp = rotation_matrix @ object_points.T + translation_vecs[:, i].reshape(-1, 1)
            ax.scatter(rotated_objp[0, :], rotated_objp[1, :], rotated_objp[2, :])
        
        return ax
    
    def display_extrinsic_parameters(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray
    ) -> None:
        """
        Display the extrinsic parameters of a camera calibration.
        
        Args:
            rotation_vecs (np.ndarray): Array of rotation vectors.
            translation_vecs (np.ndarray): Array of translation vectors.
            object_points (np.ndarray): Array of object points.
        
        Returns:
            None
        """
        ax_movement = self.plot_camera_movement(rotation_vecs, translation_vecs, np.meshgrid(object_points[:, 0], object_points[:, 1]))
        ax_movement.view_init(elev=-61, azim=-90)
        ax_movement._dist = 8
        plt.show()
        
        ax_pattern = self.plot_moving_pattern(rotation_vecs, translation_vecs, object_points)
        ax_pattern.view_init(elev=-45, azim=-90)
        ax_pattern._dist = 8
        
        plt.show()

class Model(Enum):
    """
    Enum class representing different models.
    
    Attributes:
        BEBOP: Model representing Bebop.
        MAMBO: Model representing Mambo.
        ANAFI: Model representing Anafi.
    """
    BEBOP = auto()
    MAMBO = auto()
    ANAFI = auto()

class UserVision:
    def __init__(self, vision: DroneVision):
        """
        Initialize the UserVision class with a vision object.
        
        Args:
            vision (DroneVision): The DroneVision object responsible for image capture.
        """
        self.image_index = 0
        self.vision = vision
    
    def save_image(self, save: bool = False, path: str = None) -> None:
        """
        Saves the latest valid picture captured by the vision system.
        Args:
            save (bool, optional): Flag indicating whether to save the image or not. Defaults to False.
            path (str, optional): The path where the image should be saved. Defaults to None.
        Returns:
            None
        """
        image = self.vision.get_latest_valid_picture()
        
        if image is not None:
            cv2.imshow('Captured Image', image)
            cv2.waitKey(1)
            filename = f"calibration_image_{self.image_index:05d}.png"
            
            if save:
                if not os.path.exists(path):
                    os.makedirs(path)
                
                cv2.imwrite(filename, image)
                print(f"Saved {filename}")
            self.image_index += 1
