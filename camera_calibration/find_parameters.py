import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Union, Tuple


def setup_3d_plot(ax: Axes3D = None, figsize: tuple =(9, 8), x_lim: tuple = (-2, 2), y_lim: tuple = (-2, 2), z_lim: tuple = (-2, 2)) -> Axes3D:
    """
    Set up a 3D plot for camera calibration.

    Parameters:
    - ax (Axes3D, optional): The 3D axes to plot on. If not provided, a new figure and axes will be created.
    - figsize (tuple, optional): The size of the figure in inches. Default is (9, 8).
    - x_lim (tuple, optional): The limits of the x-axis. Default is (-2, 2).
    - y_lim (tuple, optional): The limits of the y-axis. Default is (-2, 2).
    - z_lim (tuple, optional): The limits of the z-axis. Default is (-2, 2).

    Returns:
    - ax (Axes3D): The 3D axes object.

    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Camera Calibration")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    return ax


def plot_camera_axes(origin: Union[list,tuple], rotation_matrix: np.ndarray, ax: Axes3D, axis_length: float = 1.5) -> Axes3D:
    """
    Plots the camera axes on a given matplotlib axis.

    Parameters:
    - origin: The origin point of the camera axes. It can be a list or tuple containing the x, y, and z coordinates.
    - rotation_matrix: The rotation matrix representing the camera axes orientation.
    - ax: The matplotlib axis on which to plot the camera axes.
    - axis_length: The length of the camera axes arrows. Default is 1.5.

    Returns:
    - The modified matplotlib axis with the camera axes plotted.
    """
    axis_colors = ['red', 'green', 'blue']

    for i in range(3):
        ax.quiver(
            origin[0], origin[1], origin[2],
            rotation_matrix[0, i], rotation_matrix[1, i], rotation_matrix[2, i],
            color=axis_colors[i], pivot='tail', length=axis_length
        )

    return ax


def process_images(images: list, num_images: int, chessboard_size: tuple, square_size: int, criteria: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a list of images to find chessboard corners and extract object and image points.

    Args:
        images (list): A list of image file paths.
        num_images (int): The number of images to process.
        chessboard_size (tuple): The size of the chessboard as a tuple (rows, columns).
        square_size (int): The size of each square in the chessboard.
        criteria (tuple): The criteria for corner refinement as a tuple (criteria_type, max_iterations, epsilon).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the object points, image points, and object pattern points.
    """

    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    object_points = []
    image_points = []
    image_number_read = 0

    for idx, image_file in enumerate(images):
        if idx % int(len(images)/num_images) == 0:
            image_number_read += 1
            print(f"Processing image {idx}")

            image = cv2.imread(image_file)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

            if ret:
                object_points.append(objp)
                refined_corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
                image_points.append(refined_corners)

                image = cv2.drawChessboardCorners(image, chessboard_size, refined_corners, ret)
                cv2.imshow('Image', image)
                cv2.waitKey(100)

    cv2.destroyAllWindows()
    print(f'\nNumber of images read: {image_number_read}\n')
    print(f'\nNumber of images where corners were detected: {len(object_points)}\n')

    return object_points, image_points, objp


def calibrate_camera(object_points: np.ndarray, image_points: np.ndarray, image_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrates the camera using object points, image points, and image size.

    Args:
        object_points (np.ndarray): Array of object points in the world coordinate system.
        image_points (np.ndarray): Array of corresponding image points in the image coordinate system.
        image_size (np.ndarray): Size of the input images.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the intrinsic matrix, distortion coefficients, rotation vectors, and translation vectors.

    """
    ret, intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs = cv2.calibrateCamera(object_points, image_points, image_size[::-1], None, None)
    np.savez('calibration_result_2608.npz', intrinsic_matrix=intrinsic_matrix, distortion_coeffs=distortion_coeffs,
            rotation_vecs=rotation_vecs, translation_vecs=translation_vecs)

    print('\nIntrinsic Matrix:\n', intrinsic_matrix)
    print('\nRadial Distortion Coefficients:\n', distortion_coeffs)

    return intrinsic_matrix, distortion_coeffs, rotation_vecs, translation_vecs

def plot_camera_movement(rotation_vecs: np.ndarray, translation_vecs: np.ndarray, object_points: np.ndarray, axis_length: int) -> Axes3D: 
    """
    Plots the camera movement in a 3D plot.
    Args:
        rotation_vecs (np.ndarray): Array of rotation vectors.
        translation_vecs (np.ndarray): Array of translation vectors.
        object_points (np.ndarray): Array of object points.
        axis_length (int): Length of the camera axes.
    Returns:
        Axes3D: The 3D plot with the camera movement.
    """
    
    ax = setup_3d_plot(x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[-1000, 0])
    unit_vectors = np.eye(3).T
    camera_positions = np.zeros(translation_vecs.shape)

    for i in range(rotation_vecs.shape[1]):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
        camera_positions[:, i] = -rotation_matrix.T @ translation_vecs[:, i]
        rotated_axes = rotation_matrix.T @ unit_vectors
        ax = plot_camera_axes(camera_positions[:, i], rotated_axes, ax, axis_length)

    ax.plot_wireframe(object_points[0], object_points[1], np.zeros_like(object_points[0]))

    return ax

def plot_moving_pattern(rotation_vecs: np.ndarray, translation_vecs: np.ndarray, object_points: np.ndarray, axis_length: int) -> Axes3D:
    """
    Plots the moving pattern of a camera calibration.

    Parameters:
    rotation_vecs (np.ndarray): Array of rotation vectors.
    translation_vecs (np.ndarray): Array of translation vectors.
    object_points (np.ndarray): Array of object points.
    axis_length (int): Length of the camera axes.

    Returns:
    Axes3D: The 3D plot axes.
    """
    ax = setup_3d_plot(x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[0, 1000])
    unit_vectors = np.eye(3)
    ax = plot_camera_axes([0, 0, 0], unit_vectors, ax, axis_length)

    for i in range(rotation_vecs.shape[1]):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
        rotated_objp = rotation_matrix @ object_points.T + translation_vecs[:, i].reshape(-1, 1)
        ax.scatter(rotated_objp[0, :], rotated_objp[1, :], rotated_objp[2, :])

    return ax

def display_extrinsic_parameters(rotation_vecs: np.ndarray, translation_vecs: np.ndarray, object_points: np.ndarray, axis_length: int = 80) -> None:
    """
    Display the extrinsic parameters of a camera calibration.

    Parameters:
    rotation_vecs (np.ndarray): Array of rotation vectors.
    translation_vecs (np.ndarray): Array of translation vectors.
    object_points (np.ndarray): Array of object points.
    axis_length (int, optional): Length of the axis lines. Defaults to 80.

    Returns:
    None
    """
    ax_movement = plot_camera_movement(rotation_vecs, translation_vecs, np.meshgrid(object_points[:, 0], object_points[:, 1]), axis_length)
    ax_movement.view_init(elev=-61, azim=-90)
    ax_movement.dist = 8

    ax_pattern = plot_moving_pattern(rotation_vecs, translation_vecs, object_points, axis_length)
    ax_pattern.view_init(elev=-45, azim=-90)
    ax_pattern.dist = 8

    plt.show()

def main() -> None:
    """
    Main function for camera calibration.

    This function performs camera calibration using a set of calibration images.
    It detects corners in the calibration images, prepares object points and
    image points, and then calibrates the camera using the detected points.

    Returns:
        None
    """
    # Define termination criteria and chessboard size
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (7, 10) # (rows, columns)
    square_size = 22 # millimeters

    # Path to calibration images
    image_dir = 'Projects_GE_VIO/camera_calibration/dataset_images_uav/*.png'
    calibration_images = glob.glob(image_dir)

    print(f'Number of images read: {len(calibration_images)}')

    # Process images to detect corners and prepare object points
    object_points, image_points, objp = process_images(calibration_images, 15, chessboard_size, square_size, termination_criteria)

    # Calibrate the camera
    example_image = cv2.imread(calibration_images[0])
    gray_example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)
    _, _, rotation_vectors, translation_vectors = calibrate_camera(object_points, image_points, gray_example_image.shape)

    # Display extrinsic parameters in different scenarios
    display_extrinsic_parameters(np.hstack(rotation_vectors), np.hstack(translation_vectors), objp)

if __name__ == "__main__":
    main()