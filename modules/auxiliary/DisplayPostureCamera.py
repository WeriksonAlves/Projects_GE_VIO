import cv2

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, Optional
from mpl_toolkits.mplot3d import Axes3D


class DisplayPostureCamera:
    @staticmethod
    def setup_3d_plot(
        ax: Optional[Axes3D] = None,
        figsize: Tuple[int, int] = (9, 8),
        projection: str = "3d",
        x_lim: Tuple[int, int] = (-2, 2),
        y_lim: Tuple[int, int] = (-2, 2),
        z_lim: Tuple[int, int] = (-2, 2),
    ) -> Axes3D:
        """
        Set up a 3D plot with specified axis limits and labels.

        :param ax: The 3D axes to plot on. If None, a new one will be created.
        :param figsize: The size of the figure in inches. Default is (9, 8).
        :param projection: The projection of the 3D plot. Default is '3d'.
        :param x_lim: The limits of the x-axis. Default is (-2, 2).
        :param y_lim: The limits of the y-axis. Default is (-2, 2).
        :param z_lim: The limits of the z-axis. Default is (-2, 2).
        :return: The 3D axes object.
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

    @staticmethod
    def plot_camera_axes(
        origin: Tuple[float, float, float],
        rotation_matrix: np.ndarray,
        ax: Axes3D,
        axis_length: float = 80.0,
    ) -> Axes3D:
        """
        Plot the camera axes on the given 3D axis.

        :param origin: The origin of the camera axes.
        :param rotation_matrix: The rotation matrix of the camera.
        :param ax: The 3D axis to plot on.
        :param axis_length: The length of the camera axes. Default is 80.0.
        :return: The 3D axis with the camera axes plotted.
        """
        axis_colors = ["red", "green", "blue"]

        for i in range(3):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                rotation_matrix[0, i],
                rotation_matrix[1, i],
                rotation_matrix[2, i],
                color=axis_colors[i],
                pivot="tail",
                length=axis_length,
            )

        return ax

    def plot_camera_movement(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray,
    ) -> Axes3D:
        """
        Plot the camera movement in a 3D plot.

        :param rotation_vecs: Array of rotation vectors.
        :param translation_vecs: Array of translation vectors.
        :param object_points: Array of object points.
        :return: The 3D plot axes.
        """
        ax = self.setup_3d_plot(
            x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[-1000, 0]
        )
        unit_vectors = np.eye(3).T
        camera_positions = np.zeros(translation_vecs.shape)

        for i in range(rotation_vecs.shape[1]):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
            camera_positions[:, i] = (
                -rotation_matrix.T @ translation_vecs[:, i]
            )
            rotated_axes = rotation_matrix.T @ unit_vectors
            ax = self.plot_camera_axes(
                camera_positions[:, i], rotated_axes, ax
            )

        ax.plot_wireframe(
            object_points[0], object_points[1], np.zeros_like(object_points[0])
        )

        return ax

    def plot_moving_pattern(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray,
    ) -> Axes3D:
        """
        Plot the moving pattern of a camera calibration.

        :param rotation_vecs: Array of rotation vectors.
        :param translation_vecs: Array of translation vectors.
        :param object_points: Array of object points.
        :return: The 3D plot axes.
        """
        ax = self.setup_3d_plot(
            x_lim=[-500, 500], y_lim=[-500, 500], z_lim=[0, 1000]
        )
        unit_vectors = np.eye(3)
        ax = self.plot_camera_axes([0, 0, 0], unit_vectors, ax)

        for i in range(rotation_vecs.shape[1]):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vecs[:, i])
            rotated_objp = (
                rotation_matrix @ object_points.T
                + translation_vecs[:, i].reshape(-1, 1)
            )
            ax.scatter(
                rotated_objp[0, :], rotated_objp[1, :], rotated_objp[2, :]
            )

        return ax

    def display_extrinsic_parameters(
        self,
        rotation_vecs: np.ndarray,
        translation_vecs: np.ndarray,
        object_points: np.ndarray,
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
        ax_movement = self.plot_camera_movement(
            rotation_vecs,
            translation_vecs,
            np.meshgrid(object_points[:, 0], object_points[:, 1]),
        )
        ax_movement.view_init(elev=-61, azim=-90)
        ax_movement._dist = 8
        plt.show()

        ax_pattern = self.plot_moving_pattern(
            rotation_vecs, translation_vecs, object_points
        )
        ax_pattern.view_init(elev=-45, azim=-90)
        ax_pattern._dist = 8

        plt.show()
