import cv2
import numpy as np
import glob
import os

def undistort_image(image_path, intrinsic_matrix, distortion_coeffs):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get the image size
    h, w = image.shape[:2]
    
    # Undistort the image
    undistorted_image = cv2.undistort(image, intrinsic_matrix, distortion_coeffs, None)
    
    # Show the original and undistorted images
    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return undistorted_image

def validate_calibration(object_points, image_points, rotation_vecs, translation_vecs, intrinsic_matrix, distortion_coeffs):
    """
    Validate camera calibration by reprojecting 3D object points onto the 2D image plane and comparing
    them with the original image points.

    Args:
        object_points (list): List of 3D object points.
        image_points (list): List of 2D image points.
        rotation_vecs (list): List of rotation vectors.
        translation_vecs (list): List of translation vectors.
        intrinsic_matrix (np.ndarray): Intrinsic matrix of the camera.
        distortion_coeffs (np.ndarray): Distortion coefficients of the camera.

    Returns:
        float: Mean reprojection error.
    """

    total_error = 0
    total_points = 0

    for i in range(len(object_points)):
        # Project 3D object points to the image plane
        projected_image_points, _ = cv2.projectPoints(object_points[i], rotation_vecs[i], translation_vecs[i], intrinsic_matrix, distortion_coeffs)
        
        # Calculate the error between the detected and reprojected points
        error = cv2.norm(image_points[i], projected_image_points, cv2.NORM_L2) / len(projected_image_points)
        total_error += error
        total_points += len(projected_image_points)

    mean_error = total_error / len(object_points)
    print(f"\nMean Reprojection Error: {mean_error}")

    return mean_error

def draw_axes(img: np.ndarray, corners: np.ndarray, imgpts: np.ndarray) -> np.ndarray:
    """Draw 3D axes on the image.

    Args:
        img (numpy.ndarray): The input image.
        corners (numpy.ndarray): The corners of the object in the image.
        imgpts (numpy.ndarray): The image points of the object.

    Returns:
        numpy.ndarray: The image with 3D axes drawn on it.
    """

    corner = tuple(corners[0].astype(int).ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (0,0,255), 5)  # X axis (red)
    img = cv2.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)  # Y axis (green)
    img = cv2.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (255,0,0), 5)  # Z axis (blue)

    # Draws point in the center of the image
    img = cv2.circle(img, corner, 10, (255, 255, 255), -1)

    # Writte the position of the point in the image
    cv2.putText(img, f'Pixel: ({corner[0]},{corner[1]})', corner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def load_calibration_data(calibration_file: str) -> tuple:
    """
    Load previously saved intrinsic and distortion coefficients.

    Parameters:
        calibration_file (str): The file path to the calibration data containing the 
        intrinsic_matrix, distortion_coeffs, rotation_matrix, and translation_vector.

    Returns:
        np.ndarray: intrinsic matrix
        np.ndarray: distortion coefficients
        np.ndarray: rotation matrix
        np.ndarray: translation vector
    """
    with np.load(calibration_file) as data:
        intrinsic_matrix = data['intrinsic_matrix']
        distortion_coeffs = data['distortion_coeffs']
        rotation_matrix = data['rotation_vecs']
        translation_vector = data['translation_vecs']
    return intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_vector

def camera_model(par: str, K: np.ndarray, PI: np.ndarray, position: np.ndarray, scale: int = 1) -> np.ndarray:
    """
    Calculates the projection matrix given the intrinsic matrix, rotation matrix, and translation vector.
    Args:
        par (str): 'pixel' or 'position'  What is be find, position in world or pixel in image.
        k (np.ndarray): The intrinsic matrix of the camera.
        PI (np.ndarray): The projection matrix
        position (np.ndarray): [x,y,1] or [X,Y,Z,1] The position of the object in the camera's coordinate system or in world.
        
    Returns:
        np.ndarray: Position.
    """
    if par == 'pixel':
        return (K @ PI @ position)/ scale
    elif par == 'position':
        return np.linalg.pinv(K @ PI) @ position * scale
    else:
        print("Invalid parameter. Please enter 'pixel' or 'position'.")
        return None

def general_projection_matrix(K: np.ndarray, PI: np.ndarray, scale_factor: int, pixel_pos: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
    """
    Calculates the extrinssic matrix given the intrinsic matrix
    
    Args:
        K (np.ndarray): The intrinsic matrix of the camera.
        PI (np.ndarray): The projection matrix
        scale_factor (int): The scale factor of the 2D pivel vector.
        pixel_pos (np.ndarray): [x,y,1] The pixel position of the object in the image's coordinate system.
        world_pos (np.ndarray): [X,Y,Z,1] The world position of the object in the camera's coordinate system.
    Output:
        np.ndarray: The extrinsic matrix.

    Description:
        scale_factor * 2D_point  =     K     *      PI    *   E   *   3D_point
        scale_factor * [x,y,1].T = intrinsic * [eye(3)|0] * [R|t] * [X,Y,Z,1].T
          int        *   3x1     =    3x3    *     3x4    *  4x4  *    4x1
    """
    
    aux1 = K @ PI
    aux2 = np.linalg.pinv(aux1)
    aux3 = scale_factor * (aux2 @ np.expand_dims(pixel_pos,1))
    aux4 = np.linalg.pinv(np.expand_dims(world_pos,1))
    E = aux3 @ aux4
    return E

def prepare_object_points(square_size: float, pattern_size: tuple) -> np.ndarray:
    """Prepare 3D object points based on the chessboard pattern size.

    Args:
        square_size (float): The size of each square in the chessboard pattern.
        pattern_size (tuple): The size of the chessboard pattern in terms of number of corners.

    Returns:
        numpy.ndarray: The 3D object points representing the corners of the chessboard pattern.
    """

    objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0]*square_size:square_size, 
                            0:pattern_size[1]*square_size:square_size].T.reshape(-1, 2)
    return objp

def process_images(image_paths: list, num_images: int, pattern_size: tuple, objp: np.ndarray, criteria: tuple, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray, axis: np.ndarray, show: bool = False) -> tuple:
    """
    Process each image, find corners, and draw 3D axes.
    
    Args:
        image_paths (list): List of paths to the input images.
        pattern_size (tuple): Size of the chessboard pattern (width, height).
        objp (ndarray): 3D coordinates of the chessboard corners.
        criteria (tuple): Criteria for corner refinement.
        intrinsic_matrix (ndarray): Camera intrinsic matrix.
        distortion_coeffs (ndarray): Camera distortion coefficients.
        axis (ndarray): 3D coordinates of the axes.
    
    Returns:
        tuple: Object points, image points, rotation vectors, translation vectors.
    """
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane
    rvecs = []  # Rotation vectors
    tvecs = []  # Translation vectors

    for idx, fname in enumerate(image_paths):
        if idx % int(len(image_paths)/num_images) == 0:
            print(f"Processing image {idx}")

            img = cv2.imread(fname)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (pattern_size[0], pattern_size[1]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

            if ret:
                objpoints.append(objp)
                cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # SolvePnP to find the rotation and translation vectors
                ret, rvecs_i, tvecs_i = cv2.solvePnP(objp, corners, intrinsic_matrix, distortion_coeffs)
                rvecs.append(rvecs_i)
                tvecs.append(tvecs_i)

                # Project the 3D points to the image plane
                imgpts, _ = cv2.projectPoints(axis, rvecs_i, tvecs_i, intrinsic_matrix, distortion_coeffs)

                # Draw the axes on the image
                img = draw_axes(img, corners, imgpts)

                # Save the image with the drawn axes
                # filename = os.path.join("output_images", f"result_{idx}.png")
                if show:
                    cv2.imshow("output_images", img)
                    cv2.waitKey(100)
    
    cv2.destroyAllWindows()

    return objpoints, imgpoints, rvecs, tvecs

def main():
    # Load calibration data
    calibration_file = 'Projects_GE_VIO/camera_calibration/calibration_result_2608.npz'
    intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_vector = load_calibration_data(calibration_file)
    
    # Path to calibration images
    image_dir = 'Projects_GE_VIO/camera_calibration/dataset_images_uav/*.png'
    image_paths = glob.glob(image_dir)

    # Number of images to process and the size of the chessboard pattern
    num_images = 200
    pattern_size = (7,10)

    # Prepare object points and criteria
    objp = prepare_object_points(22, pattern_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Define the axis for the 3D points
    axis_length = 3  # Adjust the length of the axes as needed
    axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)

    # Validate calibration
    objpoints, imgpoints, rvecs, tvecs = process_images(image_paths, num_images, pattern_size, objp, criteria, intrinsic_matrix, distortion_coeffs, axis)
    validate_calibration(objpoints, imgpoints, rvecs, tvecs, intrinsic_matrix, distortion_coeffs)
    
    # # Undistort image
    # image_path = "test_image.png"
    # undistorted_image = undistort_image(image_path, intrinsic_matrix, distortion_coeffs)

if __name__ == "__main__":
    main()
