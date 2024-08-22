import cv2
import numpy as np
import glob
import os

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

def process_images(image_paths: list, pattern_size: tuple, objp: np.ndarray, criteria: tuple, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray, axis: np.ndarray) -> None:
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
        None
    Raises:
        None
    """
    for idx, fname in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"Processing image {idx}")

            img = cv2.imread(fname)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (pattern_size[0], pattern_size[1]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS) #PDI

            cv2.imshow('img', img) # Deletar

            if ret:
                corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors.
                _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, intrinsic_matrix, distortion_coeffs)

                # Project 3D points to image plane
                imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, intrinsic_matrix, distortion_coeffs)
                img2 = draw_axes(img, corners2, imgpts)
                # extrinsic_matrix = general_projection_matrix(intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))), 1, np.array([int(corners2[0][0][0]), int(corners2[0][0][1]), 1], dtype=np.int32), np.array([-110,-66,30,1]).T)

                # Show the image with 3D axes
                cv2.imshow('img2', img2)
                if cv2.waitKey(100) & 0xFF == ord('s'):
                    cv2.imwrite(f"output_{idx}.png", img)

                # Calculate the position of the board in the camera's coordinate system
                # lambda * 2D_point  =     K     *      pi    *   E   *   3D_point
                # lambda * [x,y,1].T = intrinsic * [eye(3)|0] * [R|t] * [X,Y,Z,1].T
                # scalar *   3x1     =    3x3    *     3x4    *  4x4  *    4x1
                
                # pixel_position = camera_model('pixel', intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))), np.array([0,0,30,1]).T)
                # world_position = camera_model('position', intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))), corners)
                # print(f"\nPixel position: {pixel_position.ravel()}")
                # print(f"World position: {world_position.ravel()}")
                # board_position = calculate_chessboard_position(corners2[0][0], intrinsic_matrix, 1)
                # print(f"Board position in camera coordinate system (image {idx}): \n{board_position.ravel()}")
                
                

    cv2.destroyAllWindows()

def main() -> None:
    # Main script execution
    path_main = os.path.dirname(__file__)
    calibration_file = os.path.join(path_main, 'calibration_result_16-08.npz')


    intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_vector = load_calibration_data(calibration_file)
    print(f"\nIntrinsic Matrix:\n {intrinsic_matrix}")
    
    # Termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard parameters
    square_size = 22  # Size of a square on the chessboard in your units (e.g., millimeters)
    chessboard_size = (10, 7)  # Number of inner corners per chessboard row and column (l, c)
    
    # Prepare object points
    objp = prepare_object_points(square_size, chessboard_size)

    # Define the axis size for plotting on the image
    axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]]).reshape(-1, 3)

    # Load images
    image_dir = os.path.join(path_main, 'dataset_images')
    images = glob.glob(os.path.join(image_dir, '*.png'))
    print(f'\nNumber of images read: {len(images)}\n')

    # Process images
    process_images(images, chessboard_size, objp, criteria, intrinsic_matrix, distortion_coeffs, axis)

if __name__ == "__main__":
    main()
