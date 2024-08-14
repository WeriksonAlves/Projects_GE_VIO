import cv2
import numpy as np
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

#%% Intrinsic matrix

images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"Intrinsic matrix: \n{mtx}\n")
print(f"Distortion coefficients: \n{dist}\n")

#%% Extrinsic matrix

# Example: Converting rotation vectors to a rotation matrix
rvec = rvecs[0]  # Example: using the first rotation vector
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Extrinsic matrix
extrinsic_matrix = np.hstack((rotation_matrix, tvecs[0]))
print(f"Extrinsic matrix: \n{extrinsic_matrix}\n")
