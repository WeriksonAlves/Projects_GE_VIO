# Camera Resectioning

**Camera Resectioning Overview**  
Camera resectioning is the process of estimating the parameters of a pinhole camera model to determine the association between incoming light rays and image pixels. This process is key in applications such as stereo vision and 3D reconstruction, where understanding the camera's pose (position and orientation) and its internal settings (focal length, pixel size, etc.) is crucial.

**Camera Parameters**  
The parameters of a camera are often encapsulated in a 3 × 4 projection matrix known as the camera matrix. This matrix combines both intrinsic and extrinsic parameters:

- **Intrinsic Parameters**: Describe the camera's internal characteristics, including focal length, pixel dimensions, and the principal point (the image center). These are typically represented in a matrix \(K\), which also includes a skew coefficient.
  
- **Extrinsic Parameters**: Define the camera's pose relative to the world coordinate system. This includes a 3 × 3 rotation matrix \(R\) and a translation vector \(T\).

**Projection and Homogeneous Coordinates**  
In camera resectioning, 2D image points and 3D world points are represented using homogeneous coordinates. The projection matrix \(M\) is used to map 3D world coordinates to 2D pixel coordinates. This process can be expressed mathematically as:

\[ M = K [R | T] \]

Where:
- \(K\) is the intrinsic matrix.
- \([R | T]\) is the extrinsic matrix.

The image coordinates \(u\) and \(v\) are derived by applying the projection matrix to the world coordinates.

**Application in Stereo Vision**  
Camera resectioning is critical in stereo vision, where the projection matrices of two cameras are used to calculate the 3D coordinates of points seen by both cameras.

**Nonlinear Parameters**  
In addition to the linear intrinsic parameters, nonlinear factors like lens distortion are also important. These are usually estimated through optimization techniques such as bundle adjustment.

**Importance in Computer Vision**  
Camera calibration, which includes both resectioning and distortion estimation, is a foundational step in many computer vision tasks. By understanding the relationship between the 3D world and its 2D projection, various applications such as object tracking, 3D reconstruction, and navigation become possible.

---

# Bebop2 Camera Calibration using Python

This repository provides scripts to estimate the intrinsic and extrinsic parameters of the Bebop2 camera using Visual-Inertial Odometry (VIO) and Python.

## Overview

### 1. **Intrinsic Parameters Estimation**
   - **Objective**: Determine the focal length, optical center, and lens distortion coefficients.
   - **Method**: Capture multiple images of a checkerboard pattern and use OpenCV functions (`cv2.calibrateCamera`) for calibration.
   - **Output**: Intrinsic matrix and distortion coefficients.

### 2. **Extrinsic Parameters Estimation**
   - **Objective**: Estimate the rotation and translation between the camera and the IMU.
   - **Method**: Utilize rotation and translation vectors from intrinsic calibration and refine using optimization techniques.
   - **Output**: Extrinsic matrix.

## Usage

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Capture calibration images**: Use a checkerboard pattern.
3. **Run calibration**: Execute the provided scripts to compute the intrinsic and extrinsic parameters.
4. **Verify**: Use test images to assess the calibration quality by calculating the reprojection error.

---
