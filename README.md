# UAV Camera Calibration System

This repository contains routines and scripts to perform camera calibration for UAVs, specifically designed for the Parrot Bebop2 drone. The project enables accurate calibration of the camera's intrinsic and extrinsic parameters, facilitating reliable navigation and pose estimation in real-world environments.

## Features

- **Image Capture**: Capture images directly from the drone's camera for calibration purposes.
- **Intrinsic Calibration**: Calculate the intrinsic matrix and distortion coefficients of the camera.
- **Extrinsic Calibration**: Determine the camera's pose (rotation and translation) relative to a known object pattern.
- **Calibration Validation**: Validate the calibration results by comparing the re-projected points with the original image points.

## Prerequisites

- Python 3.9
- NumPy
- OpenCV
- Matplotlib
- PyParrot (for drone communication)

## Installation

1. Clone this repository

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the project structure:
    - Place your drone camera images in the `datasets/` directory.
    - Ensure the directory structure is maintained as follows:
      ```
      UAV-Camera-Calibration/
      ├── datasets/
      │   └── uav_B6_1/
      ├── results/
      └── scripts/
      ```

## File Structure

- **`modules/calibration_system.py`**: Core module containing the calibration logic.
- **`datasets/`**: Directory to store images captured by the drone.
- **`results/`**: Directory where calibration results (intrinsic matrix, distortion coefficients, etc.) are saved.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.