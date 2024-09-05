import cv2
import glob
import os

from pyparrot.Bebop import Bebop

from modules.calibration.Camera import Camera
from modules.mainRoutines import run_camera_calibration, run_feature_matching, run_pose_estimation


# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets/calibration/board_B6_11")
RESULT_FILE = os.path.join(BASE_DIR, "results/calibration/B6_1.npz")

IMAGE_FILES = sorted(glob.glob(os.path.join(BASE_DIR, "datasets/matching/images/*.png")))

# Modo de operação
MODES = ["camera_calibration", "feature_matching", "pose_estimation"]
mode = 1
img1 = IMAGE_FILES[4]
img2 = IMAGE_FILES[5]

# Inicialização do drone Bebop e da câmera
bebop_drone = Bebop()
camera = Camera(
    uav=bebop_drone,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
)

# Escolha do modo de operação
if MODES[mode] == "camera_calibration":
    run_camera_calibration(
        camera=camera,
        dataset_path=DATASET_PATH,
        result_file=RESULT_FILE,
        attempts=100,
        save=False,
        num_images=50,
        display=False
    )
elif MODES[mode] == "feature_matching":
    run_feature_matching(
        img1,
        img2
    )
elif MODES[mode] == "pose_estimation":
    run_pose_estimation()

# Finaliza o programa ao pressionar 'q'
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
    cv2.waitKey(100)
