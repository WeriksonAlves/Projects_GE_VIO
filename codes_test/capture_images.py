from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from enum import Enum, auto
import cv2

isAlive = False

class Model(Enum):
    BEBOP = auto()
    MAMBO = auto()
    ANAFI = auto()

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        img = self.vision.get_latest_valid_picture()

        if img is not None:
            cv2.imshow('Captured Image', img)
            cv2.waitKey(1)
            filename = f"calibration_image_{self.index:05d}.png"
            # cv2.imwrite(filename, img)
            # print(f"Saved {filename}")
            self.index += 1

# Initialize the Bebop object
bebop = Bebop()

# Connect to the Bebop
success = bebop.connect(100)

if success:
    # Start up the video
    bebopVision = DroneVision(bebop, Model.BEBOP, buffer_size=10)

    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    
    success = bebopVision.open_video()
    
    print(f"\nGet ready to capture images for calibration. You have 10 seconds to prepare.\n")
    bebop.smart_sleep(10)
    
    if success:
        print("Vision successfully started!")

        # Capture images for a specified duration
        capture_duration = 30  # seconds
        print(f"Move the drone around and hold the pattern in front of the camera for {capture_duration} seconds.")
        bebop.smart_sleep(capture_duration)

        print("Finishing and stopping vision")
        bebopVision.close_video()

    # Disconnect nicely
    bebop.disconnect()
else:
    print("Error connecting to Bebop. Please retry.")
