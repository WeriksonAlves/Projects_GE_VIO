from pyparrot.DroneVision import DroneVision
import os
import cv2

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
