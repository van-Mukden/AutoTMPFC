import os
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

class Skeletonizer:
    def __init__(self, segment_img_path, skeleton_img_path):
        """
        Initialize the Skeletonizer class.

        Args:
            segment_img_path (str): Path to the folder containing segmented frames.
            skeleton_img_path (str): Path to the folder where skeletonized images will be saved.
        """
        self.segment_img_path = segment_img_path
        self.skeleton_img_path = skeleton_img_path
        os.makedirs(self.skeleton_img_path, exist_ok=True)

    def process_and_skeletonize_frames(self):
        """
        Process all segmented frames, skeletonize non-red colorful regions, and save the results.
        """
        for filename in sorted(os.listdir(self.segment_img_path)):
            if filename.endswith(".png"):
                img_path = os.path.join(self.segment_img_path, filename)
                skeleton = self.skeletonize_frame(img_path)

                # Save the skeletonized image
                skeleton_filename = os.path.join(self.skeleton_img_path, filename)
                cv2.imwrite(skeleton_filename, skeleton)
                print(f"Saved skeletonized frame: {skeleton_filename}")

    def skeletonize_frame(self, img_path):
        """
        Skeletonize the non-red, colorful pixels in a segmented frame.

        Args:
            img_path (str): Path to the segmented image file.

        Returns:
            numpy.ndarray: Skeletonized image of the colorful structures.
        """
        # Step 1: Load the image and convert to HSV
        pil_image = Image.open(img_path)
        image = np.array(pil_image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Step 2: Create a mask for red pixels (to exclude them)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(hsv_image, lower_red2, upper_red2)

        # Step 3: Identify non-red colorful areas based on HSV saturation value
        saturation_threshold = 50
        colorful_mask = (hsv_image[:, :, 1] > saturation_threshold) & (red_mask == 0)

        # Step 4: Convert to binary format for skeletonization
        binary_bool = colorful_mask.astype(np.uint8) * 255  # Convert to 8-bit format for cv2 operations

        # Step 5: Apply Gaussian blur to reduce noise before skeletonization
        blurred = cv2.GaussianBlur(binary_bool, (5, 5), 0)

        # Step 6: Skeletonize using a thinning method
        skeleton = cv2.ximgproc.thinning(blurred)

        return skeleton

    def analyze_skeleton_properties(self):
        """
        Analyze all skeletonized frames and print the total length and number of branches.

        Returns:
            dict: Dictionary with frame filenames as keys and a tuple of (length, branches) as values.
        """
        skeleton_properties = {}
        for filename in sorted(os.listdir(self.skeleton_img_path)):
            if filename.endswith(".png"):
                skeleton_img = cv2.imread(os.path.join(self.skeleton_img_path, filename), cv2.IMREAD_GRAYSCALE)
                length = self.get_skeleton_length(skeleton_img)

                skeleton_properties[filename] = length
                print(f"Frame {filename}: Length = {length}")

        return skeleton_properties

    def get_skeleton_length(self, skeleton_img):
        """
        Calculate the total length of the skeleton by counting non-zero pixels.

        Args:
            skeleton_img (numpy.ndarray): Binary skeleton image.

        Returns:
            int: Total length of the skeleton.
        """
        return np.count_nonzero(skeleton_img)

# Usage example:
# skeletonizer = Skeletonizer('/path/to/segmented_frames', '/path/to/save/skeletons')
# skeletonizer.process_and_skeletonize_frames()
# skeleton_properties = skeletonizer.analyze_skeleton_properties()
