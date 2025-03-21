import cv2
import numpy as np
import matplotlib.pyplot as plt

class SkeletonFiller:
    def __init__(self, image_path, kernel_size=8, iterations=3, thickness=1):
        self.image_path = image_path
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.thickness = thickness
        self.skeleton = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.completed_skeleton = None

    def process_image(self):
        # Threshold the image to binary
        _, binary = cv2.threshold(self.skeleton, 127, 255, cv2.THRESH_BINARY)

        # Define a kernel for morphological operations
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        # Perform dilation to connect nearby skeleton segments
        dilated = cv2.dilate(binary, kernel, iterations=self.iterations)

        # Perform morphological closing to bridge small gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Thin the skeleton again if necessary
        thinned = cv2.ximgproc.thinning(closed)

        # Additional dilation to make the skeleton thicker
        if self.thickness > 1:
            thick_kernel = np.ones((self.thickness, self.thickness), np.uint8)
            self.completed_skeleton = cv2.dilate(thinned, thick_kernel, iterations=1)
        else:
            self.completed_skeleton = thinned

    def show_images(self):
        if self.completed_skeleton is None:
            print("Image has not been processed. Please call process_image() first.")
            return

        # Display the original and processed images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Skeleton")
        plt.imshow(cv2.threshold(self.skeleton, 127, 255, cv2.THRESH_BINARY)[1], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Completed and Thickened Skeleton")
        plt.imshow(self.completed_skeleton, cmap='gray')
        plt.show()

    def save_image(self, output_path):
        if self.completed_skeleton is None:
            print("Image has not been processed. Please call process_image() first.")
            return

        cv2.imwrite(output_path, self.completed_skeleton)
        print(f"Processed image saved to {output_path}")