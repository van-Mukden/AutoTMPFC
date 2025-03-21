
'''
# Example usage:
dicom_path = 'path/to/your/dicom_file.dcm'
output_dir = 'path/to/your/output_folder'
model_path = 'path/to/your/model.pt'
output_video_path = os.path.join(output_dir, "segmented_video.mp4")
processor = DicomVideoPreprocessor(dicom_path, output_dir, model_path)
processor.extract_frames_from_dicom()
df = processor.analyze_and_visualize()
processor.save_segmented_video(output_video_path)
'''

import os
import re
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from matplotlib import pyplot as plt

class DicomVideoPreprocessor:
    def __init__(self, dicom_path, output_dir, model_path, ref_number):
        """
        Initialize the DicomVideoPreprocessor.

        Args:
            dicom_path (str): Path to the input DICOM file.
            output_dir (str): Directory to save output files.
            model_path (str): Path to the segmentation model file.
            ref_number (str): Reference number for naming folders.
        """
        self.dicom_path = dicom_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(self.model_path).to(self.device).eval()

        # Determine whether this is Left or Right Coronary based on folder name
        coronary_type = "LCA" if "Left" in dicom_path else "RCA"

        # Set up output directories with reference number and coronary type suffix
        self.extract_img_path = os.path.join(output_dir, f"extract_frames_{ref_number}_{coronary_type}")
        self.segment_img_path = os.path.join(output_dir, f"segment_frames_{ref_number}_{coronary_type}")
        os.makedirs(self.extract_img_path, exist_ok=True)
        os.makedirs(self.segment_img_path, exist_ok=True)

    def extract_frames_from_dicom(self):
        """
        Extract individual frames from the DICOM file and save them as PNG images.
        """
        ds = pydicom.dcmread(self.dicom_path)
        frames = ds.pixel_array
        for i, frame in enumerate(frames):
            image = Image.fromarray(frame)
            frame_filename = f"frame_{i+1:04d}.png"
            image.save(os.path.join(self.extract_img_path, frame_filename))
            print(f"Saved {frame_filename}")
        print(f"Extracted and saved {len(frames)} frames.")

    def process_and_visualize(self, img_path):
        """
        Process a single image frame, apply segmentation, and visualize the result.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            numpy.ndarray: Processed and visualized image with segmentation overlay.
        """
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError(f"Image could not be loaded from {img_path}")
        img_preprocessed = self.preprocess_image(original_img)
        mask = self.run_inference(torch.from_numpy(img_preprocessed).float())

        result = self.overlay_mask_on_image(original_img, mask)
        return result

    def preprocess_image(self, img):
        """
        Preprocess the input image for segmentation.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Preprocessed image stack.
        """
        img_sharp = self.unsharp_masking(img)
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe2 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        img_clahe1 = clahe1.apply(img_sharp)
        img_clahe2 = clahe2.apply(img_sharp)
        return np.stack((img_sharp / 255.0, img_clahe1 / 255.0, img_clahe2 / 255.0), axis=0)

    def unsharp_masking(self, img):
        """
        Apply unsharp masking to enhance image details.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Sharpened image.
        """
        if img is None:
            raise ValueError("Image for unsharp masking is None")
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)

    def run_inference(self, img_tensor):
        """
        Run segmentation inference on the preprocessed image.

        Args:
            img_tensor (torch.Tensor): Preprocessed image tensor.

        Returns:
            numpy.ndarray: Binary segmentation mask.
        """
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            mask = (probs[:, 1, :, :] > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        return mask

    def overlay_mask_on_image(self, original_img, mask):
        """
        Overlay the segmentation mask on the original image.

        Args:
            original_img (numpy.ndarray): Original grayscale image.
            mask (numpy.ndarray): Binary segmentation mask.

        Returns:
            numpy.ndarray: Image with segmentation overlay.
        """
        # Convert grayscale image to BGR
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # Create a red color for the arteries
        red_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR format

        # Create the red arteries
        red_arteries = np.where(mask[:, :, None] == 0, red_color, original_img_bgr)

        # Blend the red arteries with the original image
        result = cv2.addWeighted(original_img_bgr, 0.7, red_arteries, 0.3, 0)

        return result

    def analyze_and_visualize(self):
        """
        Process all extracted frames, apply segmentation, and generate visualization data.

        Returns:
            pandas.DataFrame: DataFrame containing frame numbers and red pixel counts.
        """
        results = []
        for filename in sorted(os.listdir(self.extract_img_path)):
            if filename.endswith(".png"):
                img_path = os.path.join(self.extract_img_path, filename)
                processed_img = self.process_and_visualize(img_path)

                frame_number = int(filename.split('_')[1].split('.')[0])
                save_frame_path = os.path.join(self.segment_img_path, f"frame_{frame_number:04d}.png")
                cv2.imwrite(save_frame_path, processed_img)

                red_pixel_count = self.count_red_pixels(processed_img)
                results.append({"Frame Number": frame_number, "Red Pixels": red_pixel_count})
                print(f"Frame {frame_number}: {red_pixel_count} red pixels")

        df = pd.DataFrame(results)

        # Save CSV
        csv_path = os.path.join(self.output_dir, "red_pixel_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Red pixel data saved to {csv_path}")

        # Save plot
        self.plot_results(df)

        return df


    def count_red_pixels(self, img):
        """
        Count the number of red pixels in the segmented image.

        Args:
            img (numpy.ndarray): Input BGR image.

        Returns:
            int: Number of red pixels.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lower_red = (150, 0, 0)
        upper_red = (255, 100, 100)
        mask = cv2.inRange(img_rgb, lower_red, upper_red)
        return cv2.countNonZero(mask)

    def plot_results(self, df):
        """
        Plot the red pixel count per frame and save it as an image.

        Args:
            df (pandas.DataFrame): DataFrame containing frame numbers and red pixel counts.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df['Frame Number'], df['Red Pixels'], marker='o')
        plt.title('Red Pixel Count per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Red Pixels')
        plt.xticks(ticks=df['Frame Number'][::3], labels=df['Frame Number'][::3])
        plt.grid(True, linestyle='--', linewidth=0.5)

        plot_path = os.path.join(self.output_dir, "red_pixel_plot.png")
        plt.savefig(plot_path)
        print(f"Red pixel plot saved to {plot_path}")
        plt.close()

    def save_segmented_video(self, output_video_path):
        """
        Create a video from the segmented frames.

        Args:
            output_video_path (str): Path to save the output video file.
        """
        frame = cv2.imread(os.path.join(self.segment_img_path, sorted(os.listdir(self.segment_img_path))[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

        for filename in sorted(os.listdir(self.segment_img_path)):
            if filename.endswith(".png"):
                frame = cv2.imread(os.path.join(self.segment_img_path, filename))
                video.write(frame)

        video.release()
        print(f"Video saved at {output_video_path}")


    @staticmethod
    def process_folder(input_folder, output_base_dir, model_path):
        """
        Process all DICOM files in the given folder.

        Args:
            input_folder (str): Path to the folder containing DICOM files.
            output_base_dir (str): Base directory to save output files for all videos.
            model_path (str): Path to the segmentation model file.

        Returns:
            dict: A dictionary with DICOM filenames as keys and their respective DataFrames as values.
        """
        results = {}
        for filename in os.listdir(input_folder):
            if filename.endswith('.dcm'):
                dicom_path = os.path.join(input_folder, filename)
                output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])
                output_video_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segmented.mp4")

                processor = DicomVideoPreprocessor(dicom_path, output_dir, model_path)
                processor.extract_frames_from_dicom()
                df = processor.analyze_and_visualize()
                processor.save_segmented_video(output_video_path)

                results[filename] = df
                print(f"Processed {filename}")

        return results
