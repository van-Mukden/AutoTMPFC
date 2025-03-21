# process_longest_path_overlay.py
import os
from classes.longest_path import LongestPathOverlay, store_longest_path_csv

def process_longest_path_overlay(group_dir):
    """
    Process all patient folders within a given group directory.
    For each patient, process images from the "completed_skeletons" folder
    by overlaying the longest path and store a CSV file with the path lengths
    in the patient folder.

    Args:
        group_dir (str): Path to the group directory containing patient folders.
    """
    for patient_folder in os.listdir(group_dir):
        patient_path = os.path.join(group_dir, patient_folder)
        if os.path.isdir(patient_path):
            print(f"\nProcessing longest path overlay for patient: {patient_folder}")

            completed_skeletons_path = os.path.join(patient_path, "completed_skeletons")
            if os.path.isdir(completed_skeletons_path):
                longest_path_data = []  # Collect (filename, longest_path_length) tuples.
                for filename in sorted(os.listdir(completed_skeletons_path)):
                    if filename.endswith(".png"):
                        image_path = os.path.join(completed_skeletons_path, filename)
                        try:
                            overlay_processor = LongestPathOverlay(image_path)
                            overlay_processor.process()
                            longest_path_data.append((filename, overlay_processor.max_length))
                        except Exception as e:
                            print(f"Error processing image {image_path}: {e}")
                if longest_path_data:
                    # Save the CSV file in the patient folder.
                    store_longest_path_csv(patient_path, longest_path_data)
            else:
                print(f"  No 'completed_skeletons' folder found in {patient_path}.")