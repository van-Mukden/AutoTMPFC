# optical_analysis.py
import os
from classes.optical_flow import RedPixelFlowAnalyzer
from src.config import (
    EPS, MIN_SAMPLES, STEP
)

def process_optical_analysis(group_dir):
    """
    Perform optical flow analysis on all patient folders within a group directory.

    Args:
        group_dir (str): Path to the group directory containing patient folders.
    """
    # Process each patient folder in the group
    for patient_folder in os.listdir(group_dir):
        patient_path = os.path.join(group_dir, patient_folder)

        if os.path.isdir(patient_path):
            print(f"Processing optical flow for patient: {patient_folder}")

            # Look for "segment_frames" folder inside the patient's preprocessed directories
            segment_frames_dir = None
            for subfolder in os.listdir(patient_path):
                if subfolder.startswith("preprocessed_"):
                    potential_segment_frames_dir = os.path.join(patient_path, subfolder, f"segment_frames_{patient_folder}_{subfolder.split('_')[1]}")
                    if os.path.isdir(potential_segment_frames_dir):
                        segment_frames_dir = potential_segment_frames_dir
                        break

            if not segment_frames_dir:
                print(f"  No segment_frames folder found for patient {patient_folder}. Skipping.")
                continue

            # Define output directories for this patient
            flow_arrows_dir = os.path.join(patient_path, "optical_flow_analysis")
            clustered_frames_dir = os.path.join(patient_path, "cluster")
            flow_data_csv = os.path.join(patient_path, "flow.csv")
            cluster_points_csv = os.path.join(patient_path, "clustered_points.csv")

            # Ensure directories exist
            os.makedirs(flow_arrows_dir, exist_ok=True)
            os.makedirs(clustered_frames_dir, exist_ok=True)

            # Create an instance of the RedPixelFlowAnalyzer
            analyzer = RedPixelFlowAnalyzer(
                frame_folder=segment_frames_dir,
                flow_arrows_dir=flow_arrows_dir,
                flow_data_csv=flow_data_csv,
                clustered_frames_dir=clustered_frames_dir,
                cluster_points_csv=cluster_points_csv,
                eps=EPS,
                min_samples=MIN_SAMPLES,
                step=STEP,
            )

            # Run optical flow analysis
            analyzer.calculate_flow()
            print(f"  Optical flow analysis completed for patient {patient_folder}.")