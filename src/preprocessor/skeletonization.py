# skeletonization.py
import os
import pandas as pd
from classes.skeletonization import Skeletonizer


def process_skeletonization(group_dir):
    """
    Perform skeletonization on all patient folders within a group directory.

    Args:
        group_dir (str): Path to the group directory containing patient folders.
    """
    # Process each patient folder in the group
    for patient_folder in os.listdir(group_dir):
        patient_path = os.path.join(group_dir, patient_folder)

        if os.path.isdir(patient_path):
            print(f"Processing skeletonization for patient: {patient_folder}")

            # Look for "cluster" folder inside the patient's directory
            cluster_folder_path = os.path.join(patient_path, "cluster")
            if os.path.isdir(cluster_folder_path):
                # Define the output path for skeletonized frames
                skeleton_img_path = os.path.join(patient_path, "skeleton_frames")
                os.makedirs(skeleton_img_path, exist_ok=True)

                # Create an instance of Skeletonizer and process the frames
                skeletonizer = Skeletonizer(cluster_folder_path, skeleton_img_path)
                skeletonizer.process_and_skeletonize_frames()

                # Analyze skeleton properties
                skeleton_properties = skeletonizer.analyze_skeleton_properties()

                # Convert the skeleton properties dictionary to a DataFrame
                skeleton_properties_df = pd.DataFrame.from_dict(
                    skeleton_properties, orient='index', columns=['Length']
                ).reset_index().rename(columns={'index': 'Frame'})

                # Save the DataFrame to a CSV file within the patient's directory
                skeleton_properties_csv = os.path.join(patient_path, "skeleton_properties.csv")
                skeleton_properties_df.to_csv(skeleton_properties_csv, index=False)
                print(f"  Skeleton properties saved to {skeleton_properties_csv}")