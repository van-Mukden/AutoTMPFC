import os
import pandas as pd
from classes.feature_extraction import FeatureExtractor

def demo_feature_extraction(patient_folder):

    fe = FeatureExtractor(patient_folder)
    features_df = fe.extract_features()

    if features_df is not None and not features_df.empty:
        patient_id = os.path.basename(os.path.normpath(patient_folder))
        print(f"[{patient_id}] Extracted features shape: {features_df.shape}")
        print(features_df.head(10))

        # Output directory
        output_dir = os.path.join("data", "feature", patient_id)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{patient_id}_feature.csv")

        features_df.to_csv(output_file, index=False)
        print(f"Features saved => {output_file}")
    else:
        print(f"[{patient_folder}] No features or empty DataFrame")

if __name__ == "__main__":
    # Assume all your patient folders are located in data/all_data
    base_all_data = os.path.join("data", "all_data")

    for patient_id in os.listdir(base_all_data):
        patient_folder = os.path.join(base_all_data, patient_id)
        if os.path.isdir(patient_folder):
            print(f"\nProcessing patient: {patient_id}")
            demo_feature_extraction(patient_folder)
