from src.config import DATA_DIR, get_preprocessed_dir, MODEL_PATH
from classes.dicom_video_preprocessor import DicomVideoPreprocessor
import os
def process_dicom_videos():
    """Process all patients' DICOM videos."""
    # Traverse each patient folder
    for patient_folder in os.listdir(DATA_DIR):
        patient_path = os.path.join(DATA_DIR, patient_folder)

        # Check if it's a valid patient folder
        if os.path.isdir(patient_path):
            print(f"Processing patient: {patient_folder}")

            # Traverse Right and Left Coronary folders
            for coronary_folder in os.listdir(patient_path):
                coronary_path = os.path.join(patient_path, coronary_folder)

                if os.path.isdir(coronary_path) and "Coronary" in coronary_folder:
                    coronary_type = "LCA" if "Left" in coronary_folder else "RCA"

                    # Create preprocessed directory
                    preprocessed_dir = get_preprocessed_dir(patient_folder, coronary_type)
                    os.makedirs(preprocessed_dir, exist_ok=True)

                    # Process DICOM files
                    for file in os.listdir(coronary_path):
                        if file.endswith(".dcm"):
                            dicom_path = os.path.join(coronary_path, file)
                            print(f"  Found DICOM file: {dicom_path}")

                            # Initialize and process using DicomVideoPreprocessor
                            processor = DicomVideoPreprocessor(
                                dicom_path=dicom_path,
                                output_dir=preprocessed_dir,
                                model_path=MODEL_PATH,
                                ref_number=patient_folder
                            )
                            processor.extract_frames_from_dicom()
                            df = processor.analyze_and_visualize()
if __name__ == "__main__":
    process_dicom_videos()