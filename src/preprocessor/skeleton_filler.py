import os
from classes.skeleton_filler import SkeletonFiller


def process_skeleton_filling(group_dir):
    """
    Perform skeleton filling on all patient folders within a group directory.

    Args:
        group_dir (str): Path to the group directory containing patient folders.
    """
    kernel_size = 8
    iterations = 6
    thickness = 2

    # Process each patient folder in the group
    for patient_folder in os.listdir(group_dir):
        patient_path = os.path.join(group_dir, patient_folder)

        if os.path.isdir(patient_path):
            print(f"Processing skeleton filling for patient: {patient_folder}")

            # Look for "skeleton_frames" folder inside the patient's directory
            skeleton_frames_path = os.path.join(patient_path, "skeleton_frames")
            if os.path.isdir(skeleton_frames_path):
                # Define the output path for completed skeleton frames
                completed_skeleton_path = os.path.join(patient_path, "completed_skeletons")
                os.makedirs(completed_skeleton_path, exist_ok=True)

                # Iterate over each skeleton image in the skeleton_frames folder
                for filename in sorted(os.listdir(skeleton_frames_path)):
                    if filename.endswith(".png"):
                        image_path = os.path.join(skeleton_frames_path, filename)
                        output_path = os.path.join(completed_skeleton_path, filename)

                        # Process the skeleton image to fill gaps
                        skeleton_completer = SkeletonFiller(
                            image_path,
                            kernel_size=kernel_size,
                            iterations=iterations,
                            thickness=thickness
                        )

                        skeleton_completer.process_image()

                        # Save the completed skeleton image
                        skeleton_completer.save_image(output_path)
                        print(f"  Completed skeleton saved: {output_path}")