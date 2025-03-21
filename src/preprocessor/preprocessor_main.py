'''
This is the script for stage 1 to stage 4 after dicom video preprocessing
'''

import os
from src.config import RESULTS_DIR
from src.preprocessor.optical_analysis import process_optical_analysis
from src.preprocessor.skeletonization import process_skeletonization
from src.preprocessor.skeleton_filler import process_skeleton_filling
from src.preprocessor.longest_path_calc import process_longest_path_overlay

i = 11
group_dir = os.path.join(RESULTS_DIR, f"group_{i}")

# Stage 1: Optical Analysis
process_optical_analysis(group_dir)

# Stage 2: Skeletonization
process_skeletonization(group_dir)

# Stage 3: Skeleton Filler
process_skeleton_filling(group_dir)

# Stage 4: Longest Path Calculation
process_longest_path_overlay(group_dir)

print(f"Group {i} has been processed successfully.")
