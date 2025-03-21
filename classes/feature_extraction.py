#%%
#%%
# file: feature_extraction.py

import os
import re
import pandas as pd
import numpy as np

class FeatureExtractor:
    """
    Old-style approach: Load 4 CSV files, extract and aggregate per-frame features from each,
    then perform an outer merge and fill NaN values with 0.
    The final features include:
      - frame (integer frame number)
      - cluster_count
      - mean_flow_magnitude
      - longest_path
      - skeleton_total_length
    """

    def __init__(self, patient_folder):
        self.patient_folder = patient_folder
        self.cp_file = os.path.join(patient_folder, "clustered_points.csv")
        self.flow_file = os.path.join(patient_folder, "flow.csv")
        self.lp_file = os.path.join(patient_folder, "longest_path_lengths.csv")
        self.sp_file = os.path.join(patient_folder, "skeleton_properties.csv")

    def _parse_frame_number(self, val):
        """
        Extract the frame number (e.g., 10) from a string like 'clusters_10.png' or 'clusters_10'.
        If parsing fails, return None.
        """
        if pd.isnull(val):
            return None
        s = str(val)
        match = re.search(r'(\d+)', s)
        if match:
            return int(match.group(1))
        return None

    def extract_features(self):
        """
        Sequentially extract features from 4 CSV files, perform an outer merge,
        fill NaN values with 0, and return the merged DataFrame.
        """
        # 1) Extract cluster_count
        cluster_counts = None
        if os.path.exists(self.cp_file):
            cp_df = pd.read_csv(self.cp_file)
            # Parse frame number from column "Frame"
            if "Frame" in cp_df.columns:
                cp_df["frame"] = cp_df["Frame"].apply(self._parse_frame_number)
            else:
                # If the actual column name is different, handle accordingly
                pass
            # Count clusters per frame
            cluster_counts = cp_df.groupby("frame").size().reset_index(name="cluster_count")

        # 2) Extract flow to compute mean_flow_magnitude
        flow_agg = None
        if os.path.exists(self.flow_file):
            flow_df = pd.read_csv(self.flow_file)
            if "Frame" in flow_df.columns:
                flow_df["frame"] = flow_df["Frame"].apply(self._parse_frame_number)
            # Calculate magnitude
            flow_df["magnitude"] = np.sqrt(flow_df["fx"]**2 + flow_df["fy"]**2)
            # Average magnitude per frame
            flow_agg = flow_df.groupby("frame")["magnitude"].mean().reset_index()
            flow_agg.rename(columns={"magnitude": "mean_flow_magnitude"}, inplace=True)

        # 3) Extract longest_path
        lp_agg = None
        if os.path.exists(self.lp_file):
            lp_df = pd.read_csv(self.lp_file)
            if "filename" in lp_df.columns:
                lp_df["frame"] = lp_df["filename"].apply(self._parse_frame_number)
            # If the column is named 'longest_path_length'
            if "longest_path_length" in lp_df.columns:
                lp_agg = lp_df[["frame", "longest_path_length"]].copy()
                lp_agg.rename(columns={"longest_path_length": "longest_path"}, inplace=True)

        # 4) Extract skeleton_properties to compute skeleton_total_length
        sp_agg = None
        if os.path.exists(self.sp_file):
            sp_df = pd.read_csv(self.sp_file)
            # Parse frame number from column "Frame"
            if "Frame" in sp_df.columns:
                sp_df["frame"] = sp_df["Frame"].apply(self._parse_frame_number)
            # If the column is named 'Length'
            if "Length" in sp_df.columns:
                sp_agg = sp_df.groupby("frame")["Length"].sum().reset_index()
                sp_agg.rename(columns={"Length": "skeleton_total_length"}, inplace=True)

        # 5) Perform sequential outer merge of available dataframes
        dfs = []
        for item in (cluster_counts, flow_agg, lp_agg, sp_agg):
            if item is not None and not item.empty:
                dfs.append(item)

        if not dfs:
            return None

        df_merged = dfs[0]
        for other in dfs[1:]:
            df_merged = pd.merge(df_merged, other, on="frame", how="outer")

        # Fill NaN values with 0 (consistent with the old version), sort by frame, and reset index
        df_merged.fillna(0, inplace=True)
        df_merged.sort_values("frame", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        return df_merged
