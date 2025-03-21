import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def determine_tmpfc(
        df,
        clearance_threshold,
        frame_rate=15,
        standard_frame_rate=30,
        sigma=1.0,
        peak_prominence=0.05
):
    """
    TMPFC Calculation Function:
      1) Assumes that df already contains the pre-calculated normalized features: norm_lp, norm_pc, norm_mf,
         as well as two composite scores:
           - composite_score_f1 (calculated using F1 weights without threshold)
           - composite_score_f2 (calculated using F2 weights for comparison with clearance_threshold)
      2) First, apply Gaussian smoothing to composite_score_f1 and then use peak detection to determine F1:
           - For example, use the rule "0.8 * M" to select the first peak that meets the condition.
      3) Once F1 is determined, in the frames following F1, first apply Gaussian smoothing to composite_score_f2
         to eliminate abrupt spikes, then choose the first frame where the value is below clearance_threshold as F2;
         if not found directly, perform dynamic shifting (subtracting the minimum value) and search again;
         if still not found, take the last frame.
      4) Finally, TMPFC = (F2 - F1) * (standard_frame_rate / frame_rate)
    """
    if df.empty:
        return None

    # 1) Sort by frame in ascending order
    df_sorted = df.sort_values("frame").copy()
    frames = df_sorted["frame"].values

    # --- F1 Detection ---
    # Apply Gaussian smoothing to composite_score_f1
    scores_f1 = df_sorted["composite_score_f1"].values
    smooth_scores_f1 = gaussian_filter1d(scores_f1, sigma=sigma)

    # Perform peak detection, filtering out peaks that are not prominent enough
    peaks, properties = find_peaks(smooth_scores_f1, prominence=peak_prominence)
    if len(peaks) == 0:
        return None  # Fallback if no significant peak is found

    M = smooth_scores_f1.max()
    threshold_value = 0.8 * M
    valid_peaks_idx = [i for i in peaks if smooth_scores_f1[i] >= threshold_value]
    if not valid_peaks_idx:
        F1_idx = np.argmax(smooth_scores_f1)  # Fallback to the global maximum
    else:
        F1_idx = valid_peaks_idx[0]  # Choose the earliest peak that meets the condition

    F1_frame = frames[F1_idx]

    # --- F2 Detection ---
    # Search only in frames after F1
    df_after = df_sorted[df_sorted["frame"] > F1_frame].copy()
    if df_after.empty:
        return None

    # Apply Gaussian smoothing to composite_score_f2 to remove the effect of abrupt spikes
    scores_f2 = df_after["composite_score_f2"].values
    smooth_scores_f2 = gaussian_filter1d(scores_f2, sigma=sigma)

    # Directly search for frames where the smoothed composite_score_f2 falls below clearance_threshold
    indices_below = np.where(smooth_scores_f2 < clearance_threshold)[0]
    if len(indices_below) > 0:
        F2_idx = indices_below[0]
        F2_frame = df_after.iloc[F2_idx]["frame"]
    else:
        # Dynamic shifting: subtract the minimum value from composite_score_f2 and search again
        min_score = df_after["composite_score_f2"].min()
        df_after["score_adj"] = df_after["composite_score_f2"] - min_score
        below_thresh_adj = df_after[df_after["score_adj"] < clearance_threshold]
        if not below_thresh_adj.empty:
            F2_frame = below_thresh_adj.iloc[0]["frame"]
        else:
            # If still not found, use the last frame
            F2_frame = df_after.iloc[-1]["frame"]

    # Calculate TMPFC
    tmpfc = (F2_frame - F1_frame) * (standard_frame_rate / frame_rate)
    return (tmpfc, F1_frame, F2_frame)
