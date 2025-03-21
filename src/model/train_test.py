#%%
import os
import math
import pandas as pd
import numpy as np
from math import sqrt
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Assume the updated determine_tmpfc is implemented (including sigma and peak_prominence parameters)
from src.model.tmpfc_calculation_piecewise import determine_tmpfc

# -----------------------------
# Global Constants: Frame Rates & Data Paths
# -----------------------------
FRAME_RATE = 15           # Original video frame rate (fps)
STANDARD_FRAME_RATE = 30  # Standard frame rate after TIMI correction

TRAIN_DIR = os.path.join("data", "train")
TEST_DIR = os.path.join("data", "test")
GROUND_TRUTH_FILE = os.path.join("data", "patient_info.xlsx")

# -----------------------------
# 1. Basic Functions: Load & Calculate TMPFC
# -----------------------------
def load_feature(patient_folder):
    """
    Load a patient's feature CSV file and return a DataFrame.
    The file is assumed to be named {patient_id}_feature.csv located in the patient's folder.
    """
    patient_id = os.path.basename(os.path.normpath(patient_folder))
    feature_file = os.path.join(patient_folder, f"{patient_id}_feature.csv")
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        return df
    else:
        print(f"Feature file not found for patient: {patient_id}")
        return None

def calculate_tmpfc_linear(patient_folder, f1_weights, f2_weights, clearance_threshold, sigma, peak_prominence):
    """
    Calculate the predicted TMPFC based on the pre-saved feature file:
      - Uses features: longest_path_length, point_count, total_modulus (for calculating mean_flow)
      - Computes mean_flow = total_modulus / point_count (if point_count==0 then 0)
      - Normalizes each of the three features
      - Uses f1_weights to calculate composite_score_f1 for F1 detection
      - Uses f2_weights to calculate composite_score_f2 for F2 detection (later compared with clearance_threshold)
      - Calls determine_tmpfc to return (tmpfc, F1, F2)
      - sigma and peak_prominence are passed as tuning parameters to determine_tmpfc
    """
    df = load_feature(patient_folder)
    if df is None or df.empty:
        return None

    # Standardize frame numbering column: if "Frame" exists, rename it to "frame"
    if "frame" not in df.columns:
        if "Frame" in df.columns:
            df.rename(columns={"Frame": "frame"}, inplace=True)
        else:
            return None

    # Check for required feature columns
    required_columns = ["longest_path_length", "point_count", "total_modulus"]
    for col in required_columns:
        if col not in df.columns:
            pid = os.path.basename(os.path.normpath(patient_folder))
            print(f"Missing required column '{col}' in patient {pid}")
            return None

    # Calculate mean_flow
    df["mean_flow"] = df.apply(
        lambda row: row["total_modulus"] / row["point_count"] if row["point_count"] > 0 else 0,
        axis=1
    )

    # Normalization function
    def normalize(series):
        max_val = series.max()
        return series / max_val if max_val > 0 else series

    df["norm_lp"] = normalize(df["longest_path_length"])
    df["norm_pc"] = normalize(df["point_count"])
    df["norm_mf"] = normalize(df["mean_flow"])

    # Normalize weights (ensure the sum of weights equals 1)
    def normalize_weights(weights):
        s = sum(weights)
        return tuple(w/s for w in weights) if s > 0 else weights

    f1_w = normalize_weights(f1_weights)
    f2_w = normalize_weights(f2_weights)

    # Compute two sets of composite scores
    df["composite_score_f1"] = (
            f1_w[0] * df["norm_lp"] +
            f1_w[1] * df["norm_pc"] +
            f1_w[2] * df["norm_mf"]
    )
    df["composite_score_f2"] = (
            f2_w[0] * df["norm_lp"] +
            f2_w[1] * df["norm_pc"] +
            f2_w[2] * df["norm_mf"]
    )

    # Call the updated determine_tmpfc, passing sigma and peak_prominence
    result = determine_tmpfc(
        df,
        clearance_threshold,
        frame_rate=FRAME_RATE,
        standard_frame_rate=STANDARD_FRAME_RATE,
        sigma=sigma,
        peak_prominence=peak_prominence
    )
    if result is None:
        return None
    tmpfc, F1, F2 = result
    return tmpfc

# -----------------------------
# 2. Evaluation Functions: OLS & Deming
# -----------------------------
def deming_regression(x, y, lambda_=1.0):
    """
    Simple Deming regression (error-in-variables), assuming the error variance ratio of X and Y is lambda_.
    Returns (intercept, slope).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.mean((x - x_mean) ** 2)
    syy = np.mean((y - y_mean) ** 2)
    sxy = np.mean((x - x_mean) * (y - y_mean))

    numerator = (syy - lambda_ * sxx) + math.sqrt((syy - lambda_ * sxx) ** 2 + 4 * lambda_ * (sxy**2))
    slope = numerator / (2 * sxy)
    intercept = y_mean - slope * x_mean
    return intercept, slope

def compute_regression_reports(merged):
    """
    Perform two regressions based on the idea of treating predicted as X and true as Y:
      1) OLS
      2) Deming (lambda=1.0)
    Returns (ols_intercept, ols_slope, ols_r2, deming_intercept, deming_slope, deming_r2)
    """
    X = merged[["pred"]].values  # predicted
    y = merged["TMPFC"].values   # true

    # OLS regression
    ols_model = LinearRegression()
    ols_model.fit(X, y)
    y_ols = ols_model.predict(X)
    ols_r2 = r2_score(y, y_ols)
    ols_intercept = ols_model.intercept_
    ols_slope = ols_model.coef_[0]

    # Deming regression
    deming_i, deming_b = deming_regression(X.flatten(), y.flatten(), lambda_=1.0)
    y_deming = deming_i + deming_b * X.flatten()
    deming_r2 = r2_score(y, y_deming)

    return ols_intercept, ols_slope, ols_r2, deming_i, deming_b, deming_r2

def evaluate_linear(set_dir, f1_weights, f2_weights, clearance_threshold, gt_df, sigma, peak_prominence, exclude_ids=None):
    """
    Iterate over all patients in set_dir (train or test directory), calculate the predicted TMPFC,
    and compare with the ground truth.
    Parameter exclude_ids: if provided, skip patients with these IDs.
    Returns:
      - predictions dictionary (key: patient ID, value: predicted TMPFC)
      - rmse, mae, and directly computed r2_score
      - Also prints OLS/Deming regression results
    """
    preds = {}
    for patient in os.listdir(set_dir):
        if exclude_ids is not None and patient in exclude_ids:
            continue

        patient_folder = os.path.join(set_dir, patient)
        if not os.path.isdir(patient_folder):
            continue
        tmpfc = calculate_tmpfc_linear(patient_folder, f1_weights, f2_weights, clearance_threshold, sigma, peak_prominence)
        if tmpfc is not None:
            preds[patient] = tmpfc

    pred_df = pd.DataFrame(list(preds.items()), columns=["id", "pred"])
    pred_df["id"] = pred_df["id"].astype(str)
    gt_df["id"] = gt_df["id"].astype(str)
    merged = pd.merge(pred_df, gt_df, on="id", how="inner")
    merged = merged.dropna(subset=["TMPFC", "pred"])
    if merged.empty:
        return preds, None, None, None

    print("\nIndividual patient results:")
    for index, row in merged.iterrows():
        print(f"Patient {row['id']}: Predicted TMPFC = {row['pred']}, Actual TMPFC = {row['TMPFC']}")

    y_true = merged["TMPFC"].values
    y_pred = merged["pred"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    direct_r2 = r2_score(y_true, y_pred)

    (ols_intercept, ols_slope, ols_r2,
     deming_i, deming_b, deming_r2) = compute_regression_reports(merged)

    print("\n--- OLS Regression (Observed on Predicted) ---")
    print(f"Intercept: {ols_intercept:.3f}")
    print(f"Slope: {ols_slope:.3f}")
    print(f"OLS R²: {ols_r2:.3f}")
    print("\n--- Deming Regression (Observed on Predicted) ---")
    print(f"Intercept: {deming_i:.3f}")
    print(f"Slope: {deming_b:.3f}")
    print(f"Deming R² (pseudo): {deming_r2:.3f}")

    return preds, rmse, mae, direct_r2

# -----------------------------
# 4. Objective Function: Optimize using MAE
# -----------------------------
def objective_linear(params):
    """
    Objective function: the input parameters are 9 numbers:
      [w1_f1, w2_f1, w3_f1, w1_f2, w2_f2, w3_f2, sigma, peak_prominence, clearance_threshold]
    These represent weights for F1 and F2 (internally normalized so that each group sums to 1),
    and sigma, peak_prominence, and clearance_threshold for determine_tmpfc.
    Computes the MAE on the training set and returns the MAE for minimization.
    """
    # Unpack parameters
    w1_f1, w2_f1, w3_f1, w1_f2, w2_f2, w3_f2, sigma, peak_prominence, clearance_threshold = params
    f1_weights = (w1_f1, w2_f1, w3_f1)
    f2_weights = (w1_f2, w2_f2, w3_f2)

    predictions = {}
    for p in os.listdir(TRAIN_DIR):
        # Since the train folder is already cleaned, no exclusions are necessary.
        patient_folder = os.path.join(TRAIN_DIR, p)
        if not os.path.isdir(patient_folder):
            continue
        tmpfc = calculate_tmpfc_linear(patient_folder, f1_weights, f2_weights, clearance_threshold, sigma, peak_prominence)
        if tmpfc is not None:
            predictions[p] = tmpfc

    if not predictions:
        return 1e6

    pred_df = pd.DataFrame(list(predictions.items()), columns=["id", "pred"])
    pred_df["id"] = pred_df["id"].astype(str)
    gt_subset = gt_df[gt_df["id"].isin(pred_df["id"])]
    merged = pd.merge(pred_df, gt_subset, on="id", how="inner")
    merged = merged.dropna(subset=["TMPFC", "pred"])

    if merged.empty:
        return 1e6

    y_true = merged["TMPFC"].values
    y_pred = merged["pred"].values
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Params=(F1: {f1_weights}, F2: {f2_weights}, sigma: {sigma:.3f}, "
          f"peak_prominence: {peak_prominence:.3f}, clearance_threshold: {clearance_threshold:.3f}) "
          f"=> MAE={mae:.2f}, R²={r2:.2f}")
    return mae

# -----------------------------
# 5. Main Script Flow
# -----------------------------
if __name__ == "__main__":
    from skopt import gp_minimize

    # Load ground truth
    gt_df = pd.read_excel(GROUND_TRUTH_FILE).copy()
    gt_df["id"] = gt_df["id"].astype(str)
    gt_df["TMPFC"] = pd.to_numeric(gt_df["TMPFC"], errors="coerce")

    # Since the train and test folders are already cleaned, no exclusions are needed.
    excluded_train_ids = set()
    excluded_test_ids = set()

    # Calculate the number of valid training and testing observations
    valid_train_ids = [p for p in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, p))]
    n_train = len(valid_train_ids)
    valid_test_ids = [p for p in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, p))]
    n_test = len(valid_test_ids)

    # Define the search space (total of 9 parameters)
    space = [
        Real(0.1, 2.0, name="w1_f1"),
        Real(0.1, 1.0, name="w2_f1"),
        Real(0.1, 1.0, name="w3_f1"),
        Real(0.01, 0.2, name="w1_f2"),
        Real(0.1, 1.2, name="w2_f2"),
        Real(0.1, 1.0, name="w3_f2"),
        Real(0.5, 2.0, name="sigma"),
        Real(0.01, 0.1, name="peak_prominence"),
        Real(0.01, 0.1, name="clearance_threshold")
    ]

    # Bayesian optimization: using the MAE returned by objective_linear as the objective
    res = gp_minimize(objective_linear, space, n_calls=100, random_state=2025)
    best_params = res.x
    best_score = res.fun

    print("\n===============================")
    print("Best parameters found on train set:")
    print(f"  F1 weights: {best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}")
    print(f"  F2 weights: {best_params[3]:.3f}, {best_params[4]:.3f}, {best_params[5]:.3f}")
    print(f"  sigma: {best_params[6]:.3f}, peak_prominence: {best_params[7]:.3f}")
    print(f"  clearance_threshold: {best_params[8]:.3f}")
    print(f"Best train MAE: {best_score:.3f}")
    print(f"Total training observations used: {n_train}")
    print("===============================\n")

    # Evaluate on the TEST set (no exclusions)
    print("Evaluate on TEST set:")
    preds_test, test_rmse, test_mae, test_direct_r2 = evaluate_linear(
        TEST_DIR,
        (best_params[0], best_params[1], best_params[2]),
        (best_params[3], best_params[4], best_params[5]),
        best_params[8],
        gt_df,
        best_params[6],
        best_params[7],
        exclude_ids=excluded_test_ids  # No exclusions in test set
    )
    print("\n== Final Test set metrics ==")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MAE : {test_mae:.3f}")
    print(f"R² (direct): {test_direct_r2:.3f}")
    print(f"Total test observations used: {n_test}")
    print("===============================\n")
