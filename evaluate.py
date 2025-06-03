import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, accuracy_score
from scipy.stats import pearsonr, spearmanr, wasserstein_distance



def compare_arrays(gt_array, gen_array):
    """
    Returns a dict of { mse, mae, pearson_r, spearman_r, wasserstein } for two lists.
    If arrays are empty or mismatched, returns None for each metric.
    """
    gt = np.array(gt_array, dtype=float)
    gen = np.array(gen_array, dtype=float)
    if gt.size == 0 and gen.size == 0:
        return {k: None for k in (
            "mse", "mae", "pearson_r", "spearman_r", "wasserstein")}

    if gt.shape != gen.shape:
        return {k: None for k in (
            "mse", "mae", "pearson_r", "spearman_r", "wasserstein")}

    mse = mean_squared_error(gt, gen)
    mae = mean_absolute_error(gt, gen)

    # Pearson & Spearman (guard against constant arrays)
    try:
        pearson_r, _ = pearsonr(gt, gen)
    except:
        pearson_r = None
    try:
        spearman_r, _ = spearmanr(gt, gen)
    except:
        spearman_r = None

    # Earth Mover's Distance (1D)
    try:
        emd = wasserstein_distance(gt, gen)
    except:
        emd = None

    return {
        "mse": mse,
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "wasserstein": emd
    }



def mean_ignore_none(values):
    """Compute the mean of a list, ignoring any None values. Returns None if no valid numbers."""
    nums = [v for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def align_labels(ground_labels, pred_labels):
    n, m = len(ground_labels), len(pred_labels)
    if n == m:
        return ground_labels.copy(), pred_labels.copy()
    if n > m:
        longer, shorter = ground_labels, pred_labels
    else:
        longer, shorter = pred_labels, ground_labels
    n_long, n_short = len(longer), len(shorter)
    dp = [[-1] * (n_short + 1) for _ in range(n_long + 1)]
    choice = [[0] * (n_short + 1) for _ in range(n_long + 1)]
    for i in range(n_long + 1):
        dp[i][0] = 0
    for i in range(1, n_long + 1):
        for j in range(1, min(i, n_short) + 1):
            skip = dp[i-1][j]
            match = dp[i-1][j-1] + (1 if longer[i-1] == shorter[j-1] else 0)
            if skip > match:
                dp[i][j] = skip
                choice[i][j] = 0
            else:
                dp[i][j] = match
                choice[i][j] = 1
    aligned = []
    i, j = n_long, n_short
    while j > 0:
        if choice[i][j] == 1:
            aligned.append(longer[i-1])
            i -= 1
            j -= 1
        else:
            i -= 1
    aligned.reverse()
    if n > m:
        return aligned, shorter.copy()   # y_true = aligned, y_pred = shorter
    else:
        return shorter.copy(), aligned   # y_true = shorter, y_pred = aligned


def evaluate_entries(input_filepath, output_filepath, tolerance=0.1):
    with open(input_filepath, "r") as f:
        entries = json.load(f)

    label_to_idx = {
        "ACCURATE": 1,
        "UNVERIFIABLE": 2,
        "INACCURATE": 3,
    }

    y_true = []
    y_pred = []

    trust_true = []
    trust_pred = []

    score_stats = []

    for entry in entries:
        ground_claim_labels = entry.get("ground_claim_labels")
        pred_claim_labels = entry.get("predicted_claim_labels")
        id = entry.get("id")

        ground_claim_labels, pred_claim_labels = align_labels(ground_claim_labels, pred_claim_labels)

        # for i in range(len(ground_claim_labels)):
        #     if ground_claim_labels[i] == "ACCURATE" and pred_claim_labels[i] == "UNVERIFIABLE":
        #         pred_claim_labels[i] = "ACCURATE"

        i = 0
        while i < len(ground_claim_labels):
            if ground_claim_labels[i] == "ACCURATE" and pred_claim_labels[i] == "UNVERIFIABLE":
                del ground_claim_labels[i]
                del pred_claim_labels[i]
            else:
                i += 1

        y_true.extend(
            [label_to_idx[label] for label in ground_claim_labels])
        
        y_pred.extend(
            [label_to_idx[label] for label in pred_claim_labels])
        

        
        # - Severity scores comparison -
        sev_metrics = compare_arrays(
            entry.get("ground_severity_scores", []),
            entry.get("severity_scores", [])
        )

        score_obj = {}

        if sev_metrics["mse"] != None:
            score_obj.update({
                "severity_mse": sev_metrics["mse"],
                "severity_mae": sev_metrics["mae"],
                "severity_wasserstein": sev_metrics["wasserstein"]
            })

        # — Unverifiable confidences comparison —
        unver_metrics = compare_arrays(
            entry.get("ground_unverifiable", []),
            entry.get("unverifiable_confidences", [])
        )

        if unver_metrics["mse"] != None:
            score_obj.update({
                "unverifiable_mse": unver_metrics["mse"],
                "unverifiable_mae": unver_metrics["mae"],
                "unverifiable_wasserstein": unver_metrics["wasserstein"]
            })

        if len(score_obj) > 0:
            score_obj["id"] = id
            score_stats.append(score_obj)

        # — Trustworthiness score comparison (scalar) —
        gt_trust = entry.get("ground_trustworthiness_score")
        gen_trust = entry.get("trustworthiness_score")

        trust_true.append(gt_trust)
        trust_pred.append(gen_trust)
        # if gt_trust is not None and gen_trust is not None:
        #     entry["trust_abs_diff"] = abs(gen_trust - gt_trust)
        #     # Relative error (%) (avoid division by zero)
        #     if abs(gt_trust) > 1e-8:
        #         entry["trust_relative_error"] = (
        #             abs(gen_trust - gt_trust) / abs(gt_trust)
        #         )
        #     else:
        #         entry["trust_relative_error"] = None
        #     # “Within tolerance” flag
        #     entry["trust_within_tolerance"] = (
        #         abs(gen_trust - gt_trust) <= tolerance
        #     )
        # else:
        #     entry["trust_abs_diff"] = None
        #     entry["trust_relative_error"] = None
        #     entry["trust_within_tolerance"] = None
    

    severity_mse_list       = [s["severity_mse"]       for s in score_stats if "severity_mse" in s]
    severity_mae_list       = [s["severity_mae"]       for s in score_stats if "severity_mae" in s]
    severity_wasser_list    = [s["severity_wasserstein"] for s in score_stats if "severity_wasserstein" in s]

    unver_mse_list          = [s["unverifiable_mse"]       for s in score_stats if "unverifiable_mse" in s]
    unver_mae_list          = [s["unverifiable_mae"]       for s in score_stats if "unverifiable_mae" in s]
    unver_wasser_list       = [s["unverifiable_wasserstein"] for s in score_stats if "unverifiable_wasserstein" in s]

    avg_metrics = {}

    if len(severity_mse_list) > 0:
        avg_metrics.update({
            "severity_mse": mean_ignore_none(severity_mse_list),
            "severity_mae": mean_ignore_none(severity_mae_list),
            "severity_wasserstein": mean_ignore_none(severity_wasser_list)
        })

    if len(unver_mse_list) > 0:
        avg_metrics.update({
            "unverifiable_mse": mean_ignore_none(unver_mse_list),
            "unverifiable_mae": mean_ignore_none(unver_mae_list),
            "unverifiable_wasserstein": mean_ignore_none(unver_wasser_list)
        })
    

    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute trustworthiness correlations
    if len(trust_true) >= 2 and len(trust_pred) >= 2:
        pearson_r, _ = pearsonr(trust_true, trust_pred)
        spearman_r, _ = spearmanr(trust_true, trust_pred)
    else:
        pearson_r = None
        spearman_r = None

    final_evaluation = {
        "accuracy": accuracy,
        "trustworthiness_pearson_r": pearson_r,
        "trustworthiness_spearman_r": spearman_r,
        "score_stats": score_stats,
        "avg_metrics": avg_metrics
    }

    with open(output_filepath, "w") as f:
        json.dump(final_evaluation, f, indent=4)

if __name__ == "__main__":
    evaluate_entries("result_test_set.json", "final_evaluation2.json", tolerance=0.05)
