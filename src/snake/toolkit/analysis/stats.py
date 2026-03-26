"""Statistical functions for analysis of fMRI data with respect to ground truth."""

import logging
from typing import Literal
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
import numpy as np

from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm import compute_contrast, expression_to_contrast_vector
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

HeightControl = Literal["fpr", "fdr"]


def contrast_zscore(
    data: NDArray,
    TR_vol: float,
    bold_signal: NDArray,
    bold_sample_time: NDArray,
    contrast_name: str,
) -> NDArray:
    """Compute the contrast Z-score."""
    frame_times = (np.arange(len(data)) + 0.5) * TR_vol
    # regs =  _resample_regressor(
    #     bold_signal,
    #     bold_sample_time,
    #     frame_times,
    # )

    f = interp1d(bold_sample_time, bold_signal, fill_value="extrapolate")
    regs = f(frame_times).T

    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=None,
        add_regs=regs[:, None],
        add_reg_names=[contrast_name],
        drift_model=None,
    )
    image_ = abs(data).reshape(data.shape[0], -1)
    logger.debug(f"image_={image_.shape}, design matrix={design_matrix.shape}")
    labels, results = run_glm(image_, design_matrix.values)
    # Translate formulas to vectors
    con_val = expression_to_contrast_vector(
        contrast_name, design_matrix.columns.tolist()
    )

    contrast = compute_contrast(labels, results, con_val, contrast_type="t")

    z_image = np.zeros(data.shape[1:])
    z_image = contrast.z_score().reshape(z_image.shape)

    return z_image


def get_scores(
    contrast: np.ndarray, roi_mask: np.ndarray, roi_threshold: float
) -> dict[str, float]:
    """Get sklearn metrics scores.

    Parameters
    ----------
    thresh_map : np.ndarray
        Thresholded map.
    ground_truth : np.ndarray
        Ground truth map.
    alphas: list of float

    Returns
    -------
    scores : dict
        Dictionary of scores (accuracy, precision, recall, f1, jaccard)

    """
    stats = {}
    gt_f = roi_mask.ravel() >= roi_threshold
    P = np.sum(gt_f)
    N = gt_f.size - P

    fpr, tpr, thresholds = roc_curve(gt_f, contrast.ravel())
    stats["fpr"] = fpr.tolist()
    stats["tpr"] = tpr.tolist()
    stats["tresh"] = thresholds.tolist()
    stats["tp"] = np.int_(tpr * P).tolist()
    stats["fp"] = np.int_(fpr * N).tolist()
    stats["tn"] = np.int_(N * (1 - fpr)).tolist()
    stats["fn"] = np.int_(P * (1 - tpr)).tolist()
    return stats


def bacc(tpr: NDArray, fpr: NDArray, adjusted: bool = False) -> NDArray:
    """Compute Balanced Accuracy from TPR and FPR."""
    bacc = (tpr + 1 - fpr) / 2
    if adjusted:
        return (bacc - 0.5) * 2
    return bacc


def mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Compute Matthew's Correlation Coefficient."""
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    fnr = 1 - tpr
    fpr = 1 - tnr
    for_ = 1 - npv
    fdr = 1 - ppv

    return np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fnr * fpr * for_ * fdr)
