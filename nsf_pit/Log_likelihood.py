import numpy as np
from scipy.stats import norm, beta
import pandas as pd

from pit import pit



##############################
# Original PDF function for TLP
##############################

def tlp_pdf(
        forecast: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        y: np.ndarray,
        sigma_y: float,
) -> np.ndarray:

    K,T = forecast.shape
    P,T_w, K_w = weights.shape
    M = x.shape[0]

    assert (T, K) == (T_w, K_w), 'Shape mismatch'

    pdf_values = np.zeros((K, T, M), dtype=float)

    for k in range(K):
        for t in range(T):
            # Vectorized over all grid points at once
            pdf_values[k, t, :] = norm.pdf(x, loc=(forecast.to_numpy())[k, t], scale=sigma_y)

    TL = np.einsum('ktm,ptk -> ptm', pdf_values, weights)

    pdf = pit(TL, x, y)

    return pdf



def log_score_comparison(cdfs, a, b, TLP, TLP_pdf, before_log_score,discount_factors):
    """
    Compare mean log-scores before and after calibration.
    """

    P, T = cdfs.shape
    P,T = a.shape
    P,T = b.shape

    log_scores = np.zeros((P, T), dtype=float)

    for p in range(P):
        for t in range(T):
            # raw cdf
            F_lp = TLP[p,t] 

            # raw pdf
            f_lp = TLP_pdf[p,t]
            
            # 2) Beta‐PDF at the PIT value u = new_cdf[p,t]
            a_t = a[p, t]   # shape used *before* seeing y_t
            b_t = b[p, t]
            f_beta = beta.pdf(F_lp, a_t, b_t) * f_lp
            
            log_scores[p, t] = np.log(f_beta + 1e-12)  # jitter to avoid log(0)

    after_mean_log_score = log_scores.mean(axis=1)
    before_mean_log_score = before_log_score.mean(axis=1)

    results_df = pd.DataFrame({
    'discount_factor': discount_factors,
    'Before Calibration': before_mean_log_score,
    'After Calibration': after_mean_log_score,})

    # Add comparison column
    results_df['Comparison'] = results_df['After Calibration'] > results_df['Before Calibration']
    results_df['Comparison'] = results_df['Comparison'].map({True: 'Higher Log Score', False: 'Lower Log Score'})
    return results_df