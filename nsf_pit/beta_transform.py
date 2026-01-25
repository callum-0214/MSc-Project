"""
Online Beta Calibration with Shape-Specific Corrections
"""

import numpy as np
from scipy.stats import beta, chi2, norm
from scipy.special import gamma as gamma_fn, digamma
from scipy.interpolate import interp1d


# ============================================================================
# Algorithm 1: Shape Detection
# ============================================================================

def compute_chi_square_decomposition(pit_values, k, significance_level):
    """
    Algorithm 1: Chi-square decomposition with Ends contrast
    
    Args:
        pit_values: Array of PIT values
        k: Number of bins for chi-square test
        significance_level: Threshold for statistical tests
        
    Returns:
        p_ends_i: P-value for ends contrast test
        u_i: Ends contrast statistic
        is_over: Boolean, True if over-dispersed (bell-shaped)
        is_under: Boolean, True if under-dispersed (U-shaped)
    """
    w = len(pit_values)
    
    # Histogram on [0,1] with equal-width bins
    hist, _ = np.histogram(pit_values, bins=k, range=(0, 1))
    
    # Expected frequency e = w/k
    e = w / k
    
    # Chi-square test statistic
    T_i = np.sum((hist - e)**2 / e)
    
    # Overall significance test
    p_i = 1 - chi2.cdf(T_i, df=k-1)
    
    if p_i >= significance_level:
        return np.nan, np.nan, False, False  # Not significant
    
    # Pearson residuals x_j = (n_j - e) / sqrt(e)
    x = (hist - e) / np.sqrt(e)
    
    # Construct unnormalized Ends contrast
    h = k // 2
    l_ends_unnorm = np.zeros(k)
    
    if k % 2 == 0:  # k = 2h even
        l_ends_unnorm[0] = -(h - 1)      # j=1 (index 0)
        l_ends_unnorm[-1] = -(h - 1)     # j=k (index k-1)
        l_ends_unnorm[1:-1] = 1          # j in {2,...,k-1}
    else:  # k = 2h+1 odd
        l_ends_unnorm[0] = -(2*h + 1)    # j=1 (index 0)
        l_ends_unnorm[-1] = -(2*h + 1)   # j=k (index k-1)
        l_ends_unnorm[1:-1] = 2          # j in {2,...,k-1}
    
    # Normalize contrast
    l_ends = l_ends_unnorm / np.sqrt(np.sum(l_ends_unnorm**2))
    
    # Individual decomposition
    u_i = np.sum(l_ends * x)
    chi2_i = u_i**2
    p_ends_i = 1 - chi2.cdf(chi2_i, df=1)
    
    if p_ends_i >= significance_level:
        return np.nan, np.nan, False, False  # Ends test not significant
        
    # Sample variance for dispersion classification
    pit_centred = pit_values - 0.5
    S_squared = np.mean(pit_centred**2)
    
    uniform_var = 1.0 / 12.0
    
    # Compute threshold for dispersion test
    z_critical = norm.ppf(1 - significance_level / 2)
    delta_i = z_critical * np.sqrt(1 / (180 * w))
    
    # Dispersion classification
    is_over = S_squared < (uniform_var - delta_i)   # Over-dispersed (bell)
    is_under = S_squared > (uniform_var + delta_i)  # Under-dispersed (U)
    
    return p_ends_i, u_i, is_over, is_under


def detect_bell_shape_by_model(pit_history_by_model, 
                                significance_level, 
                                k_bins,
                                detection_window):
    """
    Algorithm 1: Detect Over/Under dispersion using chi-square decomposition
    
    Args:
        pit_history_by_model: List of PIT arrays, one per model
        significance_level: Significance threshold for tests
        k_bins: Number of bins for chi-square test
        detection_window: Number of recent observations to use
        
    Returns:
        bell_shaped_models: List of model indices with bell-shaped PITs
        u_shaped_models: List of model indices with U-shaped PITs
        p_ends_arr: Array of p-values for ends contrast
        u_ends_arr: Array of ends contrast statistics
    """
    P = len(pit_history_by_model)
    bell_shaped_models = []
    u_shaped_models = []
    u_ends_arr = np.full(P, np.nan)
    p_ends_arr = np.full(P, np.nan)

    for i in range(P):
        model_pits = np.array(pit_history_by_model[i])
        if len(model_pits) < 20:
            continue

        # Use recent data only
        recent_pits = (model_pits[-detection_window:] 
                      if len(model_pits) > detection_window 
                      else model_pits)

        # Apply chi-square decomposition
        p_ends, u_ends, is_over, is_under = compute_chi_square_decomposition(
            recent_pits, k_bins, significance_level
        )
        
        u_ends_arr[i] = u_ends
        p_ends_arr[i] = p_ends

        if not np.isnan(p_ends) and not np.isnan(u_ends):
            if is_over:
                bell_shaped_models.append(i)  # Over-dispersed (bell-shaped)
            elif is_under:
                u_shaped_models.append(i)     # Under-dispersed (U-shaped)

    return bell_shaped_models, u_shaped_models, p_ends_arr, u_ends_arr


# ============================================================================
# Algorithm 2: Shape-Specific Corrections
# ============================================================================

def apply_model_specific_corrections(a_c, b_c, pit_history_by_model,
                                    significance_level,
                                    k_bins_for_detection,
                                    window,
                                    min_samples,
                                    correction_strength,
                                    param_floor,
                                    param_ceiling,
                                    x_constant):
    """
    Algorithm 2: Apply Shape-Specific Corrections
    
    Args:
        a_c: Current alpha parameters (array of length P)
        b_c: Current beta parameters (array of length P)
        pit_history_by_model: List of PIT histories per model
        significance_level: Significance threshold for detection
        k_bins_for_detection: Bins for chi-square decomposition
        window: Rolling window size for recent data
        min_samples: Minimum samples required for correction
        correction_strength: Blending weight for corrections (0-1)
        param_floor: Lower bound for parameters
        param_ceiling: Upper bound for parameters
        x_constant: Additive constant for target parameter calculation
        
    Returns:
        a: Corrected alpha parameters
        b: Corrected beta parameters
        diag: List of diagnostic information per model
    """
    def _clip(x, lo, hi): 
        return lo if x < lo else (hi if x > hi else x)

    def _blend(old, target, lam):
        return (1.0 - lam) * old + lam * target

    # Detect shape and significance
    bell_idx, u_idx, p_ends_arr, u_ends_arr = detect_bell_shape_by_model(
        pit_history_by_model,
        significance_level=significance_level,
        k_bins=k_bins_for_detection,
        detection_window=window
    )

    P = len(pit_history_by_model)
    a = np.asarray(a_c, float).copy()
    b = np.asarray(b_c, float).copy()
    diag = []

    # Loop over models
    for i in range(P):
        pits = np.asarray(pit_history_by_model[i], float)
        n_total = pits.size
        
        if n_total < min_samples:
            diag.append({'n_used': n_total, 'direction': 'insufficient_data'})
            continue

        # Recent window
        recent = pits[-window:] if n_total > window else pits
        n_used = recent.size

        uE_d = float(u_ends_arr[i]) if i < len(u_ends_arr) else np.nan
        pE = float(p_ends_arr[i]) if i < len(p_ends_arr) else np.nan

        is_bell_shaped = i in bell_idx
        is_u_shaped = i in u_idx

        t_i = x_constant + np.tanh(abs(uE_d - np.log(pE + 1e-10)))
        
        # Parameter targeting based on detected shape
        if is_bell_shaped:
            alpha_star = t_i
            beta_star = t_i
            direction = 'bell'
        elif is_u_shaped:
            alpha_star = 1.0 / t_i
            beta_star = 1.0 / t_i
            direction = 'U'
        else:
            alpha_star = a[i]
            beta_star = b[i]
            direction = 'none'
        
        # Blend corrections with existing parameters
        a[i] = float(_clip(_blend(a[i], alpha_star, correction_strength), 
                          param_floor, param_ceiling))
        b[i] = float(_clip(_blend(b[i], beta_star, correction_strength), 
                          param_floor, param_ceiling))

        diag.append({
            'n_used': n_used,
            'u_ends_d': uE_d,
            'p_ends': pE,
            'is_bell': is_bell_shaped,
            'is_u': is_u_shaped,
            'direction': direction,
            'alpha_star': alpha_star,
            'beta_star': beta_star,
            't_i': t_i
        })

    return a, b, diag


# ============================================================================
# Gradient Computation
# ============================================================================

def compute_crps_gradients(a_c, b_c, tl_t, y_t, x_grid, P, geom_3f2, eps=1e-5):
    """
    Compute analytical CRPS gradients for all models
    
    Args:
        a_c: Current alpha parameters (length P)
        b_c: Current beta parameters (length P)
        tl_t: Linear pool CDF values at y_t (length P)
        y_t: Current observation
        x_grid: Grid of x values for integration
        P: Number of models
        geom_3f2: Generalized hypergeometric function
        eps: Small constant for numerical stability
        
    Returns:
        gradients_a: Gradient w.r.t. alpha (length P)
        gradients_b: Gradient w.r.t. beta (length P)
    """
    gradients_a = np.zeros(P)
    gradients_b = np.zeros(P)

    # Heaviside function H_t(x) = 1{x >= y_t}
    H_t = (x_grid >= y_t).astype(float)

    for i in range(P):
        a_i = a_c[i]
        b_i = b_c[i]
        z_i = np.clip(tl_t[i], eps, 1 - eps)

        # Beta CDFs
        bl_t = beta.cdf(z_i, a_i, b_i)
        bl_inv_t = beta.cdf(1 - z_i, b_i, a_i)

        # Common term: F(z) - H_t
        left = bl_t - H_t

        # ∂CRPS/∂a gradient computation
        term1 = bl_t * (np.log(bl_t + eps) - digamma(a_i) + digamma(a_i + b_i))

        coef_a = gamma_fn(a_i) * gamma_fn(a_i + b_i) / gamma_fn(b_i)

        G_a = geom_3f2(
            z=z_i,
            a_params=[a_i, a_i, 1.0 - b_i],
            b_params=[a_i + 1, a_i + 1],
            m=10
        )

        right_a = term1 - coef_a * (z_i**a_i) * G_a
        grad_a = 2 * a_i * np.trapz(left * right_a, x_grid, axis=0)

        # ∂CRPS/∂b gradient computation
        coef_b = gamma_fn(a_i + b_i) * gamma_fn(b_i) / gamma_fn(a_i)

        G_b = geom_3f2(
            z=1 - z_i,
            a_params=[b_i, b_i, 1.0 - a_i],
            b_params=[b_i + 1, b_i + 1],
            m=10
        )

        digb = digamma(b_i) - digamma(a_i + b_i) - np.log(1 - z_i + eps)
        right_b = coef_b * ((1.0 - z_i)**b_i) * G_b + bl_inv_t * digb
        grad_b = 2 * b_i * np.trapz(left * right_b, x_grid, axis=0)

        gradients_a[i] = grad_a
        gradients_b[i] = grad_b

    return gradients_a, gradients_b


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_robbins_monro_eta(t, eta0, tau, alpha):
    """
    Robbins-Monro learning rate schedule: η_t = η_0 / (t + τ)^α
    """
    return eta0 / (t + tau)**alpha


# ============================================================================
# Main Algorithm: Online Beta Calibration
# ============================================================================

def online_beta_calibration(x:np.ndarray,
                            y_observations:np.ndarray,
                            pool_func,
                            pool_param:dict,
                            geom_3f2,
                           # Initialization parameters
                           a0=1.0, b0=1.0,
                           # Learning rate parameters
                           gamma=0.5, eta0=0.01, tau=50, alpha=0.85,
                           # Algorithm 1: Shape detection parameters
                           significance_level=0.5,
                           k_bins_for_detection=5,
                           # Algorithm 2: Correction parameters
                           correction_strength=0.30,
                           param_floor=0.25,
                           param_ceiling=5.00,
                           x_constant=1.0,
                           # General parameters
                           min_samples=20,
                           outer_loop_freq=20,
                           rolling_window_size=100,
                           eps=1e-5):
    """
    Online Beta Calibration with Shape-Specific Corrections
    
    Args:
        forecast: Forecast data
        x: Grid points for evaluation
        weights: Model weights
        sigma_y: Observation noise standard deviation
        y_observations: Actual observations (length T)
        tlp: Function to compute linear pool CDFs
        geom_3f2: Generalized hypergeometric function
        
    Returns:
        results: Dictionary with 'a_hist', 'b_hist', 'all_calibrated_cdfs'
    """
    
    # Compute linear pool CDFs over grid
    TL = pool_func(**pool_param)  # Shape: (P, T, M)
    P, T, M = TL.shape

    print(f"Starting Online Beta Calibration")
    print(f"  Models (P): {P}")
    print(f"  Time steps (T): {T}")
    print(f"  Grid points (M): {M}")
    print(f"  Initial parameters: α₀={a0}, β₀={b0}")
    print(f"  Learning rate: η₀={eta0}, τ={tau}, α={alpha}")
    print(f"  Correction strength: {correction_strength}")
    print("-" * 60)

    # Initialize parameters
    a_c = np.full(P, a0, dtype=float)
    b_c = np.full(P, b0, dtype=float)
    A_inv = np.broadcast_to(np.eye(2) * (gamma**2), (P, 2, 2)).copy()

    # History tracking
    pit_history_by_model = [[] for _ in range(P)]
    pit_rolling_window_by_model = [[] for _ in range(P)]
    a_hist = [a_c.copy()]
    b_hist = [b_c.copy()]
    new_cdf_hist = []

    # Main Online Learning Loop
    for t in range(T):
        y_t = y_observations[t]
        eta_t = get_robbins_monro_eta(t, eta0, tau, alpha)

        # Interpolate to get CDF values at observation
        tl_at_yt = np.zeros(P)
        for p in range(P):
            interp_func = interp1d(x, TL[p, t, :], kind='linear',
                                  bounds_error=False, fill_value=(0, 1))
            tl_at_yt[p] = interp_func(y_t)

        # Compute CRPS gradients
        crps_grad_a, crps_grad_b = compute_crps_gradients(
            a_c, b_c, tl_at_yt, y_t, x, P, geom_3f2, eps
        )

        # Online Newton Step with BFGS
        new_cdf = np.zeros(P)

        for i in range(P):
            # Gradient vector
            g = np.array([[crps_grad_a[i]], [crps_grad_b[i]]], dtype=float)

            # Quasi-Newton direction with BFGS
            A_inv_g = A_inv[i] @ g

            # BFGS update of inverse Hessian
            denom = 1 + g.T @ A_inv_g
            if abs(denom) > eps:
                A_inv[i] -= (A_inv_g @ A_inv_g.T) / denom

            # Newton step: (η_t/γ) A_t^{-1}[i] ∇_t[i]
            step = (eta_t / gamma) * A_inv_g
            step_a = float(step[0, 0])
            step_b = float(step[1, 0])

            # Update parameters in log space for positivity
            new_log_a = np.log(a_c[i]) - step_a
            new_log_b = np.log(b_c[i]) - step_b

            # Clip to prevent overflow
            new_log_a = np.clip(new_log_a, np.log(0.1), np.log(50.0))
            new_log_b = np.clip(new_log_b, np.log(0.1), np.log(50.0))

            # Update parameters
            a_c[i] = np.exp(new_log_a)
            b_c[i] = np.exp(new_log_b)

            # Compute calibrated CDF value
            new_cdf[i] = np.clip(beta.cdf(tl_at_yt[i], a_c[i], b_c[i]), 
                                eps, 1 - eps)

        # Maintain PIT history
        for i in range(P):
            pit_history_by_model[i].append(new_cdf[i])
            pit_rolling_window_by_model[i].append(new_cdf[i])
            if len(pit_rolling_window_by_model[i]) > rolling_window_size:
                pit_rolling_window_by_model[i] = pit_rolling_window_by_model[i][-rolling_window_size:] 

        # Periodic Shape-Specific Corrections
        if t % outer_loop_freq == 0 and t > 0:
            if any(len(pit_rolling_window_by_model[i]) >= min_samples for i in range(P)):
                a_c, b_c, _ = apply_model_specific_corrections(
                    a_c, b_c, pit_rolling_window_by_model,
                    significance_level=significance_level,
                    k_bins_for_detection=k_bins_for_detection,
                    window=rolling_window_size,
                    min_samples=min_samples,
                    correction_strength=correction_strength,
                    param_floor=param_floor,
                    param_ceiling=param_ceiling,
                    x_constant=x_constant
                )

        # Conservative parameter bounds
        a_c = np.clip(a_c, 0.1, 4.0)
        b_c = np.clip(b_c, 0.1, 4.0)

        # Store history
        a_hist.append(a_c.copy())
        b_hist.append(b_c.copy())
        new_cdf_hist.append(new_cdf.copy())

        # Progress reporting
        if t % 200 == 0:
            print(f"Step {t:4d}: α={a_c.mean():.4f}, β={b_c.mean():.4f}")

    print("-" * 60)
    print("Online Beta Calibration Complete")
    
    results = {
        'a_hist': np.array(a_hist).T,
        'b_hist': np.array(b_hist).T,
        'all_calibrated_cdfs': np.array(new_cdf_hist).T
    }

    return results




def plot_parameter_evolution(beta_results, discount_factors):
    """
    Plot evolution of alpha and beta parameters for all discount factors
    
    Args:
        beta_results: Results from online_beta_calibration
        discount_factors: List/array of discount factor values (length P)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    a_hist = beta_results['a_hist']  # Shape: (P, T)
    b_hist = beta_results['b_hist']  # Shape: (P, T)
    
    P, T = a_hist.shape
    
    # Create figure with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Generate colors for each discount factor
    colors = plt.cm.viridis(np.linspace(0, 1, P))
    
    # Plot 1: Alpha parameters
    ax = axes[0]
    for i in range(P):
        ax.plot(range(T), a_hist[i, :], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'δ={discount_factors[i]:.3f}')
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, linewidth=1.5, 
               label='α=1 (no correction)')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Alpha (α)', fontsize=12)
    ax.set_title('Alpha Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2 if P > 6 else 1)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Beta parameters
    ax = axes[1]
    for i in range(P):
        ax.plot(range(T), b_hist[i, :], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'δ={discount_factors[i]:.3f}')
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, linewidth=1.5,
               label='β=1 (no correction)')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Beta (β)', fontsize=12)
    ax.set_title('Beta Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2 if P > 6 else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()









