import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.optim as optim
from rational_quadratic import rational_quadratic_spline
from nn import ResidualNet
from torch.nn import functional as F
from scipy.stats import kstest, anderson, jarque_bera
from scipy.stats import norm, chi2
import math


#####################################
# PIT function
#####################################

def pit(pit_vals: np.ndarray, x: np.ndarray,y: np.ndarray):
    P, T, M = pit_vals.shape

    pit_values = np.zeros((P, T))
    for p in range(P):
        for t in range(T):
            interp_func = interpolate.interp1d(x, pit_vals[p, t, :],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=1e-10)
            pit_values[p, t] = interp_func(y[t])

    return pit_values


#####################################
# exchange rate linear pool pdf function
#####################################


def ex_tlp_pdf(
        mean: np.ndarray,          # Shape (K, T) - forecast means
        var: np.ndarray,           # Shape (K, T) - forecast variances
        x: np.ndarray,             # Grid points
        weights: np.ndarray,       # Shape (P, T, K) - combination weights
        y: np.ndarray              # Observed values (T,)
) -> np.ndarray:

    K, T = mean.shape
    P, T_w, K_w = weights.shape
    M = x.shape[0]

    assert (T, K) == (T_w, K_w), 'Shape mismatch'
    assert mean.shape == var.shape

    pdf_values = np.zeros((K, T, M), dtype=float)

    for k in range(K):
        for t in range(T):
            # PDF instead of CDF
            pdf_values[k, t, :] = norm.pdf(x, loc= mean[k, t], scale=np.sqrt(var[k, t]) )

    # Linear pool: weighted combination of PDFs
    TL = np.einsum('ktm,ptk -> ptm', pdf_values, weights)

    pdf = pit(TL, x, y)  # Shape (T,)

    return pdf


#####################################
# exchange rate calibrated pdf function
#####################################


def ex_calibrated_ll(
        log_original_pdf: np.ndarray,
        results: list,
        discount_factors: np.ndarray,
):
    calibrated_ll = np.zeros_like(log_original_pdf)
    for i in range(len(discount_factors)):
        absdet_i = results[i]['log_likelihood']
        calibrated_ll[i] = log_original_pdf[i,:] + absdet_i
    return calibrated_ll




#####################################
# Plot PIT histograms
#####################################


def plot_pit_histograms(TLP: np.ndarray, model_names=None):
    
    P, T = TLP.shape
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(P)]
    
    # Create 4x3 grid (12 subplots total)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for p in range(P):        
        ax = axes_flat[p]
        pits = TLP[p, :]
        
        # Create histogram
        ax.hist(pits, bins=20, density=True, alpha=0.7)
        
        # Add uniform reference line
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Uniform', alpha=0.8)

        # Set labels and title
        ax.set_xlim(0, 1)
        ax.set_xlabel('PIT Values', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(labelsize=16)  # This sets font size for both x and y axes
        ax.set_title(f'δ = {model_names[p]}', fontsize=18, pad=10)
        ax.grid(True, alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def test_and_interpret_pit(
    pit_values: np.ndarray,
    discount_factors,
    model_names = None,
    alpha: float = 0.05,
    ):
    
    P, T = pit_values.shape
    if model_names is None:
        model_names = [f"δ = {i}" for i in discount_factors]
    if len(model_names) != P:
        raise ValueError("Length of model_names must match number of rows in pit_values (P).")

    results = []
    interpretation = []

    for i in range(P):
        pit_p = pit_values[i, :]

        # KS test
        ks_stat, ks_pvalue = stats.kstest(pit_p, 'uniform')

        # 2. Anderson-Darling test (more sensitive to tail deviations)
        # Transform to normal for AD test, then interpret
        normal_transformed = stats.norm.ppf(np.clip(pit_p, 1e-10, 1-1e-10))
        ad_stat, ad_critical_values, ad_significance_levels = anderson(normal_transformed, dist='norm')
        
        # 3. Jarque-Bera test (tests for normality of transformed data)
        jb_stat, jb_pvalue = jarque_bera(normal_transformed)

        # 4. Chi-square goodness of fit test
        # Divide [0,1] into bins and test if frequencies are equal
        n_bins = min(20, T//5)  # Use fewer bins for small samples
        observed_freq, bin_edges = np.histogram(pit_p, bins=n_bins, range=(0, 1))
        expected_freq = T / n_bins
        chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=n_bins-1)

        # 5. Simple descriptive statistics
        mean_pit = np.mean(pit_p)
        var_pit = np.var(pit_p)
        skew_pit = stats.skew(pit_p)
        kurt_pit = stats.kurtosis(pit_p)

        # 6. Coverage at different quantiles (should be close to the quantile value)
        coverage_05 = np.mean(pit_p <= 0.05)
        coverage_10 = np.mean(pit_p <= 0.10)
        coverage_90 = np.mean(pit_p <= 0.90)
        coverage_95 = np.mean(pit_p <= 0.95)

        # 7. KL divergence from uniform distribution
        # Create histogram and compute KL divergence
        n_bins_kl = min(30, T//3)  # Use more bins for KL divergence
        observed_freq, _ = np.histogram(pit_p, bins=n_bins_kl, range=(0, 1))
        observed_prob = observed_freq / T
        uniform_prob = np.ones(n_bins_kl) / n_bins_kl

        epsilon = 1e-10
        observed_prob = np.maximum(observed_prob, epsilon)
        
        kl_divergence = np.sum(observed_prob * np.log(observed_prob / uniform_prob))

        results.append({
                "Model": model_names[i],
                "Count": int(T),
                "KS_statistic": float(ks_stat),
                "KS_pvalue": float(ks_pvalue),
                "AD_statistic": float(ad_stat),
                "Chi2_statistic": float(chi2_stat),
                "Chi2_pvalue": float(chi2_pvalue),
                "JB_statistic": float(jb_stat),
                "JB_pvalue": float(jb_pvalue),
                "KL_divergence": kl_divergence,
                "Mean": mean_pit,
                "Variance": var_pit,
                "Skewness": skew_pit,
                "Kurtosis": kurt_pit,
                "Coverage_5%": coverage_05,
                "Coverage_10%": coverage_10,
                "Coverage_90%": coverage_90,
                "Coverage_95%": coverage_95,
            })
        
        # Interpretation based on p-values and statistics
        ks_good = ks_pvalue > alpha
        chi2_good = chi2_pvalue > alpha
        jb_good = jb_pvalue > alpha
        ad_good = False if np.isnan(ad_stat) else (ad_stat < 0.787)

        zcrit = norm.ppf(1 - alpha / 2)
        mean_good = abs(mean_pit - 0.5) <= zcrit * np.sqrt(1/(12*T))
        var_good = abs(var_pit - 1/12) <= zcrit * math.sqrt(1/(180*T))

        se = np.sqrt(0.05 * (1 - 0.05) / T)
        coverage_5_good = abs(coverage_05 - 0.05) <= zcrit * se
        coverage_95_good = abs(coverage_95 - 0.95) <= zcrit * se

        kl_good = kl_divergence < 0.01


        good_tests = sum(
            [
                ks_good,
                chi2_good,
                jb_good,
                ad_good,
                mean_good,
                var_good,
                coverage_5_good,
                coverage_95_good,
                kl_good,
            ]
        )

        interpretation.append(
            {
                "Model": model_names[i],
                "Tests_Passed": f"{good_tests}/9",
                "KS_Test": "PASS" if ks_good else "FAIL",
                "Chi2_Test": "PASS" if chi2_good else "FAIL",
                "JB_Test": "PASS" if jb_good else "FAIL",
                "AD_Test": "PASS" if ad_good else "FAIL",
                "Mean_Test": "PASS" if mean_good else "FAIL",
                "Variance_Test": "PASS" if var_good else "FAIL",
                "KL_Test": "PASS" if kl_good else "FAIL",
                "Coverage_5%_Test": "PASS" if coverage_5_good else "FAIL",
                "Coverage_95%_Test": "PASS" if coverage_95_good else "FAIL"
            }
        ) 

    tests_df = pd.DataFrame(results)
    interpretation_df = pd.DataFrame(interpretation)

    return tests_df, interpretation_df


def split_equal(Raw_pit):
    num_cols = Raw_pit.shape[1]

    # if odd, drop last column
    if num_cols % 2 == 1:
        Raw_pit = Raw_pit[:, :-1]

    # now guaranteed even
    d = Raw_pit.shape[1] // 2

    first_half  = Raw_pit[:, :d]
    second_half = Raw_pit[:, d:]

    return first_half, second_half, d


def plot_ecdf(ax, data, title, color='blue'):
    """Helper to plot ECDF."""
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    ecdf = np.arange(1, n+1) / n
    
    ax.plot(sorted_vals, ecdf, color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label="45° line")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("PIT Value", fontsize=10)
    ax.set_ylabel("ECDF", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')



def nsf_pit(
        K,
        num_epochs,
        pit_data,
        discount_factors,
        seed=42
):
    
    torch.manual_seed(seed)

    first_half, second_half, d = split_equal(pit_data)

    spline_params_size = 3 * K + 1

    results = []

    for i, delta in enumerate(discount_factors):
        print(f"\n{'='*60}")
        print(f"Training for δ = {delta}")
        print(f"{'='*60}")
        
        # Prepare data
        first_half_data = first_half[i].copy()
        second_half_data = second_half[i].copy()
        
        first_half_tensor = torch.tensor(first_half_data, dtype=torch.float32).unsqueeze(0)
        second_half_tensor = torch.tensor(second_half_data, dtype=torch.float32)
        first_half_tensor.requires_grad = False
        second_half_tensor.requires_grad = False
        
        # Create model
        residual_net = ResidualNet(
            in_features=first_half_tensor.shape[1],
            out_features=spline_params_size,
            hidden_features=64,
            context_features=None,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.1,
            use_batch_norm=False
        )

        optimizer = optim.Adam(residual_net.parameters(), lr=1e-3)
    
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            theta = residual_net(first_half_tensor)
            theta_w = theta[:, :K]
            theta_h = theta[:, K:2*K]
            theta_d = theta[:, 2*K:]
            
            batch_size = second_half_tensor.shape[0]
            theta_w_expanded = theta_w.expand(batch_size, -1)
            theta_h_expanded = theta_h.expand(batch_size, -1)
            theta_d_expanded = theta_d.expand(batch_size, -1)
            
            v_i, logabsdet_i, derivative_i = rational_quadratic_spline(
                inputs=second_half_tensor,
                unnormalized_widths=theta_w_expanded,
                unnormalized_heights=theta_h_expanded,
                unnormalized_derivatives=theta_d_expanded,
                inverse=False
            )
            
            loss = -logabsdet_i.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():8.4f}")
        
        # Evaluate
        residual_net.eval()
        with torch.no_grad():
            theta = residual_net(first_half_tensor)
            theta_w = theta[:, :K].expand(batch_size, -1)
            theta_h = theta[:, K:2*K].expand(batch_size, -1)
            theta_d = theta[:, 2*K:].expand(batch_size, -1)
            
            v_transformed, log_likelihood, derivative = rational_quadratic_spline(
                inputs=second_half_tensor,
                unnormalized_widths=theta_w,
                unnormalized_heights=theta_h,
                unnormalized_derivatives=theta_d,
                inverse=False
            )
            
            u_original = second_half_tensor.numpy()
            v_transformed = v_transformed.numpy()
            log_likelihood = log_likelihood.numpy()
            derivative = derivative.numpy()
        
        # Calculate statistics
        ks_before = stats.kstest(u_original, 'uniform')
        ks_after = stats.kstest(v_transformed, 'uniform')
        
        # Store results
        results.append({
            'delta': delta,
            'u_original': u_original,
            'v_transformed': v_transformed,
            'log_likelihood': log_likelihood,
            'derivative': derivative,
            'ks_before': ks_before.statistic,
            'ks_after': ks_after.statistic,
            'mean_before': u_original.mean(),
            'mean_after': v_transformed.mean()
        })
    
    return results


######################################
# PIT CRPS function
#######################################


        