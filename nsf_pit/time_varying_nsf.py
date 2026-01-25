import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.optim as optim
from rational_quadratic import rational_quadratic_spline
from nn import ResidualNet
from torch import nn
from torch.nn import functional as F
from scipy.stats import kstest, anderson, jarque_bera
from scipy.stats import norm, chi2
import math
from beta_transform import compute_crps_gradients
from scipy.stats import beta, chi2, norm
from scipy.special import gamma as gamma_fn, digamma, betaln
from scipy.interpolate import interp1d

def extract_features(F_t, x_grid):
    """
    Extract functional features from CDF.
    Helper function to be used inside tvnsf_pit.
    """
    # Ensure inputs are numpy arrays for gradient/trapz operations
    if isinstance(F_t, torch.Tensor):
        F_t = F_t.detach().cpu().numpy()
        
    pdf = np.gradient(F_t, x_grid)
    
    # Calculate moments
    mean = np.trapz(x_grid * pdf, x_grid)
    variance = np.trapz((x_grid - mean)**2 * pdf, x_grid)
    
    # Handle skewness stability
    denom = variance**1.5 + 1e-8
    skewness = np.trapz((x_grid - mean)**3 * pdf, x_grid) / denom
    
    # Tail properties
    left_tail = F_t[int(0.1 * len(x_grid))]
    right_tail = 1 - F_t[int(0.9 * len(x_grid))]
    
    # Curvature (smoothness)
    curvature = np.mean(np.abs(np.gradient(pdf, x_grid)))
    
    return np.array([mean, variance, skewness, left_tail, right_tail, curvature])


def compute_crps_per_step(calibrated_cdf_t, x_grid, obs_index):
    """
    Computes CRPS for a single time step using grid approximation.
    CRPS = Integral( (F_cal(x) - H(x - y))^2 dx )
    """
    # Create Heaviside step function H(x - y)
    # It is 0 before the observation index, and 1 after
    heaviside = np.zeros_like(x_grid)
    heaviside[obs_index:] = 1.0
    
    # Squared difference
    diff_sq = (calibrated_cdf_t - heaviside) ** 2
    
    # Integrate using Trapezoidal rule
    crps = np.trapz(diff_sq, x_grid)
    return crps

def compute_ece(pit_values, n_bins):
    """
    Computes ECE for PIT values using manual binning logic.
    Adapted from _ECELoss to compare Observed vs Expected Frequency.
    """
    # Define boundaries exactly as in the snippet
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Logic based on snippet: in_bin mask
        in_bin = (pit_values > bin_lower) & (pit_values <= bin_upper)
        
        # prop_in_bin: The Observed Frequency (equivalent to 'accuracy' in classif.)
        prop_in_bin = np.mean(in_bin)
        
        # expected_prop: The Ideal Frequency (equivalent to 'confidence' if perfect)
        # For Uniform PITs, this is just the bin width.
        expected_prop = bin_upper - bin_lower
        
        # ECE calculation: Sum of absolute differences
        ece += np.abs(prop_in_bin - expected_prop)
        
    return ece



class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual  # The "Skip Connection"
        return nn.functional.relu(out)
    
    
##########################################################################


def tvnsf_pit(
        K,
        x,
        y,
        num_epochs,
        pit_data,
        cdf_data,
        discount_factors,
        encoding_dim,      
        hidden_size,      
        seed=42
):
    """
    Time-Varying Neural Spline Flow for PIT recalibration
    
    Args:
        K: Number of spline bins
        num_epochs: Training epochs
        pit_data: PIT values, shape (num_sequences, T) where T is sequence length
        discount_factors: Discount factors for each sequence
        encoding_dim: Dimension of GRU encoding (N in the math)
        hidden_size: Hidden layer size for parameter network
        seed: Random seed
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # For univariate PITs, we don't need to split!
    # pit_data shape: (num_sequences, T)
    # Each row is a sequence of PIT values over time
    
    spline_params_size = 3 * K + 1  # K widths, K heights, K-1 derivatives
    INPUT_FEATURE_DIM = 6

    results = []
    
    for i, delta in enumerate(discount_factors):
        print(f"\n{'='*60}")
        print(f"Training for δ = {delta}")
        print(f"{'='*60}")
        
        # Prepare data - single sequence of PITs
        pit_sequence = pit_data[i].copy()  # shape: (T,)
        cdf_sequence = cdf_data[i]         # shape: (T, M)
        T = len(pit_sequence)
        
        pit_tensor = torch.tensor(pit_sequence, dtype=torch.float32)  # shape: (T,)

        features_list = []
        for t in range(T):
            # Extract features for this specific time step
            feats = extract_features(cdf_sequence[t], x)
            features_list.append(feats)
        
        features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32) # shape: (T, 7)

        # ========================================
        # Define GRU for encoding
        # ========================================
        gru = nn.GRU(
            input_size=INPUT_FEATURE_DIM,           # univariate PIT values
            hidden_size=encoding_dim,
            batch_first=True
        )
        
        # ========================================
        # Define NN to map encoding -> spline parameters
        # ========================================
        param_net = nn.Sequential(
            nn.Linear(encoding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, spline_params_size)
        )
        
        # ========================================
        # Collect all parameters for optimization
        # ========================================
        all_params = list(gru.parameters()) + list(param_net.parameters())
        optimizer = optim.Adam(all_params, lr=1e-4)

        losses = []
    
        
        # ========================================
        # Training loop
        # ========================================
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # -------------------------------------
            # STEP 1: Prepare input sequence for GRU
            # -------------------------------------
            dummy_input = torch.zeros(1, INPUT_FEATURE_DIM)
            gru_input_seq = torch.cat([dummy_input, features_tensor[:-1]])
            gru_input = gru_input_seq.unsqueeze(0)  # Shape: (1, T, 6)
            
            # -------------------------------------
            # STEP 2: Compute encodings via GRU
            # -------------------------------------
            e_0 = torch.zeros(1, 1, encoding_dim)  # Initial hidden state
            encodings, _ = gru(gru_input, e_0)  # shape: (1, T, encoding_dim)
            encodings = encodings.squeeze(0)  # shape: (T, encoding_dim)
            
            # -------------------------------------
            # STEP 3: Generate time-varying spline parameters
            # -------------------------------------
            theta_all = param_net(encodings)  # (T, 3K+1)
            
            theta_w_all = theta_all[:, :K]           # (T, K)
            theta_h_all = theta_all[:, K:2*K]        # (T, K)
            theta_d_all = theta_all[:, 2*K:]         # (T, K+1)
            
            # VECTORIZED: Transform all time steps at once
            v_all, logabsdet_all, _ = rational_quadratic_spline(
                inputs=pit_tensor,
                unnormalized_widths=theta_w_all,
                unnormalized_heights=theta_h_all,
                unnormalized_derivatives=theta_d_all,
                inverse=False
            )
            
            # Compute loss (average negative log-likelihood)
            loss = -logabsdet_all.mean()
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()


            losses.append(loss.item())

            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():8.4f}")
        
        # ========================================
        # Evaluation
        # ========================================
        gru.eval()
        param_net.eval()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, INPUT_FEATURE_DIM)
            gru_input_seq = torch.cat([dummy_input, features_tensor[:-1]])
            gru_input = gru_input_seq.unsqueeze(0)

            e_0 = torch.zeros(1, 1, encoding_dim)
            encodings, _ = gru(gru_input, e_0)
            encodings = encodings.squeeze(0)
            
            theta_all = param_net(encodings)
            theta_w_all = theta_all[:, :K]
            theta_h_all = theta_all[:, K:2*K]
            theta_d_all = theta_all[:, 2*K:]
            
            v_transformed, log_likelihood, derivative = rational_quadratic_spline(
                inputs=pit_tensor,
                unnormalized_widths=theta_w_all,
                unnormalized_heights=theta_h_all,
                unnormalized_derivatives=theta_d_all,
                inverse=False
            )
            
            u_original = pit_tensor.numpy()
            v_transformed = v_transformed.numpy()
            log_likelihood = log_likelihood.numpy()
            derivative = derivative.numpy()

            crps_history = []

            

            ece_score = compute_ece(v_transformed, n_bins=K) # 20 bins is standard for PITs
        
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
            'mean_after': v_transformed.mean(),
            'losses': losses,
            'gru_model': gru,
            'param_net': param_net,
            'features': features_tensor,
            'crps_history': crps_history,
            'ece_score': ece_score
        })
    
    return results




