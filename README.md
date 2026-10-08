# Online Recalibration of Combined Probabilistic Forecasts

Code for my MSc project at University College London (Department of Statistical Science), which won the MSc Best Project Prize.

When several probabilistic forecasts are combined, the resulting predictive distribution is often **miscalibrated**: its PIT (probability integral transform) values are not uniform, so the stated uncertainty is too wide or too narrow. This project builds an **online recalibration layer** that sits on top of a forecast combination and corrects it in real time, one observation at a time, without looking ahead.

The pipeline is:

1. **Combine** a pool of forecasters with **Loss-Discounting Frameworks (LDF)**, a two-level, discount-factor-based scheme for time-varying combination weights.
2. **Pool** the forecasters' Gaussian predictive densities into a linear pool, evaluated on a grid.
3. **Recalibrate** the pooled CDF online with a **Beta transform**, `G(F(y)) = Beta(F(y); a_t, b_t)`, where `(a_t, b_t)` are updated sequentially by second-order stochastic gradient steps on the CRPS, with periodic shape-specific corrections.
4. **Evaluate** calibration (PIT histograms and uniformity tests) and sharpness (mean log score) before and after recalibration.

The method is tested on a simulation study and on GBP/USD exchange-rate forecasting.

---

## Method overview

### 1. Forecast combination: LDF (`ldf.py`)
Given the per-period log predictive likelihoods of `K` forecasters, `ldf()` produces combination weights.

- **Level 1** discounts each forecaster's past log scores with a factor `α` and turns them into weights (softmax or argmax).
- **Level 2** discounts the level-1 performance with a factor `δ`, mixing over a grid of `α` values.

This is run for a grid of discount factors (`1, 0.99, 0.95, …, 0.001`), giving `P` combined forecasts, each with weights of shape `(T, K)`.

### 2. Linear pool on a grid (`utils/utils.py`, `pit.py`)
- `tlp()` / `ex_tlp()` compute the pooled CDF on a grid `x` for every combination `p` and time `t`, giving an array of shape `(P, T, M)`.
- `pit()` interpolates the pooled CDF at the realised `y_t` to obtain PIT values.

### 3. Online Beta calibration (`beta_transform.py`)
`online_beta_calibration()` processes observations sequentially:

- **Parameter updates:** `(a, b)` are updated with Newton-style steps using CRPS gradients (`compute_crps_gradients`, which uses the regularised hypergeometric function `₃F₂` in `geom_3f2`). A Robbins–Monro learning-rate schedule, `η_t = η₀ / (1 + t/τ)^α`, controls step sizes.
- **Algorithm 1, shape detection:** every `outer_loop_freq` steps, a chi-square test on recent PIT values is decomposed with an "ends" contrast to decide whether the PIT histogram is bell-shaped (over-dispersed forecast) or U-shaped (under-dispersed).
- **Algorithm 2, shape-specific corrections:** if a significant shape is detected, `(a, b)` are nudged in the corrective direction, kept within `[param_floor, param_ceiling]`.

It returns the parameter paths (`a_hist`, `b_hist`) and the recalibrated PIT values (`all_calibrated_cdfs`). `plot_parameter_evolution()` plots how `(a, b)` evolve over time.

### 4. Evaluation (`pit.py`, `Log_likelihood.py`)
- `plot_pit_histograms()`: PIT histograms per discount factor, with the uniform reference line.
- `test_and_interpret_pit()`: uniformity tests (e.g. Kolmogorov–Smirnov, chi-square) with a summary table.
- `log_score_comparison()`: mean log score before vs. after calibration. The calibrated density is `beta.pdf(F(y); a, b) · f(y)`.

---

## Repository structure

```
nsf_pit/
├── Simulation_study.ipynb     # Simulated regime-switching DGP with 20 forecasters
├── Exchange_Rate_Study.ipynb  # GBP/USD exchange-rate application
├── ldf.py                     # Loss-Discounting Framework (forecast combination)
├── beta_transform.py          # Online Beta calibration and shape detection/correction
├── pit.py                     # PIT computation, pooled densities, PIT tests and plots
├── Log_likelihood.py          # Pooled pdf and log-score comparison
├── utils/
│   ├── utils.py               # Linear pool CDFs (tlp, ex_tlp) and 3F2 function
│   ├── torchutils.py          # PyTorch helpers
│   └── typechecks.py
├── data/
│   ├── Y.xlsx                 # Realised exchange-rate series
│   ├── y_t_pred.mat           # Individual forecasters' predictive means
│   ├── sigma.mat              # Predictive variances
│   └── log lik ind.mat        # Individual log predictive likelihoods
└── requirements.txt
```

---

## Experiments

### Simulation study (`Simulation_study.ipynb`)
- The data-generating process is an AR-type process whose mean shifts between levels `{-1, 0, 1}` across `T = 2001` steps.
- There are `K = 20` Gaussian forecasters, each centred on a fixed level in `[-2, 2]`.
- LDF combines them, and the Beta transform recalibrates the pool. PIT histograms, uniformity tests and log scores are compared before and after.
- Commented-out cells contain alternative DGPs with Markov-switching levels (time-constant and time-varying transition matrices).

### Exchange-rate study (`Exchange_Rate_Study.ipynb`)
- The target is GBP/USD (`currency_idx = 8`) over the last 342 periods.
- Individual forecasters' means, variances and log likelihoods are loaded from `data/`.
- The same LDF, linear pool, Beta calibration and evaluation pipeline is applied, with combined point forecasts plotted for `δ = 1` and `δ = 0.001`.

---

## Getting started

```bash
git clone https://github.com/callum-0214/MSc-Project.git
cd MSc-Project/nsf_pit
pip install -r requirements.txt
pip install pandas openpyxl seaborn jupyter
jupyter notebook
```

Open either notebook and run all cells. The notebooks import the modules directly, so run them from inside `nsf_pit/`.

### Minimal usage

```python
import numpy as np
from ldf import ldf
from utils.utils import tlp, geom_3f2
from beta_transform import online_beta_calibration

discount_factors = np.array([1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.001])

# log_likelihoods: (K, T) array of forecasters' log predictive likelihoods
res = ldf(log_likelihoods, levels=2,
          discount_factors=[discount_factors, discount_factors],
          activation_functions=["softmax", "softmax"], c=1e-20)

results = online_beta_calibration(
    x=x_grid, y_observations=y,
    pool_func=tlp,
    pool_param={"forecast": forecasts, "x": x_grid,
                "weights": res["weights"], "sigma_y": sigma_y},
    geom_3f2=geom_3f2,
    a0=1.0, b0=1.0, gamma=0.05, eta0=1.2, tau=50, alpha=0.55,
    outer_loop_freq=70, min_samples=40,
)

calibrated_pits = results["all_calibrated_cdfs"]  # (P, T)
```

### Key calibration hyperparameters

| Parameter | Meaning |
|---|---|
| `a0`, `b0` | Initial Beta parameters (1, 1 is the identity map) |
| `gamma` | Scale of the initial inverse-Hessian for the Newton updates |
| `eta0`, `tau`, `alpha` | Robbins–Monro learning-rate schedule |
| `outer_loop_freq` | How often shape detection and correction run |
| `min_samples` | Minimum PIT history before corrections are allowed |
| `significance_level`, `k_bins_for_detection` | Chi-square shape-detection test settings |
| `correction_strength`, `param_floor`, `param_ceiling` | Size and bounds of shape corrections |
| `rolling_window_size` | Window of recent PITs used for detection |

---

## Author

**Callum Wu**, Department of Statistical Science, University College London
