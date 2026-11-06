"""
Prediction of conduction velocity and delay from white matter microstructure.

Fits log(velocity) ~ axon radius + f + deperp + qT1 on cortico-cortical edges with
F-TRACT data, estimates the prediction error by bootstrap, predicts velocity and delay
for all edges (including those missing in F-TRACT), and produces the figure panels:

  Prediction figure   A  original velocity / delay matrices
                      B  predicted velocity / delay matrices
                      C  histograms of F-TRACT vs predicted values
  Uncertainty figure  A  violin plots of the prediction SD
                      B  prediction SD matrices
                      C  original vs predicted values with a robust (Huber) fit

Inputs  : edge tables written by 01_microstructure-velocity_stats.R (one row per edge,
          upper triangle of the 68 x 68 cortical matrix in column-major order)
Outputs : PDF panels in FIG_DIR, velocity and delay matrices (CSV) in DATA_DIR

Run as a script (python 02_velocity-prediction_figures.py) or cell by cell
(# %% markers) in VS Code, Spyder or Jupyter.
"""

# %% Setup --------------------------------------------------------------------
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.multitest import multipletests

# Paths (edit to your own)
DATA_DIR = "path/to/output"   # same folder as out_dir in 01_microstructure-velocity_stats.R
FIG_DIR = os.path.join(DATA_DIR, "figs")
FILTERED_CSV = "filtered_micro_delay_dcm10thr_ctx.csv"  # filtered edges (training data)
ALL_CSV = "all_micro_delay_dcm10thr_ctx.csv"            # all cortico-cortical edges

N_REGIONS = 68          # cortical regions, 34 per hemisphere
PREDICTORS = ["axond", "f", "deperp", "t1"]
N_BOOT = 1000           # number of bootstrap iterations
SEED = 123
G_RATIO = 0.6           # assumed g-ratio for the Rushton model
SHOW = False            # True to display the figures when running interactively

os.makedirs(FIG_DIR, exist_ok=True)


def finish(fig, filename):
    """Save a figure to FIG_DIR, then show or close it."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def error_metrics(y_true, y_pred):
    """RMSE, MAE and median absolute percentage error (%)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = 100 * np.median(np.abs((y_true - y_pred) / y_true))
    return rmse, mae, mape


def report(name, metrics):
    """Print mean +/- SD [95% CI] of the bootstrapped error metrics."""
    metrics = np.asarray(metrics)
    for i, label in enumerate(["RMSE", "MAE", "MAPE"]):
        values = metrics[:, i]
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        print(f"{name} {label}: {values.mean():.3f} +/- {values.std():.3f} "
              f"[95% CI: {ci_low:.3f} - {ci_high:.3f}]")


def vec_to_sym_matrix(vec, size=N_REGIONS):
    """Rebuild a symmetric matrix from its upper triangle, stored column by column
    (the order used by R's m[upper.tri(m)]). The diagonal is set to 0."""
    vec = np.asarray(vec, dtype=float)
    if len(vec) != size * (size - 1) // 2:
        raise ValueError(f"Vector length {len(vec)} does not match a {size} x {size} matrix.")
    mat = np.full((size, size), np.nan)
    k = 0
    for col in range(size):
        for row in range(col):
            mat[row, col] = vec[k]
            mat[col, row] = vec[k]
            k += 1
    np.fill_diagonal(mat, 0)
    return mat


# %% Load data ----------------------------------------------------------------
df_filtered = pd.read_csv(os.path.join(DATA_DIR, FILTERED_CSV))
df_all = pd.read_csv(os.path.join(DATA_DIR, ALL_CSV))

# %% 1. Prediction model -------------------------------------------------------
m_main = smf.ols("np.log(vel) ~ axond + f + deperp + t1", data=df_filtered).fit()
print(m_main.summary())
fdr_result = multipletests(m_main.pvalues, alpha=0.05, method="fdr_bh")
p_fdr = pd.Series(fdr_result[1], index=m_main.pvalues.index)
print("FDR-corrected p-values:")
print(p_fdr)

# %% 2. Comparison with a Rushton (single-axon) model -------------------------
# Velocity proportional to d * sqrt(-ln g), with d = 2 * axon radius; fitted in log-log space
df_filtered["log_rushton"] = np.log(2 * df_filtered["axond"] * np.sqrt(-np.log(G_RATIO)))
m_rushton = smf.ols("np.log(vel) ~ log_rushton", data=df_filtered).fit()
print(m_rushton.summary())
print(f"R-squared   Rushton: {m_rushton.rsquared:.4f} | Multivariate: {m_main.rsquared:.4f}")
print(f"Derived Rushton constant k: {np.exp(m_rushton.params['Intercept']):.4f}")
print(f"AIC         Rushton: {m_rushton.aic:.2f} | Multivariate: {m_main.aic:.2f} | "
      f"Delta AIC: {m_rushton.aic - m_main.aic:.2f}")
print(f"Log-likelihood improvement: {m_main.llf - m_rushton.llf:.2f}")


# %% 3. Bootstrap prediction of velocity ----------------------------------------
def bootstrap_velocity(train, features, predict_on=None):
    """In each iteration, fit log(vel) ~ features on a bootstrap sample (80% of the
    edges with F-TRACT data, drawn with replacement) and predict the edges left out
    of that sample (test set).

    Returns:
      metrics   test-set error metrics, one row per iteration
      pred_test predictions for the test-set edges (edges x iterations, NaN when an
                edge was in the training sample)
      pred_all  predictions for all edges of `predict_on` (edges x iterations),
                or None if `predict_on` is not given
    """
    np.random.seed(SEED)
    valid = np.where(train["vel"].notna() & train["delay"].notna())[0]
    n_train = int(0.8 * len(valid))
    pred_test = np.full((len(train), N_BOOT), np.nan)
    pred_all = None
    if predict_on is not None:
        pred_all = np.full((len(predict_on), N_BOOT), np.nan)
    metrics = []

    for i in range(N_BOOT):
        idx_train = np.random.choice(valid, size=n_train, replace=True)
        idx_test = np.setdiff1d(valid, idx_train)

        X_train = sm.add_constant(train.loc[idx_train, features])
        y_train = np.log(train.loc[idx_train, "vel"])
        model = sm.OLS(y_train, X_train).fit()

        if len(idx_test) > 0:
            X_test = sm.add_constant(train.loc[idx_test, features])
            pred = np.exp(model.predict(X_test))
            pred_test[idx_test, i] = pred
            metrics.append(error_metrics(train.loc[idx_test, "vel"], pred))

        if predict_on is not None:
            X_all = sm.add_constant(predict_on[features])
            pred_all[:, i] = np.exp(model.predict(X_all))

    return np.array(metrics), pred_test, pred_all


metrics_vel, pred_test, pred_all = bootstrap_velocity(df_filtered, PREDICTORS, predict_on=df_all)
metrics_rushton, _, _ = bootstrap_velocity(df_filtered, ["log_rushton"])
report("Velocity", metrics_vel)
report("Rushton velocity", metrics_rushton)

# %% 4. Predicted velocity and delay: mean and SD across bootstrap iterations ------
# Delay is computed in each iteration as length / predicted velocity.
# Training edges are only predicted in the iterations where they were left out.
len_filtered = df_filtered["len"].values.reshape(-1, 1)   # one column, divides every iteration
len_all = df_all["len"].values.reshape(-1, 1)
delay_test = len_filtered / pred_test
delay_all = len_all / pred_all

# Edges without any prediction give an all-NaN row; numpy warns about these, the result is NaN
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    df_filtered["predicted_vel"] = np.nanmean(pred_test, axis=1)
    df_filtered["predicted_vel_sd"] = np.nanstd(pred_test, axis=1)
    df_filtered["predicted_delay"] = np.nanmean(delay_test, axis=1)
    df_filtered["predicted_delay_sd"] = np.nanstd(delay_test, axis=1)

    df_all["predicted_vel"] = np.nanmean(pred_all, axis=1)
    df_all["predicted_vel_sd"] = np.nanstd(pred_all, axis=1)
    df_all["predicted_delay"] = np.nanmean(delay_all, axis=1)
    df_all["predicted_delay_sd"] = np.nanstd(delay_all, axis=1)

# Edges without microstructure (axond and f both 0) are set to 0
no_micro = (df_all["axond"] == 0) & (df_all["f"] == 0)
pred_cols = ["predicted_vel", "predicted_vel_sd", "predicted_delay", "predicted_delay_sd"]
df_all.loc[no_micro, pred_cols] = 0

# %% 5. Delay prediction error (bootstrap over edges) ------------------------------
has_delay = (df_filtered["delay"].notna()
             & df_filtered["predicted_delay"].notna()
             & (df_filtered["delay"] != 0))
true_delay = df_filtered.loc[has_delay, "delay"].values
pred_delay = df_filtered.loc[has_delay, "predicted_delay"].values

np.random.seed(SEED)
metrics_delay = []
for i in range(N_BOOT):
    idx = np.random.choice(len(true_delay), len(true_delay), replace=True)
    metrics_delay.append(error_metrics(true_delay[idx], pred_delay[idx]))
report("Delay", metrics_delay)

# %% 6. Rebuild and save the 68 x 68 matrices ---------------------------------------
# Original values are set to 0 where there is no axon radius estimate
original = df_all[["vel", "delay"]].copy()
original.loc[df_all["axond"] == 0] = 0

mat_vel = vec_to_sym_matrix(original["vel"])
mat_vel_pred = vec_to_sym_matrix(df_all["predicted_vel"])
mat_vel_sd = vec_to_sym_matrix(df_all["predicted_vel_sd"])
mat_delay = vec_to_sym_matrix(original["delay"])
mat_delay_pred = vec_to_sym_matrix(df_all["predicted_delay"])
mat_delay_sd = vec_to_sym_matrix(df_all["predicted_delay_sd"])

np.savetxt(os.path.join(DATA_DIR, "velocity_dcm10thr_matrix_ctx.csv"), mat_vel, delimiter=",", fmt="%.5f")
np.savetxt(os.path.join(DATA_DIR, "predicted_velocity_dcm10thr_matrix_ctx.csv"), mat_vel_pred, delimiter=",", fmt="%.5f")
np.savetxt(os.path.join(DATA_DIR, "delay_dcm10thr_matrix_ctx.csv"), mat_delay, delimiter=",", fmt="%.5f")
np.savetxt(os.path.join(DATA_DIR, "predicted_delay_dcm10thr_matrix_ctx.csv"), mat_delay_pred, delimiter=",", fmt="%.5f")


# %% Prediction figure, A-B: original and predicted matrices ----------------------
def plot_matrix(mat, vmax, filename):
    fig, ax = plt.subplots(figsize=(8, 7))
    # rasterized=True avoids thin white lines between the cells in the PDF
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=vmax, square=True,
                xticklabels=False, yticklabels=False, rasterized=True, ax=ax)
    finish(fig, filename)


mat_vel_plot = np.where(mat_vel > 80, 0, mat_vel)   # implausible velocities (> 80 m/s) set to 0
vmax_vel = max(np.nanmax(mat_vel_plot), np.nanmax(mat_vel_pred))
vmax_delay = max(np.nanmax(mat_delay), np.nanmax(mat_delay_pred))

plot_matrix(mat_vel_plot, vmax_vel, "matrix_velocity_original.pdf")
plot_matrix(mat_vel_pred, vmax_vel, "matrix_velocity_predicted.pdf")
plot_matrix(mat_delay, vmax_delay, "matrix_delay_original.pdf")
plot_matrix(mat_delay_pred, vmax_delay, "matrix_delay_predicted.pdf")


# %% Prediction figure, C: histograms ---------------------------------------------
def plot_histograms(observed, predicted, missing, quantity, unit, filename):
    """F-TRACT values, predictions for edges with F-TRACT data, and predictions
    for edges missing in F-TRACT."""
    bins = np.linspace(0, 80, 41)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(observed, bins=bins, stat="frequency", color="darkblue", alpha=0.7,
                 label=f"F-TRACT {quantity}", ax=ax)
    sns.histplot(predicted, bins=bins, stat="frequency", color="darkred", alpha=0.6,
                 label=f"Predicted {quantity}", ax=ax)
    sns.histplot(missing, bins=bins, stat="frequency", color="darkgreen", alpha=0.6,
                 label=f"Predicted {quantity} for missing F-TRACT data", ax=ax)
    ax.set_xlabel(f"Value [{unit}]")
    ax.set_ylabel("Frequency")
    ax.legend()
    finish(fig, filename)


vel = df_all["vel"]
vel_pred = df_all["predicted_vel"]
vel_observed = vel[vel.notna() & (vel != 0) & (vel < 80)]
vel_predicted = vel_pred[vel.notna() & (vel_pred != 0) & (vel_pred < 80)]
vel_missing = vel_pred[vel.isna() & (vel_pred != 0) & (vel_pred < 80)]
plot_histograms(vel_observed, vel_predicted, vel_missing, "Velocity", "m/s", "histogram_velocity.pdf")

delay = df_all["delay"]
delay_pred = df_all["predicted_delay"]
delay_observed = delay[delay.notna() & (delay != 0)]
delay_predicted = delay_pred[delay.notna() & (delay_pred != 0)]
delay_missing = delay_pred[delay.isna() & (delay_pred != 0)]
plot_histograms(delay_observed, delay_predicted, delay_missing, "Delay", "ms", "histogram_delay.pdf")
print(f"Median delay   F-TRACT: {np.median(delay_observed):.2f} | "
      f"predicted: {np.median(delay_predicted):.2f} | "
      f"predicted for missing: {np.median(delay_missing):.2f}")


# %% Uncertainty figure, A: violin plots of the prediction SD (filtered edges) -------
def plot_violin(values, color, ylabel, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(y=values, inner="quartile", color=color, ax=ax)
    ax.set_ylabel(ylabel)
    finish(fig, filename)


has_vel_pred = df_filtered["predicted_vel"].notna() & (df_filtered["predicted_vel"] != 0)
has_delay_pred = (df_filtered["predicted_delay"].notna()
                  & (df_filtered["predicted_delay"] != 0)
                  & df_filtered["predicted_delay_sd"].notna())
sd_vel = df_filtered.loc[has_vel_pred, "predicted_vel_sd"]
sd_delay = df_filtered.loc[has_delay_pred, "predicted_delay_sd"]
print(f"Edges in violin plots: velocity {len(sd_vel)}, delay {len(sd_delay)}")

plot_violin(sd_vel, "skyblue", "Predicted Conduction Velocity (SD) [m/s]", "violin_velocity_sd.pdf")
plot_violin(sd_delay, "lightcoral", "Predicted Conduction Delay (SD) [ms]", "violin_delay_sd.pdf")

# %% Uncertainty figure, B: prediction SD matrices -----------------------------
plot_matrix(mat_vel_sd, 5, "matrix_velocity_sd.pdf")
plot_matrix(mat_delay_sd, 5, "matrix_delay_sd.pdf")


# %% Uncertainty figure, C: original vs predicted with a robust fit ----------------
def plot_robust_fit(predicted, original, quantity, filename):
    """Scatter plot coloured by point density, Huber regression of the original on the
    predicted values, and the line of identity. In the legend, d_o and d_p are the
    original and predicted data."""
    data = pd.DataFrame({"pred": predicted, "orig": original}).dropna()
    x = data["pred"].values
    y = data["orig"].values

    huber = HuberRegressor().fit(x.reshape(-1, 1), y)   # sklearn expects a 2D input
    intercept = huber.intercept_
    slope = huber.coef_[0]
    print(f"Robust fit ({quantity}): original = {intercept:.3f} + {slope:.3f} * predicted")

    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    lowest = min(x.min(), y.min())
    highest = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c=density, s=15, cmap="viridis", alpha=0.7)
    ax.plot(x, huber.predict(x.reshape(-1, 1)), color="orange", linewidth=1.5)
    ax.plot([lowest, highest], [lowest, highest], "r--", linewidth=1.5)
    ax.set_xlabel(f"Predicted {quantity}")
    ax.set_ylabel(f"Original {quantity}")
    ax.grid(True)

    fit_label = f"$d_o$ = {intercept:.3f} + {slope:.3f} ($d_p$)"
    legend_lines = [Line2D([0], [0], color="orange", linewidth=1.5, label=fit_label),
                    Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Line of identity")]
    ax.legend(handles=legend_lines, loc="lower right", frameon=True)
    finish(fig, filename)


plot_robust_fit(df_filtered["predicted_vel"], df_filtered["vel"], "Velocity", "robust_fit_velocity.pdf")
plot_robust_fit(df_filtered["predicted_delay"], df_filtered["delay"], "Delay", "robust_fit_delay.pdf")
