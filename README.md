# Empirical Robustness Analysis of Zero-Shot Time-Series Foundation Models (TimesFM) Under Heavy-Tailed Noise

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: TimesFM 2.5 (200M)](https://img.shields.io/badge/Model-TimesFM%202.5%20(200M)-orange.svg)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An empirical stress test of Google's **TimesFM 2.5 (200M)** under heavy-tailed perturbations drawn from a Student-$t$ distribution with $\nu=2$. The project also evaluates **post-hoc rolling Median–Interquartile Range (IQR) clipping** as a lightweight, training-free mitigation method.

> **Scope:** The reported experiments use a synthetic time series and five random seeds. The results should be interpreted as findings from this controlled experiment, not as evidence of performance on all real-world time series.

## Forecast comparison

![Forecast comparison](timesfm_robustness_plot.png)

*Representative forecast trajectories from Seed 404 over a forecast horizon of $H=64$. The quantitative table below reports the five-seed aggregate results.*

---

## 1. Research motivation

Time-series foundation models use input processing and patch-based representations to prepare sequences for forecasting. A common normalization form is:

$$\widetilde{x}_n = \frac{x_n-\mu_x}{\sigma_x+\delta}$$

where $\mu_x$ and $\sigma_x$ are the input-window mean and standard deviation, and $\delta$ is a small numerical-stability constant.

In this experiment, the observed series is modelled as:

$$x_n = s(t_n) + \epsilon_n, \qquad \epsilon_n \stackrel{\text{i.i.d.}}{\sim} t_\nu(0,\gamma^2)$$

with $\nu=2$ and $\gamma=2.5$. The Student-$t$ distribution has infinite theoretical population variance when $\nu\leq 2$. A finite sample still has a finite empirical variance, but large outlier observations can sharply increase the sample variance and affect normalization.

## 2. Median–IQR mitigation

To reduce extreme observations without fine-tuning the forecasting model, the experiment applies a rolling Median–IQR clipping operation before inference:

$$\widehat{x}_n = \text{clip}\Big(x_n,\; M_w(x_n) - k\,\text{IQR}_w(x_n),\; M_w(x_n) + k\,\text{IQR}_w(x_n)\Big)$$

where:
- $M_w(x_n)$ is the rolling median.
- $\text{IQR}_w = Q_{0.75} - Q_{0.25}$ is the interquartile range.
- Window width: $w = 15$.
- Clipping factor: $k = 2.0$.
  
  
## 3. Experimental setup

| Setting | Value |
|---|---|
| Forecasting model | TimesFM 2.5 (200M) |
| Checkpoint | `google/timesfm-2.5-200m-pytorch` |
| Execution environment | Google Colab |
| GPU shown in experiment | NVIDIA T4 |
| Synthetic series length | 512 points |
| Context length | 448 points |
| Forecast horizon | 64 points |
| Noise distribution | Student-$t$, $\nu=2$ |
| Noise scale | $\gamma=2.5$ |
| Random seeds | `42, 101, 202, 303, 404` |
| Summary statistic | Mean MAE ± population standard deviation (`ddof=0`) |

The synthetic signal is:

$$s(t_n) = 10\sin\left(\frac{2\pi t_n}{5}\right) + 0.2t_n + 50, \qquad t_n \in [0,50]$$

The first 448 observations are used as the conditioning context. The final 64 observations are held out as the forecast horizon.

## 4. Quantitative results

The table reports the mean absolute error (MAE) across five random seeds.

| Experimental condition | Horizon MAE | Relative change |
|---|---:|---:|
| **Clean baseline** | `0.0441 ± 0.0000` | — |
| **Heavy-tailed noise ($\nu=2$)** | `3.6588 ± 0.5764` | `+8,196.6%` |
| **Median–IQR filtered** | `3.2709 ± 0.8359` | **10.7% error recovered** |

The recovery percentage is calculated relative to the clean baseline:

$$\text{Recovery}(\%) = \frac{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{filtered}}}{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{clean}}} \times 100 = \frac{3.6588 - 3.2709}{3.6588 - 0.0441} \times 100 \approx 10.7\%$$

The noisy condition has approximately **83 times** the clean-baseline MAE. Median–IQR clipping reduces the reported error, but does not restore clean-baseline performance.

## 5. Diagnostic observation

The experiment indicates that clipping extreme observations alone provides only partial recovery. Smaller fluctuations that remain within the local clipping thresholds are not necessarily removed. These remaining fluctuations may continue to affect the input patches and forecast quality.

The explanation that sub-threshold high-frequency fluctuations cause the remaining amplitude attenuation is a **hypothesis**, not a mechanism established by the current experiment. It can be investigated in future work by comparing additional preprocessing methods and measuring their effect on forecast error.

Potential next steps include:
- Savitzky–Golay smoothing.
- FFT-based low-pass filtering.
- Evaluation on real-world time-series datasets.
- Ablation studies comparing preprocessing methods and parameter settings.

---

## 6. Repository structure

```text
timesfm-robustness-analysis/
├── experiment.ipynb
├── report.pdf
├── timesfm_robustness_plot.png
├── README.md
└── LICENSE
