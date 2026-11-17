# Empirical Robustness Analysis of Zero-Shot Time-Series Foundation Models (TimesFM) Under Heavy-Tailed Noise

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: TimesFM 2.5 (200M)](https://img.shields.io/badge/Model-TimesFM%202.5%20(200M)-orange.svg)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An empirical stress test of Google's **TimesFM 2.5 (200M)** under heavy-tailed perturbations drawn from a Student-$t$ distribution with $\nu=2$[cite: 18]. The project also evaluates **post-hoc rolling Median–Interquartile Range (IQR) clipping** as a lightweight, training-free mitigation method[cite: 18].

> **Scope:** The reported experiments use a synthetic time series and five random seeds[cite: 18]. The results should be interpreted as findings from this controlled experiment, not as evidence of performance on all real-world time series[cite: 18].

## Forecast comparison

![Forecast comparison](timesfm_robustness_plot.png)[cite: 18]

*Representative forecast trajectories from Seed 404 over a forecast horizon of $H=64$[cite: 18]. The quantitative table below reports the five-seed aggregate results[cite: 18].*

---

## 1. Research motivation

Time-series foundation models use input processing and patch-based representations to prepare sequences for forecasting[cite: 18]. A common normalization form is:

$$\widetilde{x}_n = \frac{x_n-\mu_x}{\sigma_x+\delta}$$
[cite: 18]

where $\mu_x$ and $\sigma_x$ are the input-window mean and standard deviation, and $\delta$ is a small numerical-stability constant[cite: 18].

In this experiment, the observed series is modelled as:

$$x_n = s(t_n) + \epsilon_n, \qquad \epsilon_n \stackrel{\text{i.i.d.}}{\sim} t_\nu(0,\gamma^2)$$
[cite: 18]

with $\nu=2$ and $\gamma=2.5$[cite: 18]. The Student-$t$ distribution has infinite theoretical population variance when $\nu\leq 2$[cite: 18]. A finite sample still has a finite empirical variance, but large outlier observations can sharply increase the sample variance and affect normalization[cite: 18].

## 2. Median–IQR mitigation

To reduce extreme observations without fine-tuning the forecasting model, the experiment applies a rolling Median–IQR clipping operation before inference:

$$\widehat{x}_n = \operatorname{clip}\Big(x_n,\, M_w(x_n) - k\,\operatorname{IQR}_w(x_n),\, M_w(x_n) + k\,\operatorname{IQR}_w(x_n)\Big)$$
[cite: 18]

where:
- $M_w(x_n)$ is the rolling median[cite: 18].
- $\operatorname{IQR}_w = Q_{0.75} - Q_{0.25}$ is the interquartile range[cite: 18].
- Window width: $w = 15$[cite: 18].
- Clipping factor: $k = 2.0$[cite: 18].

## 3. Experimental setup

| Setting | Value |
|---|---|
| Forecasting model | TimesFM 2.5 (200M)[cite: 18] |
| Checkpoint | `google/timesfm-2.5-200m-pytorch`[cite: 18] |
| Execution environment | Google Colab[cite: 18] |
| GPU shown in experiment | NVIDIA T4[cite: 18] |
| Synthetic series length | 512 points[cite: 18] |
| Context length | 448 points[cite: 18] |
| Forecast horizon | 64 points[cite: 18] |
| Noise distribution | Student-$t$, $\nu=2$[cite: 18] |
| Noise scale | $\gamma=2.5$[cite: 18] |
| Random seeds | `42, 101, 202, 303, 404`[cite: 18] |
| Summary statistic | Mean MAE ± population standard deviation (`ddof=0`)[cite: 18] |

The synthetic signal is:

$$s(t_n) = 10\sin\left(\frac{2\pi t_n}{5}\right) + 0.2t_n + 50, \qquad t_n \in [0,50]$$
[cite: 18]

The first 448 observations are used as the conditioning context[cite: 18]. The final 64 observations are held out as the forecast horizon[cite: 18].

## 4. Quantitative results

The table reports the mean absolute error (MAE) across five random seeds[cite: 18].

| Experimental condition | Horizon MAE | Relative change |
|---|---:|---:|
| **Clean baseline** | `0.0441 ± 0.0000`[cite: 18] | —[cite: 18] |
| **Heavy-tailed noise ($\nu=2$)** | `3.6588 ± 0.5764`[cite: 18] | `+8,196.6%`[cite: 18] |
| **Median–IQR filtered** | `3.2709 ± 0.8359`[cite: 18] | **10.7% error recovered**[cite: 18] |

The recovery percentage is calculated relative to the clean baseline:

$$\text{Recovery}(\%) = \frac{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{filtered}}}{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{clean}}} \times 100 = \frac{3.6588 - 3.2709}{3.6588 - 0.0441} \times 100 \approx 10.7\%$$
[cite: 18]

The noisy condition has approximately **83 times** the clean-baseline MAE[cite: 18]. Median–IQR clipping reduces the reported error, but does not restore clean-baseline performance[cite: 18].

## 5. Diagnostic observation

The experiment indicates that clipping extreme observations alone provides only partial recovery[cite: 18]. Smaller fluctuations that remain within the local clipping thresholds are not necessarily removed[cite: 18]. These remaining fluctuations may continue to affect the input patches and forecast quality[cite: 18].

The explanation that sub-threshold high-frequency fluctuations cause the remaining amplitude attenuation is a **hypothesis**, not a mechanism established by the current experiment[cite: 18]. It can be investigated in future work by comparing additional preprocessing methods and measuring their effect on forecast error[cite: 18].

Potential next steps include:
- Savitzky–Golay smoothing[cite: 18].
- FFT-based low-pass filtering[cite: 18].
- Evaluation on real-world time-series datasets[cite: 18].
- Ablation studies comparing preprocessing methods and parameter settings[cite: 18].

---

## 6. Repository structure

```text
timesfm-robustness-analysis/
├── experiment.ipynb
├── report.pdf
├── timesfm_robustness_plot.png
├── README.md
└── LICENSE
```[cite: 18]

- **`experiment.ipynb`** — Google Colab notebook containing signal generation, Student-$t$ noise injection, TimesFM inference, and the five-seed evaluation[cite: 18].
- **`report.pdf`** — Technical report describing the formulation, experimental setup, results, and diagnostic analysis[cite: 18].
- **`timesfm_robustness_plot.png`** — Forecast comparison plot[cite: 18].
- **`LICENSE`** — Project license[cite: 18].

## 7. Quickstart

### Install TimesFM

```bash
pip install git+[https://github.com/google-research/timesfm.git](https://github.com/google-research/timesfm.git)
```[cite: 18]

### Install experiment dependencies

```bash
pip install torch numpy scipy matplotlib pandas
```[cite: 18]

For reproducibility, use the notebook's recorded package versions and runtime configuration[cite: 18]. GPU availability and memory limits depend on the Colab runtime assigned to your session[cite: 18].

### Run the experiment

1. Open `experiment.ipynb` in Google Colab[cite: 18].
2. Select **Runtime → Change runtime type → T4 GPU** (if available)[cite: 18].
3. Run the notebook cells in order[cite: 18].
4. Review the per-seed results and generated forecast plot[cite: 18].

## 8. References

- Das, A., Kong, W., Sen, R., and Zhou, Y. “A decoder-only foundation model for time-series forecasting.” *Proceedings of the International Conference on Machine Learning (ICML)*, 2024[cite: 18].
- Hampel, F. R. “The influence curve and its role in robust estimation.” *Journal of the American Statistical Association*, 69(346), 383–393, 1974[cite: 18].
- [TimesFM 2.5 model checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch)[cite: 18]
- [TimesFM GitHub repository](https://github.com/google-research/timesfm)[cite: 18]

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE)[cite: 18].
