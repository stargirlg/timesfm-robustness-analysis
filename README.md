# Empirical Robustness Analysis of Zero-Shot Time-Series Foundation Models (TimesFM) Under Heavy-Tailed Noise

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: TimesFM 2.5 (200M)](https://img.shields.io/badge/Model-TimesFM%202.5%20(200M)-orange.svg)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An empirical stress-test and failure-mode study evaluating how Google's pre-trained **TimesFM 2.5 (200M)** foundation model degrades under heavy-tailed perturbations ($\text{Student-}t, \nu=2$), alongside an evaluation of **post-hoc rolling Median–IQR clipping** as a lightweight, training-free mitigation strategy.

![Forecast Comparison](timesfm_robustness_plot.png)
*(Representative forecast trajectories from Seed 404 over horizon $H=64$; legend MAE values reflect 5-seed averages[cite: 7].)*

---

## 1. Research Motivation & Mathematical Formulation

Pre-trained time-series foundation models rely on patch-based tokenization and internal instance normalization[cite: 5, 11]:

$$\tilde{x}_n = \frac{x_n - \mu_x}{\sigma_x + \delta}$$[cite: 12]

When real-world telemetry, financial, or demand streams experience heavy-tailed corruption ($x_n = s(t_n) + \epsilon_n$, where $\epsilon_n \stackrel{\text{i.i.d.}}{\sim} t_\nu(0, \gamma^2)$ with $\nu = 2$ and $\gamma = 2.5$)[cite: 11, 12], the underlying $\text{Student-}t$ distribution possesses **infinite theoretical population variance**[cite: 5, 11]. While any finite context window ($L = 448$) exhibits finite empirical variance, extreme outlier draws induce sharp spikes in sample variance $\sigma_x^2$, compressing the normalized representation $\tilde{x}_n$ and attenuating predicted forecast amplitudes over horizon $H = 64$[cite: 5, 11, 12].

To suppress extreme impulses without model retraining or gradient updates[cite: 6, 12], we apply a rolling **Median–Interquartile Range (IQR)** Hampel-style clipping operator over a centered window of width $w = 15$ ($k = 2.0$)[cite: 12]:

$$\hat{x}_n = \text{clip}\Big(x_n,\; M_w(x_n) - k \cdot \text{IQR}_w(x_n),\; M_w(x_n) + k \cdot \text{IQR}_w(x_n)\Big)$$[cite: 6, 12]

---

## 2. Experimental Setup & Quantitative Results

Experiments are executed in zero-shot mode using `google/timesfm-2.5-200m-pytorch` on an NVIDIA T4 GPU across $5$ random seeds (`{42, 101, 202, 303, 404}`)[cite: 12]. We evaluate a $512$-point synthetic series $s(t_n) = 10\sin\left(\frac{2\pi t_n}{5}\right) + 0.2t_n + 50$ for $t_n \in [0, 50]$ ($L = 448$ context, $H = 64$ horizon)[cite: 12], reporting `Mean ± Population Std (ddof=0)`:

| Experimental Condition | Horizon MAE | Relative Error Change |
| :--- | :---: | :---: |
| **Clean Baseline** | `0.0441 ± 0.0000`[cite: 7] | —[cite: 7] |
| **Heavy-Tailed Noise ($\nu=2$)** | `3.6588 ± 0.5764`[cite: 7] | `+8,196.6%` ($\approx 83\times$)[cite: 7] |
| **Median–IQR Filtered (Ours)** | `3.2709 ± 0.8359`[cite: 7] | **`10.7%` error recovered**[cite: 7] |

$$\text{Recovery (\%)} = \frac{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{filtered}}}{\text{MAE}_{\text{noisy}} - \text{MAE}_{\text{clean}}} \times 100 = \frac{3.6588 - 3.2709}{3.6588 - 0.0441} \times 100 \approx 10.7\%$$[cite: 7]

### Key Diagnostic Insight
While spatial outlier winsorization removes extreme tail spikes and recovers **10.7%** of the noise-induced error, dense sub-threshold fluctuations inside $[-2\,\text{IQR}_w, +2\,\text{IQR}_w]$ remain unaltered and continue to damp forecast amplitude. This demonstrates that spatial clipping alone is insufficient and motivates frequency-domain smoothing (e.g., Savitzky–Golay or FFT low-pass filtering) prior to patch tokenization.

---

## 3. Repository Structure

* [`report.pdf`](report.pdf) — Full 4-page technical report detailing mathematical formulation, experimental setup, diagnostic error analysis, and literature references[cite: 9, 11, 12].
* [`experiment.ipynb`](experiment.ipynb) — Reproducible Google Colab notebook containing signal generation, $\text{Student-}t$ noise injection, TimesFM 2.5 zero-shot inference, and 5-seed evaluation[cite: 12].
* [`timesfm_robustness_plot.png`](timesfm_robustness_plot.png) — High-resolution forecast horizon comparison plot (Seed 404 trajectory with 5-seed aggregate MAE legend)[cite: 7].

---

## 4. Quickstart

```bash
pip install git+[https://github.com/google-research/timesfm.git](https://github.com/google-research/timesfm.git) torch numpy scipy matplotlib pandas
