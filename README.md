# timesfm-robustness-analysis

# Robustness Analysis of Zero-Shot Time-Series Foundation Models (TimesFM) Under Heavy-Tailed Noise

An empirical investigation into how Google's pre-trained **TimesFM 2.5** foundation model degrades when exposed to infinite-variance, heavy-tailed noise ($\text{Student-}t, \nu=2$), and an evaluation of **post-hoc Median-IQR clipping** as a lightweight mitigation strategy.

![Forecast Comparison](timesfm_robustness_plot.png)

## 1. Research Motivation & Mathematical Formulation
Time-series foundation models rely on patch-based tokenization and internal z-score normalization:

$$\tilde{x}_t = \frac{x_t - \mu}{\sigma}$$

When real-world telemetry, financial, or demand signals suffer from heavy-tailed outliers ($x_t = s_t + \epsilon_t$, where $\epsilon_t \sim t_\nu(0, \sigma^2)$ with $\nu \le 2$), sample variance $\sigma^2 \to \infty$. This distorts internal normalization statistics and attenuates forecast amplitude.

To test post-hoc mitigation without model fine-tuning, we apply a rolling **Median–Interquartile Range (IQR)** clipping operator prior to tokenization:

$$\hat{x}_t = \text{clip}\Big(x_t,\; M_w(x_t) - k \cdot \text{IQR}_w(x_t),\; M_w(x_t) + k \cdot \text{IQR}_w(x_t)\Big)$$

where $M_w$ is the rolling median over window $w=15$ and $k=2.0$.

## 2. Quantitative Results (5 Random Seeds)

| Experimental Condition | Mean Absolute Error (MAE) | Relative Error Change |
| :--- | :---: | :---: |
| **Clean Baseline** | `0.0441 ± 0.0000` | — |
| **Heavy-Tailed Noise ($\nu=2$)** | `3.6588 ± 0.5764` | `+8,196.6%` |
| **Post-Hoc Median-IQR Filter** | `3.2709 ± 0.8359` | **`10.7%` error recovered** |

### Key Analytical Finding
While rolling Median-IQR clipping eliminates extreme infinite-variance spikes and recovers **10.7%** of forecasting accuracy, dense sub-threshold heavy-tailed fluctuations still attenuate amplitude predictions (as visible in the horizon trajectory above). This demonstrates that outlier winsorization alone is insufficient and motivates spectral/low-pass filtering prior to patch embedding.

## 3. Repository Structure
* `experiment.ipynb` — Complete reproducible Google Colab notebook (data generation, noise injection, TimesFM 2.5 inference, and multi-seed ablation).
* `report.pdf` — Technical report detailing problem formulation, experimental methodology, and failure-mode analysis.
* `timesfm_robustness_plot.png` — High-resolution visualization of forecast trajectories across conditions.

## 4. Quickstart
```bash
pip install git+[https://github.com/google-research/timesfm.git](https://github.com/google-research/timesfm.git) torch numpy scipy matplotlib pandas
