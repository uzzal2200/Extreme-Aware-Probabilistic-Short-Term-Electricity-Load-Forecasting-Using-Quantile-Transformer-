# Extreme-Aware Probabilistic Short-Term Electricity Load Forecasting Using Quantile Transformer Networks

<div align="center">

**Hybrid CNN + Transformer + Quantile Output for Extreme-Aware Load Forecasting**

<img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue.svg">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10%2B-red.svg">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26-blue.svg">
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.x-150458.svg">
<img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-1.x-F7931E.svg">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37626.svg">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
<img alt="Status" src="https://img.shields.io/badge/Status-Research-orange.svg">

</div>

---

## Overview
This repository provides a professional, end-to-end implementation of an extreme-aware hybrid CNN-Transformer with quantile outputs for probabilistic short-term electricity load forecasting. The pipeline preserves time-series integrity, captures multi-scale temporal dynamics, and explicitly models extreme conditions to improve reliability under peak demand and rare events. It outputs calibrated prediction intervals (P10-P90) alongside median forecasts to support risk-aware planning and grid operations.

---

## Key Features
- Hybrid multi-scale CNN + Transformer for local and long-range dependencies
- Quantile regression outputs for probabilistic forecasting
- Extreme-temperature awareness for robust peak-demand handling
- Time-series safe preprocessing with no leakage
- Deterministic + probabilistic evaluation with statistical testing
- Modular, reproducible pipeline

---

## Model Performance Highlights

| Metric | Hybrid CNN-Transformer | LSTM Baseline | Improvement |
|--------|------------------------|--------------|-------------|
| MAE | 1,013 MWh | 2,121 MWh | 52.2% |
| RMSE | 1,503 MWh | 2,935 MWh | 48.8% |
| MAPE | 1.87% | 3.73% | 49.7% |
| R2 Score | 0.980 | 0.925 | 5.6% |

**Probabilistic Metrics**
- Prediction Interval Coverage: 76.05%
- CRPS: 660.5
- Extreme Condition Improvement: 70.9% better than LSTM

---

## Architecture
```
Input Sequence (168h window)
       ↓
Multi-Scale CNN Blocks (3h, 24h, 168h kernels)
       ↓
Transformer Encoder (4 heads, 2 layers)
       ↓
Extreme Fusion Layer
       ↓
Quantile Output Heads (Q10, Q50, Q90)
```

---

## Project Structure
```
├── main.py                          # Main experiment pipeline
├── model.py                         # Model architectures
├── train.py                         # Training loop with checkpointing
├── data_loader.py                   # Data loading and preprocessing
├── EDA.py                           # Exploratory data analysis
├── enhanced_evaluation.py           # Probabilistic metrics and tests
├── generate_comparison_plot.py      # Visualization utilities
├── final_notebook_outputs.py        # Notebook export helpers
├── ercot_hybrid_forecast.ipynb      # End-to-end notebook workflow
├── requirements.txt                 # Dependencies
├── Data/
│   └── Final_dataset_ERCOT_v2.csv   # ERCOT load + weather data
├── Notebook Experiment/
│   └── ercot_hybrid_forecast.ipynb  # Experiment notebook copy
├── results/                         # Checkpoints and plots
│   ├── best_hybrid_model.pth
│   ├── hybrid_cnn_transformer_quantile_extreme_ercot.pth
│   └── [evaluation plots]
└── __pycache__/                     # Python bytecode cache
```

---

## Setup
```bash
conda create -n hybridcnn python=3.10.19 -y
conda activate hybridcnn
pip install -r requirements.txt
```

---

## Quick Start
```bash
python main.py
```

Other useful commands:
```bash
python train.py
python EDA.py
python enhanced_evaluation.py
python generate_comparison_plot.py
```

---

## Jupyter Notebook
```bash
jupyter notebook ercot_hybrid_forecast.ipynb
```

---

## Data
**Dataset**: `Data/Final_dataset_ERCOT_v2.csv`

**Target**: ERCOT system demand (MWh)

**Features**: Temperature, humidity, wind speed, precipitation

**Preprocessing**:
- Time-based interpolation for missing values
- Lag and rolling features for temporal context
- Extreme temperature flag (90th percentile)
- Scaling for neural network training

---

## Comprehensive Evaluation Report
```
COMPREHENSIVE EVALUATION REPORT
Hybrid CNN-Transformer with Quantile Output
================================================================================

1. DETERMINISTIC FORECASTING METRICS
--------------------------------------------------------------------------------
Metric                    Hybrid               LSTM                 Improvement
MAE (MWh)                     1013.27            2120.90         52.22%
RMSE (MWh)                    1502.74            2935.46         48.81%
MAPE (%)                        1.8748             3.7295         49.73%
R2 Score                        0.9802             0.9246          6.02%

2. PROBABILISTIC FORECASTING METRICS
--------------------------------------------------------------------------------
Pinball Loss (q10)                             216.2994
Pinball Loss (q50)                             506.6355
Pinball Loss (q90)                             267.8093
Total Pinball Loss                             990.7442
CRPS (Continuous Ranked Probability Score)     660.4961
PICP (Prediction Interval Coverage %)           76.05%
Average Prediction Interval Width              2837.11 MWh

3. PREDICTION INTERVAL ANALYSIS
--------------------------------------------------------------------------------
Coverage (Target: 80%)                           76.05%
Lower Bound Violation Rate                        9.33%
Upper Bound Violation Rate                       14.62%
Target Violation Rate (20%)                      23.95%
Interval Tightness (q90-q10 mean)              2837.11 MWh

4. EXTREME CONDITION HANDLING
--------------------------------------------------------------------------------
Condition                 Samples         Hybrid MAE      LSTM MAE        Improvement
Normal Temperature            10786            949.70       1740.02         45.42%
Extreme Temperature            1286           1546.48       5315.49         70.91%

5. STATISTICAL SIGNIFICANCE TEST
--------------------------------------------------------------------------------
Diebold-Mariano Test:
  Test Statistic: 52.914958
  P-value: 0.000000
  Conclusion: Hybrid model is SIGNIFICANTLY better than LSTM (p < 0.05) ✓

6. DETAILED RESIDUAL STATISTICS
--------------------------------------------------------------------------------
Statistic                      Hybrid               LSTM
Mean Residual                    206.01            1295.54
Std Dev of Residuals            1488.55            2634.11
Min Residual                   -7567.02           -6971.93
Max Residual                   17783.92           12072.94
95th Percentile Error           2822.52            6393.30
```

---

## License
MIT License.

---

## Contact
<div align="center">

**Get in Touch**

Email: [uzzal.220605@s.pust.ac.bd](uzzal.220605@s.pust.ac.bd)

LinkedIn: [https://www.linkedin.com/in/md-uzzal-mia-87a3032a1](https://www.linkedin.com/in/md-uzzal-mia-87a3032a1/)

GitHub: [https://github.com/uzzal2200](https://github.com/uzzal2200)

Institution: Pabna University of Science and Technology

</div>

---

## Acknowledgments
- ERCOT for providing the electricity load dataset
- PyTorch team for the deep learning framework
- Research community for transformer and CNN advancements

---

**Professional Developer Motto**
Build reliable systems, measure everything, and let the data speak.