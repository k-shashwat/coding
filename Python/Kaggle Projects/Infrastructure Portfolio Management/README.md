# Infrastructure Financing Portfolio - Markowitz Optimization

This repository implements the project *Construction of an Infrastructure Financing Portfolio*
developed at VGSoM, IIT Kharagpur. It applies Markowitz Mean-Variance Optimization to simulate and optimize
a 20-year infrastructure investment portfolio.

## Features
- Synthetic data generation for 100 assets
- Log return preprocessing with outlier treatment
- Covariance shrinkage estimation
- Efficient frontier, min variance and max Sharpe optimization
- Stress testing using macro shocks
- Output: portfolio weights, efficient frontier plot, summary CSV

## Run Examples
```
python main.py                 # synthetic data
python main.py --data-path prices.csv --output-dir outputs
```

Outputs are saved in the `outputs/` folder.

