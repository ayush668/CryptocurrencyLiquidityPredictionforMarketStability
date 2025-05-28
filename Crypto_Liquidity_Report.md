
# Crypto Liquidity Classifier

**Author:** Ayush Mathur  
**Project Type:** Machine Learning with Streamlit Deployment

---

## 1. Objective

The goal of this project is to build a machine learning classifier that predicts the liquidity state of the cryptocurrency market — either **High Liquidity** or **Low Liquidity**. Liquidity is a critical indicator of market health, where low liquidity may signal potential instability or crisis.

---

## 2. Dataset Overview

The dataset includes cryptocurrency market data with the following key features:
- **Price** (USD)
- **24h Volume**
- **Market Capitalization**
- **Returns** over different time periods (1h, 24h, 7d)
- **Log-transformed** financial metrics
- **Temporal features** (day of week, month, weekend indicator)
- **Engineered metrics** such as `volume_to_mktcap`, `price_x_volume`, etc.

---

## 3. Exploratory Data Analysis (EDA)

In the `EDA.ipynb` notebook, basic statistics and visualizations were used to understand:
- Distribution of liquidity labels
- Correlation among features
- Market patterns over time

Key insights included identifying skewness in volume and market cap, prompting the use of log transformations.

---

## 4. Data Preprocessing

Handled in `Data_Preprocessing.ipynb`, preprocessing included:
- Handling missing values
- Standardizing numerical features
- Encoding temporal features like day of week and month

---

## 5. Feature Engineering

In `Feature_Engineering.ipynb`, new features were created:
- **Log transformations** for `price`, `volume`, and `market_cap`
- **Ratios and interactions** like `volume_to_mktcap`, `price_x_volume`
- **Return-based features** and simple market volatility estimates

---

## 6. Model Selection and Training

The notebook `Model_Selection.ipynb` evaluates multiple classifiers, with **XGBoost** emerging as the top performer based on metrics like accuracy, precision, recall, and ROC-AUC.

The final model (`best_xgboost_model.pkl`) was saved using `joblib`.

---

## 7. Model Deployment

Using **Streamlit**, the `app.py` file provides a simple UI for users to:
- Enter market features (price, volume, market cap, etc.)
- Choose date-related inputs (day, month, weekend)
- Run predictions using the trained XGBoost model

Outputs include:
- Label: **High Liquidity** or **Low Liquidity**
- Visual and text feedback

---

## 8. Results

The model shows good performance on test data, identifying liquidity levels effectively using a mix of financial and temporal indicators.

---

## 9. Conclusion & Future Work

The Crypto Liquidity Classifier is a valuable tool for detecting unstable market periods. Future improvements could include:
- Incorporating more real-time or granular data
- Enhancing model interpretability with SHAP values
- Adding trend-based features or LSTM models for temporal patterns

---

**End of Report**
