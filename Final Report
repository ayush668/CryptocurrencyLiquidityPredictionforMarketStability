# Cryptocurrency Liquidity Prediction for Market Stability
## Problem Statement
In this data science project, you will build a machine learning system that predicts the liquidity levels of cryptocurrency markets based on various market factors. This project is highly useful for traders and exchange platforms to manage risks effectively and maintain market stability. Using data such as trading volume, transaction patterns, exchange listings, and social media activity, you will develop a model that can detect liquidity crises early and help prevent large price fluctuations.
## Solution Proposed
Now the question is, how can we dynamically predict the liquidity level of the market? One approach is to use machine learning, where we analyze various market factors such as trading volume and transaction patterns to identify liquidity patterns. By leveraging historical market data and domain knowledge, we can build a model that predicts liquidity levels in real-time to help manage risks effectively.


## Dataset Used

[Dataset Link](https://drive.google.com/drive/folders/10BRgPip2Zj_56is3DilJCowjfyT6E9AM)

## Tech Stack Used
1. Python
2. Machine learning algorithms
3.Streamlit

## Infrastructure required 
1. Github Actions

# Run the application server

```

Streamlit run app.py

```
```bash

http://localhost:8501

```
 Data Collection Architecture -

![WhatsApp Image 2022-09-22 at 15 29 10](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg)

## Deployment Architecture -

![wdtm1p8j](https://github.com/user-attachments/assets/4b0a6e5e-016c-49f7-953e-b2d5fb393e5c)




## Project Architecture -

![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)

## Models Used

* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Catboost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)

  From these above models after hyperparameter optimization we selected these two models which were Random Forest and LCatboost for classification and used the following in Pipeline.

* GridSearchCV is used for Hyperparameter Optimization in the pipeline.
  ## `EDA` and `Model Selection and training` are the main package folder which contain

**Components** : Contains all components of Machine Learning Project

- Data Ingestion
- Data Validation
- Data Transformation
- Data Clustering
- Model Trainer
- Model Evaluation
- Model Pusher

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.
## Conclusion
-
1. Best model: Random Forest with tuned parameters
2. Accuracy achieved: 0.6100 (original baseline was 0.6200)
3. Key improvements implemented:
   - Feature engineering including log transformations and interaction features
   - Stratified sampling to handle class imbalance
   - Hyperparameter optimization using GridSearchCV
   - Custom scoring function weighted toward minority classes
   - Ensemble modeling combining multiple classifiers
4.The output is shown in this way {0:'LOW',1:'Medium',2:'High'}
- This Project can be used in real-life by Users.

Next steps:
   - Collect more data, especially for minority classes
   - Add more cryptocurrency-specific features like market sentiment
   - Consider time-series cross-validation for temporal data
   - Monitor model performance regularly and retrain as needed

