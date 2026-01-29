# Diabetes-Prediction-using-Classical-ML-Models

OVERVIEW

This project applies supervised learning classifiers to a real-world health problem using the Pima Indians Diabetes Dataset. It focuses on model building, evaluation and interpretation to compare Logistic Regression and Linear Discriminant Analysis (LDA) for predicting the onset of diabetes.

Problem Statement

Early detection of diabetes is crucial for patient health management. Medical diagnostic data contains complex patterns that can indicate a patient's risk. This project aims to develop and compare multiple machine learning models, Logistic Regression, Random Forest and XGBoost, to accurately predict whether a patient has diabetes based on this diagnostic data.

Methodology and Milestones

- Data Exploration and Preprocessing: Load the dataset, perform EDA to understand feature distributions and handle erroneous zero values by replacing them with a suitable statistic.
- Model Training: Split the data into training and testing sets. Scale features before training the models.
- Performance Evaluation: Plot the ROC curve for all models on a single graph to visually compare their diagnostic ability.
- Feature Importance: Interpreted the coefficient from the models to identify the most significant predictors (Glucose and BMI).

Dataset Information:

| Feature                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| Pregnancies              | Number of times pregnant                        |
| Glucose                  | Plasma glucose concentration (2-hour oral test) |
| BloodPressure            | Diastolic blood pressure (mm Hg)                |
| SkinThickness            | Triceps skin fold thickness (mm)                |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                  |
| BMI                      | Body mass index (weight in kg / height in m²)   |
| DiabetesPedigreeFunction | Genetic influence on diabetes                   |
| Age                      | Age of the patient                              |
| Outcome                  | Class variable (0 = No Diabetes, 1 = Diabetes)  |

Project Structure

Diabetes-Prediction-using-Classical-ML-Models/
│
├── notebooks/
│ └── diabetes_prediction.ipynb
│
├── data/
│ └── diabetes.csv
│
├── plots/
│ ├── Logistic_Regression_Coefficients.png
│ ├── Logistic_Regression_confusion_matrix.png
│ ├── ROC_Curve_Comparison.png
│ ├── Random_Forest_Confusion_matrix.png
│ ├── Random_Forest_Feature_Importance.png
│ ├── XGBoost_Confusion_Matrix.png
│ ├── XGBoost_Feature_Importance.png
│ ├── correlation_heatmap.png
│ ├── histogram_for_numerical_features.png
│
├── README.md
└── requirements.txt
Modeling and Evaluation

| Model               | Accuracy | AUC  |
| ------------------- | -------- | ---- |
| Logistic Regression | 0.71     | 0.82 |
| Random Forest       | 0.73     | 0.81 |
| XGBoost             | 0.73     | 0.82 |

Key Findings

1) Glucose and BMI are the most important predictors for diabetes
2) Logistic Regression and XGBoost achieved the highest AUC of 0.82
3) Ensemble methods provide slightly better balance in precision and recall

Tech Stack

Python 3.12 | Pandas | NumPy | scikit-learn | XGBoost | Matplotlib | Seaborn

Future Improvements

1) Implement cross-validation and hyperparameter tuning for better accuracy
2) Deploy a web-based interactive UI for real-time diabetes prediction
3) Use deep learning models for comparison
