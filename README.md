\# Diabetes Risk Prediction with Explainable AI



An end-to-end machine learning project that predicts diabetes risk on the Pima Indians Diabetes dataset, explains predictions with SHAP, and serves them through an interactive Streamlit app.



\## Features

\- Data cleaning: physiologically impossible zero values treated as missing and imputed with the median

\- Exploratory data analysis (class balance, correlations, distributions)

\- Stratified 80/20 train-test split with feature scaling (fit on training data only to avoid leakage)

\- Comparison of Logistic Regression, Random Forest and XGBoost

\- Hyperparameter tuning with GridSearchCV, optimised for recall

\- SHAP explainability: global importance and per-patient explanations

\- Streamlit web app for interactive risk prediction



\## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn



\## Project Structure

\- `clean.py`: data cleaning

\- `split.py`: train/test split and scaling

\- `models.py`: model comparison

\- `tune.py`: hyperparameter tuning, saves the best model

\- `explain.py`: SHAP plots

\- `app.py`: Streamlit app



\## How to Run

pip install -r requirements.txt

python clean.py

python tune.py

python explain.py

streamlit run app.py



\## Results

(Add your best model's Recall, F1 and ROC-AUC here from the tune.py output)



\## Screenshots

(Add app and SHAP screenshots here)



\## Disclaimer

Educational project only. Not intended for medical use.

