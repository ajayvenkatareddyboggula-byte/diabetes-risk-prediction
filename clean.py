import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("Zeros BEFORE cleaning:")
print((df[cols] == 0).sum())

df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

print("\nZeros AFTER cleaning:")
print((df[cols] == 0).sum())

df.to_csv("diabetes_clean.csv", index=False)
print("\nSaved diabetes_clean.csv")