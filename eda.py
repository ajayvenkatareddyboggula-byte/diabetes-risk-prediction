import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())
print("\nOutcome counts:\n", df["Outcome"].value_counts())

sns.countplot(x="Outcome", data=df)
plt.title("Diabetes vs Non-Diabetes")
plt.show()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(12, 10))
plt.show()