import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

print("=== Membaca Dataset ===")
df = pd.read_csv("LengthOfStay.csv")
print(df.head())

df = df.copy()
df = df.replace('?', np.nan)
df = df.dropna(subset=['lengthofstay'])
df['lengthofstay'] = df['lengthofstay'].astype(float)

df['rcount'] = df['rcount'].replace('5+', 5).astype(int)

df['gender'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)

features = ['rcount', 'gender', 'asthma', 'pneum', 'depress']
target = 'lengthofstay'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/linear_regression_los.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model berhasil disimpan di: models/linear_regression_los.pkl")

print("\n=== Korelasi Antar Variabel ===")
print(df[features + [target]].corr())
