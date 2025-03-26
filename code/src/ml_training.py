import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('transaction_data.csv')

# Define risk level based on the Risk Score
df['Risk Level'] = pd.cut(df['Risk Score'], bins=[0, 40, 70, 100], labels=[0, 1, 2])  # 0 = Low, 1 = Medium, 2 = High

# Encode categorical variables
le_country = LabelEncoder()
le_currency = LabelEncoder()

df['Country'] = le_country.fit_transform(df['Country'])
df['Currency'] = le_currency.fit_transform(df['Currency'])

# Select features and target
features = ['Transaction Amount', 'Country', 'Currency', 'Account Balance', 'Overdraft Flag']
X = df[features]
y = df['Risk Level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, 'risk_model.pkl')
joblib.dump(le_country, 'country_encoder.pkl')
joblib.dump(le_currency, 'currency_encoder.pkl')

print("Model training completed and saved!")
