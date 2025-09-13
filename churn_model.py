import os
import joblib
import pandas as pd

# Path to data folder
data_folder = "data"

# Find the first CSV file inside the data folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found in the 'data' folder!")

# Pick the first CSV file
csv_path = os.path.join(data_folder, csv_files[0])

print(f"Using dataset: {csv_path}")

# Load the dataset
df = pd.read_csv(csv_path)

# Show first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())
# Dataset info
print("\n--- Dataset Info ---")
print(df.info())

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Target distribution
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())

# --- Fix TotalCharges column ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("\n--- Missing after conversion ---")
print(df['TotalCharges'].isnull().sum())

df = df.dropna(subset=['TotalCharges'])

print("\n--- Data Types After Fix ---")
print(df.dtypes)

# --- Step 3: Preprocessing & Encoding ---

# Drop customerID since it's an identifier and not useful for prediction
df = df.drop('customerID', axis=1)

# Encode the 'Churn' target variable (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Get a list of all categorical columns except 'Churn'
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

# Perform One-Hot Encoding on categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Print the new info and first 5 rows to confirm the changes
print("\n--- Dataset Info After Encoding ---")
print(df.info())

print("\n--- First 5 Rows After Encoding ---")
print(df.head())

# --- Step 4: Model Building & Evaluation ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# --- Save the trained model to a file ---
joblib.dump(model, 'logistic_regression_model.pkl')
print("\nModel saved as logistic_regression_model.pkl")