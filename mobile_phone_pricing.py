# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('mobile_price_data.csv')

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.describe())
print(df.info())

# Feature Selection and Splitting the data into features and target
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy and Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualizing the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Mobile Phone Pricing Prediction')
plt.show()

# Model Prediction Example
# Sample data with proper column names
sample_data = [[510, 1, 2.0, 1, 5, 1, 45, 0.9, 168, 6, 16, 483, 754, 3919, 19, 4, 2, 1, 1, 1]]

try:
    # Convert sample_data to a DataFrame
    sample_data_df = pd.DataFrame(sample_data, columns=X.columns)
    
    # Scale the sample data
    scaled_sample = scaler.transform(sample_data_df)
    
    # Predict the price range for the sample
    predicted_price_range = model.predict(scaled_sample)
    print(f'Predicted price range for the sample: {predicted_price_range[0]}')
except Exception as e:
    print(f"Error with sample data: {e}")
