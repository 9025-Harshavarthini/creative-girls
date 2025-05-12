from google.colab import files
uploaded = files.upload()
# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the dataset
df = pd.read_csv('traffic_accidents.csv')  # Make sure the file is uploaded or in the correct path

# Preview the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Handle missing values
df.dropna(thresh=len(df)*0.5, axis=1, inplace=True)  # Drop columns with more than 50% missing values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric columns with median
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical columns with mode

# Encoding categorical columns (if any)
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Splitting features (X) and target (y)
X = df.drop('Accident_Severity', axis=1)  # Assuming 'Accident_Severity' is the target column
y = df['Accident_Severity']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Severity', 'High Severity'], yticklabels=['Low Severity', 'High Severity'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Feature importance
importances = clf.feature_importances_
feature_names = df.drop('Accident_Severity', axis=1).columns

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

