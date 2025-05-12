import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('traffic_accidents.csv')

# Display basic info
print("Initial dataset shape:", df.shape)
print(df.head())

# --- Step 1: Handle missing values ---
# Drop columns with too many missing values or fill with median/mode
df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)  # drop columns with >50% missing
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)  # for categorical columns

# --- Step 2: Encode categorical variables ---
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Step 3: Feature scaling ---
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# --- Step 4: Train-test split ---
# Assume 'Accident_Severity' is the target
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Preprocessing complete. Training set size:", X_train.shape)
