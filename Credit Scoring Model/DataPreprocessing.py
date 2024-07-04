import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('/mnt/data/credit_scoring_data.csv')

# Split the dataset into features and target variable
X = data.drop(columns=['Creditworthy'])
y = data['Creditworthy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

X_train_scaled.to_csv('/mnt/data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('/mnt/data/X_test_scaled.csv', index=False)
y_train.to_csv('/mnt/data/y_train.csv', index=False)
y_test.to_csv('/mnt/data/y_test.csv', index=False)
