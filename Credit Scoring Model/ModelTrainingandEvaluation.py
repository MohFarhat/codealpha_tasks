AI.txt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load preprocessed data
X_train_scaled = pd.read_csv('/mnt/data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('/mnt/data/X_test_scaled.csv')
y_train = pd.read_csv('/mnt/data/y_train.csv')
y_test = pd.read_csv('/mnt/data/y_test.csv')

# Initialize models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Train models
log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

# Make predictions
log_reg_pred = log_reg.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)
gb_pred = gb.predict(X_test_scaled)

# Evaluate models
models = {'Logistic Regression': log_reg, 'Random Forest': rf, 'Gradient Boosting': gb}
predictions = {'Logistic Regression': log_reg_pred, 'Random Forest': rf_pred, 'Gradient Boosting': gb_pred}

results = {}

for model_name, pred in predictions.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

results_df = pd.DataFrame(results).T
results_df.to_csv('/mnt/data/model_evaluation_results.csv')
results_df
