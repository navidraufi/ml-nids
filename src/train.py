import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib



# Load the cleaned data
df = pd.read_csv('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.columns = df.columns.str.strip()

# Clean it (same as preprocess)
missing = df.isnull().sum()
df = df.drop(columns=missing[missing > len(df) * 0.5].index)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' not in numeric_cols:
    numeric_cols.append('Label')
df = df[numeric_cols]
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# === NEW PART ===

# Separate features (X) and label (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Check what we have
print("Features (X):")
print(f"  Rows: {X.shape[0]}")
print(f"  Columns: {X.shape[1]}")
print(f"  These columns are: {X.columns[:5].tolist()}...")  # show first 5

print("\nLabels (y):")
print(f"  Total: {len(y)}")
print(f"  Normal (0): {(y == 0).sum()}")
print(f"  Attack (1): {(y == 1).sum()}")



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # makes the split reproducible
)

# Check the sizes
print("\nAfter splitting:")
print(f"  Training set: {X_train.shape[0]} rows")
print(f"  Testing set:  {X_test.shape[0]} rows")


# Create the model
model = RandomForestClassifier(
    n_estimators=10,    # number of trees
    random_state=42,    # reproducible results
    n_jobs=-1           # use all CPU cores
)

# Train it
print("\nTraining model...")
model.fit(X_train, y_train)
print("Done! Model has learned from the training data.")




# Make predictions on the test set
print("\nTesting model on unseen data...")
y_pred = model.predict(X_test)

# Compare predictions to real answers
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"That means {accuracy * 100:.2f}% of predictions were correct.")

# Detailed report
print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=['BENIGN (0)', 'DDoS (1)']
))


# Save the model
joblib.dump(model, 'models/random_forest.pkl')
print("\nModel saved to models/random_forest.pkl")