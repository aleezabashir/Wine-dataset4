import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load Wine Quality dataset (red wine)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Convert quality to classes
def quality_class(q):
    if q <= 5:
        return "Low"
    elif q <= 7:
        return "Medium"
    else:
        return "High"

df['quality_class'] = df['quality'].apply(quality_class)
X = df.drop(['quality', 'quality_class'], axis=1)
y = df['quality_class']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(rf_model, "wine_quality_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model and LabelEncoder saved as .pkl files")
