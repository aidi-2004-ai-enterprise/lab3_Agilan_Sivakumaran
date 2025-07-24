# Training an XGBoost model on the penguins dataset
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import json
import os


# Load the seaborn penguins dataset
def load_data():
    
    df = sns.load_dataset("penguins")
    df = df.dropna()
    return df

def preprocess(df):
    # One-hot encode island and sex columns
    df = pd.get_dummies(df, columns=["island", "sex"])

    # Label encode the species column
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])

    # Separate X and y
    X = df.drop(["species", "species_encoded"], axis=1)
    y = df["species_encoded"]

    return X, y, list(le.classes_)


# Train the XGBoost model
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X, y, split_name):
    preds = model.predict(X)
    f1 = f1_score(y, preds, average='weighted')
    print(f"{split_name} F1 Score:", f1)
    print(classification_report(y, preds))

# Save the model to a JSON file
def save_model(model, classes, filepath):
    booster = model.get_booster()
    model_json = booster.save_raw("json").decode("utf-8")
    data_to_save = {
        "model": model_json,
        "classes": classes
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y, classes = preprocess(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating on training data...")
    evaluate_model(model, X_train, y_train, "Train")

    print("Evaluating on test data...")
    evaluate_model(model, X_test, y_test, "Test")

    print("Saving model...")
    save_model(model, classes, "app/data/model.json")

    print("Done.")

if __name__ == "__main__":
    main()
