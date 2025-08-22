import argparse
import warnings
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load trade data from CSV and drop rows missing required columns.
    """
    df = pd.read_csv(path)
    required = ['Entry', 'Entry Volume', 'P/L']
    df = df.dropna(subset=required)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build target column and select pre-entry features.
    Splits data chronologically into train/test subsets.
    """
    # Binary target: 1 if profit, otherwise 0
    df['Target'] = (df['P/L'] > 0).astype(int)

    # Dynamically select features that match the pattern of technical indicators (SMA, EMA, RSI, etc.)
    feature_cols = [col for col in df.columns if any(indicator in col for indicator in ['SMA', 'EMA', 'RSI'])]
    feature_cols.extend(['Entry', 'Entry Volume'])

    # Optional: Include 'Holding Period' if it exists in the dataset
    if 'Holding Period' in df.columns:
        feature_cols.append('Holding Period')

    X = df[feature_cols]
    y = df['Target']

    # Chronological split: first 80% train, last 20% test
    split_idx = int(0.8 * len(df))
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    print(f"[INFO] Dataset loaded: {len(df)} rows")
    print(f"[INFO] Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    print(f"[INFO] Target distribution (train):\n{y_train.value_counts(normalize=True)}\n")

    return (X_train, X_test, y_train, y_test)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    """
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model


def train_support_vector_classifier(X_train, y_train) -> SVC:
    """
    Train a Support Vector Classifier (SVC).
    """
    print("[INFO] Training Support Vector Classifier...")
    model = SVC(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting model.
    """
    print("[INFO] Training Gradient Boosting...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest with balanced class weights.
    """
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier handling class imbalance.
    """
    print("[INFO] Training XGBoost...")
    # Compute scale_pos_weight = (#negatives)/(#positives)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name: str) -> None:
    """
    Evaluate model performance on test set and print metrics.
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n[RESULTS] {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))


def save_models(models, out_dir: Path) -> None:
    """
    Save trained models to disk using joblib.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        model_path = out_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"[INFO] {model_name} model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models for trade classification")
    parser.add_argument(
        "--input", "-i",
        default="result.csv",
        help="Path to CSV file with trade data (default: result.csv)"
    )
    parser.add_argument(
        "--save-models", "-s",
        action="store_true",
        help="Save trained models to disk"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.input)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(df)

    # Train models
    rf_model = train_random_forest(X_train, y_train)
    evaluate(rf_model, X_test, y_test, "Random Forest")

    xgb_model = train_xgboost(X_train, y_train)
    evaluate(xgb_model, X_test, y_test, "XGBoost")

    logreg_model = train_logistic_regression(X_train, y_train)
    evaluate(logreg_model, X_test, y_test, "Logistic Regression")

    svc_model = train_support_vector_classifier(X_train, y_train)
    evaluate(svc_model, X_test, y_test, "Support Vector Classifier")

    gb_model = train_gradient_boosting(X_train, y_train)
    evaluate(gb_model, X_test, y_test, "Gradient Boosting")

    # Optional: save models
    if args.save_models:
        save_models({
            "rf": rf_model,
            "xgb": xgb_model,
            "logreg": logreg_model,
            "svc": svc_model,
            "gb": gb_model
        }, out_dir=Path("models"))


if __name__ == "__main__":
    main()
