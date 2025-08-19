from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def train_ml_model(df: pd.DataFrame, features: list, label_column: str = 'ml_label'):
    """
    Train an XGBoost model to predict whether a trade would be profitable.

    Returns:
    - trained model
    - list of features used
    """
    df = df.dropna(subset=features + [label_column])
    X = df[features]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        #use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[ML Evaluation Report]")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, features
