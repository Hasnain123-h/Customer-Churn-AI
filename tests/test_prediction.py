from pathlib import Path
import sys
import pandas as pd
import joblib



BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_tuned.pkl"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test_processed.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Model file not found"


def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_prediction_output():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_prediction_values():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    predictions = model.predict(X_test)
    assert set(predictions).issubset({0, 1})


def test_prediction_probability():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]
    assert len(probabilities) == len(X_test)

    assert all(
        0 <= probability <= 1
        for probability in probabilities
    )