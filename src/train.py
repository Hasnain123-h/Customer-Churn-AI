from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train_processed.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_tuned.pkl"
def load_training_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    return X_train, y_train

def train_model(X_train, y_train):

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    }

    grid_search = GridSearchCV(
        estimator=logistic_model,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search.best_estimator_


def save_model(model):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":

    X_train, y_train = load_training_data()
    model = train_model(X_train, y_train)
    save_model(model)

    print("successfully")

