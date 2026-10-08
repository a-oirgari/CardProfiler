import time
import joblib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

from src.config import ROOT, REPORTS_DIR, MODELS_DIR, RANDOM_STATE


MODELS = {
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10, 20]},
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        {"classifier__C": [0.5, 1, 5]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"classifier__max_depth": [5, 10, None], "classifier__min_samples_leaf": [1, 5]},
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"classifier__C": [0.1, 1, 10]},
    ),
}


# def setup_mlflow(experiment_name="classification_clients"):
#     """Suivi dans mlflow.db (SQLite) ; les artefacts vont dans mlruns/."""
#     mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
#     if mlflow.get_experiment_by_name(experiment_name) is None:
#         mlflow.create_experiment(experiment_name, artifact_location=(ROOT / "mlruns").as_uri())
#     mlflow.set_experiment(experiment_name)


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def train_and_log(name, classifier, param_grid, X_train, X_test, y_train, y_test):
    labels = sorted(y_train.unique())
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    # with mlflow.start_run(run_name=name):
    t0 = time.time()
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    duree = time.time() - t0

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

        # ---- MLflow ----
        # mlflow.log_param("model_type", name)
        # mlflow.log_params({k.replace("classifier__", ""): v for k, v in gs.best_params_.items()})
        # mlflow.log_metrics(metrics)
        # mlflow.log_metric("train_time_s", duree)
        # mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3))
        #
        # safe = name.replace(" ", "_")
        # REPORTS_DIR.mkdir(exist_ok=True)
        # fig = plot_confusion_matrix(y_test, y_pred, labels, f"Matrice de confusion - {name}")
        # cm_path = REPORTS_DIR / f"cm_{safe}.png"
        # fig.savefig(cm_path, dpi=120); plt.close(fig)
        # mlflow.log_artifact(str(cm_path))
        #
        # report = classification_report(y_test, y_pred, zero_division=0)
        # rep_path = REPORTS_DIR / f"report_{safe}.txt"
        # rep_path.write_text(report, encoding="utf-8")
        # mlflow.log_artifact(str(rep_path))

    return {"name": name, "pipeline": best, "y_pred": y_pred, "cv_scores": cv_scores,
            "train_time_s": duree, "best_params": gs.best_params_, **metrics}


def save_pipeline(pipeline, filename="best_pipeline.joblib"):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / filename)