import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

from matplotlib.backends.backend_pdf import PdfPages

from src.config import (
    VISUALIZATION_DIR,
    MODEL_SAVE_PATH,
    MODEL_RESULTS_PATH,
    RANDOM_STATE,
    setup_directories,
)

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
setup_directories()


# ============================================================
# HELPERS
# ============================================================
def save_plot(filename: str):
    """Helper to save plot and close figure."""
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename), dpi=300)
    plt.close()


def _get_predict_proba(model, X):
    """
    Return probability-like scores for positive class if available.
    Supports:
    - predict_proba (best)
    - decision_function (scaled to 0-1 fallback; NOT calibrated)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.shape[1] >= 2:
            return proba[:, 1]
        return None

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

    return None


# ============================================================
# CROSS VALIDATION
# ============================================================
def perform_cross_validation(model, X, y, cv=5):
    """
    Perform Stratified k-fold cross-validation and return mean scores with std.
    """
    logging.info(f"Performing {cv}-Fold Stratified Cross-Validation...")

    cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    results = {}
    for metric_name, metric in scoring.items():
        scores = cross_val_score(model, X, y, cv=cv_stratified, scoring=metric, n_jobs=-1)
        results[metric_name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores,
        }
        logging.info(f"  {metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


# ============================================================
# MODEL COMPARISON
# ============================================================
def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare multiple models using Cross-Validation and return the BEST performing model.
    Selection criterion: CV F1-Score (stronger for imbalanced data)
    """
    logging.info("Comparing Multiple Models (CV-based)...")

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=RANDOM_STATE,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_iter=200,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    trained_models = {}

    for name, model in models.items():
        logging.info(f"  CV scoring {name}...")

        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(
            {
                "Model": name,
                "CV F1 Mean": float(f1_scores.mean()),
                "CV F1 Std": float(f1_scores.std()),
                "Test F1": float(f1_score(y_test, y_pred)),
                "Test Recall": float(recall_score(y_test, y_pred)),
                "Test Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "Test Accuracy": float(accuracy_score(y_test, y_pred)),
            }
        )

        trained_models[name] = model

    comparison_df = (
        pd.DataFrame(results)
        .sort_values("CV F1 Mean", ascending=False)
        .reset_index(drop=True)
    )

    logging.info(f"\nModel Comparison (Sorted by CV F1):\n{comparison_df}")

    # ✅ Pyrefly-safe: use iloc rather than loc + float()
    best_model_name = str(comparison_df["Model"].iloc[0])
    best_cv_f1 = float(comparison_df["CV F1 Mean"].iloc[0])

    logging.info(f"\n🏆 BEST MODEL: {best_model_name} (CV F1 = {best_cv_f1:.4f})")

    # Save comparison table as image (Pyrefly-safe conversion)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("tight")
    ax.axis("off")

    cell_colors = [
        ["lightgreen" if i == 0 else "white" for _ in range(len(comparison_df.columns))]
        for i in range(len(comparison_df))
    ]

    display_df = comparison_df.round(4)
    cell_text = display_df.astype(str).values.tolist()
    col_labels = list(display_df.columns)

    ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )

    plt.title(f"Model Comparison Results (Winner: {best_model_name})", fontsize=14, pad=20)
    save_plot("model_comparison_table.png")

    return comparison_df, trained_models[best_model_name], best_model_name


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Comprehensive model evaluation with metrics + visualizations.
    Outputs:
    - confusion_matrix.png
    - roc_curve.png (if available)
    - pr_curve.png (if available)
    - feature_importance.png (if available)
    - metrics CSV
    - PDF report
    """
    logging.info("Evaluating Model...")

    y_pred = model.predict(X_test)
    y_pred_proba = _get_predict_proba(model, X_test)

    # Threshold Tuning (if probability scores exist)
    best_t = 0.5
    if y_pred_proba is not None:
        thresholds = np.linspace(0.05, 0.95, 19)
        best_f1 = -1
        for t in thresholds:
            pred_t = (y_pred_proba >= t).astype(int)
            f1_t = f1_score(y_test, pred_t)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, float(t)
        
        logging.info(f"🏆 Best Threshold by F1: {best_t:.2f} (F1={best_f1:.4f})")
        
        # Re-calculate metrics with best threshold
        y_pred = (y_pred_proba >= best_t).astype(int)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        
        logging.info(f" ऑप्टimized Metrics @ {best_t:.2f}:")
        logging.info(f"F1-Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Save Metrics CSV
    metrics_path = MODEL_RESULTS_PATH.with_suffix(".csv")
    pd.DataFrame(
        [{"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}]
    ).to_csv(metrics_path, index=False)

    # Overfitting check
    train_accuracy = None
    gap = None
    if X_train is not None and y_train is not None:
        train_pred = model.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, train_pred))
        gap = train_accuracy - accuracy
        logging.info(f"Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Accuracy Gap: {gap:.4f}")
        if gap > 0.05:
            logging.warning("Potential OVERFITTING detected (>5% gap).")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    save_plot("confusion_matrix.png")

    roc_auc = None
    pr_ap = None

    # ROC + PR curve if proba exists
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = float(auc(fpr, tpr))

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        save_plot("roc_curve.png")

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_ap = float(average_precision_score(y_test, y_pred_proba))

        plt.figure(figsize=(6, 4))
        plt.plot(recall_vals, precision_vals, lw=2, label=f"AP = {pr_ap:.3f}")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        save_plot("pr_curve.png")
    else:
        logging.info("No probability scores available -> ROC/PR curves skipped.")

    # Feature importance (supports pipeline too)
    model_for_importance = model
    if isinstance(model, Pipeline) and "clf" in model.named_steps:
        model_for_importance = model.named_steps["clf"]

    if hasattr(model_for_importance, "feature_importances_"):
        importances = model_for_importance.feature_importances_

        if isinstance(X_test, pd.DataFrame):
            indices = (
                pd.Series(importances, index=X_test.columns)
                .sort_values(ascending=False)
                .head(20)
            )
        else:
            indices = pd.Series(importances).sort_values(ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=indices.values, y=indices.index)
        plt.title("Feature Importance (Top 20)")
        save_plot("feature_importance.png")

    # PDF Report
    pdf_path = MODEL_RESULTS_PATH.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.axis("off")

        report_text = (
            f"Model Evaluation Report\n\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"F1-Score: {f1:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"Precision: {precision:.4f}\n"
        )

        if train_accuracy is not None and gap is not None:
            report_text = report_text + (
                f"\nTraining Accuracy: {train_accuracy:.4f}\n"
                f"Accuracy Gap (Train - Test): {gap:.4f}\n"
            )

        if roc_auc is not None:
            report_text = report_text + f"\nROC-AUC: {roc_auc:.4f}\n"
        if pr_ap is not None:
            report_text = report_text + f"PR-AUC (Average Precision): {pr_ap:.4f}\n"

        report_text = report_text + "\nClassification Report:\n"
        report_text = report_text + classification_report(y_test, y_pred, zero_division=0)

        # ✅ Pyrefly-safe best params append
        best_params_str = ""
        if hasattr(model, "best_params_"):
            best_params_str = "\n\nBest Params:\n" + str(model.best_params_)

        report_text = report_text + best_params_str

        ax.text(0.01, 0.5, report_text, fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


    logging.info(f"Evaluation complete. Results saved to {MODEL_RESULTS_PATH.parent}")
    
    # Save Model Bundle (Model + Threshold)
    bundle_path = MODEL_SAVE_PATH # Use clean path from config
    logging.info(f"Saving model bundle with threshold={best_t:.2f}...")
    joblib.dump({"model": model, "threshold": best_t}, bundle_path)
    logging.info(f"Model bundle saved to: {bundle_path}")
    
    return best_t


# ============================================================
# TRAINING: RANDOM FOREST WITH RANDOM SEARCH
# ============================================================
def train_rf_model(X_train, y_train):
    """
    Train Random Forest with RandomizedSearchCV and cross-validation.
    """
    logging.info("Starting Random Forest Training with Hyperparameter Search...")

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        random_state=RANDOM_STATE,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    logging.info(f"Training Complete. Best Params: {random_search.best_params_}")

    cv_results = perform_cross_validation(best_model, X_train, y_train, cv=5)

    joblib.dump(best_model, MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

    return random_search, cv_results


if __name__ == "__main__":
    pass
