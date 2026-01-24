import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
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
)

from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------------------------
# Add project root to sys.path
# ------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.model_defs import LogisticRegressionTorch


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
config.ensure_dirs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# DATA LOADING
# ============================================================
def load_feature_engineered_data():
    """
    Load resampled and feature-engineered data (includes SMOTE).
    Expects:
    - config.X_RESAMPLED_FILE
    - config.Y_RESAMPLED_FILE
    """
    X = pd.read_csv(config.X_RESAMPLED_FILE)
    y_df = pd.read_csv(config.Y_RESAMPLED_FILE)

    # Clean column names
    X.columns = X.columns.astype(str).str.strip()

    # Ensure y is 1D numpy int array
    if isinstance(y_df, pd.DataFrame):
        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0]
        else:
            raise ValueError("Y file contains multiple columns. Expected a single target column.")
    else:
        y = y_df

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    return X, y


# ============================================================
# PLOTTING HELPERS
# ============================================================
def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title="Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 4))

    # ✅ Pyrefly fix: xticklabels/yticklabels must be strings
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    save_plot(save_path)


def plot_roc_curve(y_true, y_proba, save_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    save_plot(save_path)

    return roc_auc


def plot_pr_curve(y_true, y_proba, save_path: Path) -> float:
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    ap = float(average_precision_score(y_true, y_proba))

    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    save_plot(save_path)

    return ap


def plot_metrics_bar(metrics_dict: dict, save_path: Path, title="Model Evaluation Metrics") -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()))
    plt.title(title)
    plt.ylim(0, 1)
    save_plot(save_path)


# ============================================================
# THRESHOLD TUNING
# ============================================================
def find_best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Find best threshold based on F1 score.
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    best_t: float = 0.5
    best_f1: float = -1.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = float(f1_score(y_true, preds))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


# ============================================================
# OVERFITTING CHECK
# ============================================================
def overfitting_check(train_accuracy: float, test_accuracy: float) -> str:
    gap = train_accuracy - test_accuracy
    msg = f"Training Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f} | Gap: {gap:.4f}"
    print(msg)

    if gap > 0.05:
        return "⚠️ Potential overfitting detected!"
    return "✅ No significant overfitting detected."


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_logistic_regression_torch(
    X: pd.DataFrame,
    y: np.ndarray,
    param_distributions=None,
    n_iter: int = 5,
    regularization: str | None = None,  # None, "l1", "l2"
    reg_lambda: float = 0.01,
    model_save_path: Path | None = None,
):
    """
    Train Logistic Regression (Torch) with random search.

    Uses:
    - Stratified train/test split
    - BCE loss
    - Optional L1 or L2 regularization
    - Threshold tuning for best F1
    """

    y = np.asarray(y, dtype=np.int64).reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    if param_distributions is None:
        param_distributions = {"lr": [0.001, 0.01, 0.1], "num_epochs": [50, 100, 200]}

    # ✅ Pyrefly-safe typing
    best_model: LogisticRegressionTorch | None = None
    best_score: float = -1.0
    best_params: dict | None = None
    best_threshold: float = 0.5

    # ----------------------------
    # Random search loop
    # ----------------------------
    for trial in range(n_iter):
        lr = float(np.random.choice(param_distributions["lr"]))
        num_epochs = int(np.random.choice(param_distributions["num_epochs"]))

        print("\n" + "=" * 70)
        print(f"Trial {trial + 1}/{n_iter}")
        print(f"lr={lr}, epochs={num_epochs}, regularization={regularization}, lambda={reg_lambda}")
        print("=" * 70)

        input_dim = int(X_train_t.shape[1])
        model = LogisticRegressionTorch(input_dim).to(device)

        if regularization == "l2":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.BCELoss()

        # Training
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            y_pred = model(X_train_t).squeeze()
            loss = criterion(y_pred, y_train_t)

            if regularization == "l1":
                l1_penalty = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l1_penalty = l1_penalty + torch.sum(torch.abs(param))
                loss = loss + reg_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.6f}")

        # ----------------------------
        # Evaluate
        # ----------------------------
        model.eval()
        with torch.no_grad():
            proba_test = model(X_test_t).squeeze().detach().cpu().numpy()
            proba_train = model(X_train_t).squeeze().detach().cpu().numpy()

        # Force numpy arrays (safety for Pyrefly)
        proba_test = np.asarray(proba_test, dtype=float)
        proba_train = np.asarray(proba_train, dtype=float)

        pred_test_default = (proba_test >= 0.5).astype(int)
        pred_train_default = (proba_train >= 0.5).astype(int)

        train_acc = float(accuracy_score(y_train, pred_train_default))
        test_acc = float(accuracy_score(y_test, pred_test_default))

        # ✅ Pyrefly fix: pass float(...) explicitly
        print(overfitting_check(float(train_acc), float(test_acc)))

        trial_threshold, trial_f1 = find_best_threshold_f1(y_test, proba_test)
        print(f"🏆 Best Threshold (F1) = {trial_threshold:.2f} | F1={trial_f1:.4f}")

        pred_test_tuned = (proba_test >= trial_threshold).astype(int)

        acc = float(accuracy_score(y_test, pred_test_tuned))
        f1 = float(f1_score(y_test, pred_test_tuned))
        rec = float(recall_score(y_test, pred_test_tuned))
        prec = float(precision_score(y_test, pred_test_tuned, zero_division=0))
        ap = float(average_precision_score(y_test, proba_test))

        print(f"Metrics @ tuned threshold={trial_threshold:.2f}")
        print(f"Accuracy={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f} | PR-AUC(AP)={ap:.4f}")

        comparison_df = pd.DataFrame(
            {
                "Actual": y_test.astype(int),
                "PredictedLabel": pred_test_tuned.astype(int),
                "PredictedProb": proba_test.astype(float),
            }
        )
        comparison_df["Correct"] = comparison_df["Actual"] == comparison_df["PredictedLabel"]

        wrong = comparison_df[comparison_df["Correct"] == False]
        print(f"Wrong predictions count: {len(wrong)}")

        if trial_f1 > best_score:
            best_score = float(trial_f1)
            best_model = model
            best_threshold = float(trial_threshold)
            best_params = {
                "lr": lr,
                "num_epochs": num_epochs,
                "regularization": regularization,
                "reg_lambda": reg_lambda,
                "best_threshold": best_threshold,
                "best_f1": best_score,
            }

    # ----------------------------
    # Final best model summary
    # ----------------------------
    print("\n" + "=" * 70)
    print("✅ BEST MODEL FOUND")
    print(best_params)
    print("=" * 70)

    # ✅ Pyrefly fix: best_model can be None
    if best_model is None or best_params is None:
        raise RuntimeError("❌ Training failed: best_model was never selected.")

    # Save weights
    if model_save_path is not None:
        torch.save(best_model.state_dict(), model_save_path)
        print(f"✅ Best model weights saved to: {model_save_path}")

        metadata_path = Path(str(model_save_path).replace(".pt", "_meta.json"))
        pd.Series(best_params).to_json(metadata_path)
        print(f"✅ Metadata saved to: {metadata_path}")

    # ----------------------------
    # Final evaluation plots
    # ----------------------------
    best_model.eval()
    with torch.no_grad():
        best_proba_test = best_model(X_test_t).squeeze().detach().cpu().numpy()

    # ✅ Pyrefly fix: ensure numpy array, not scalar/bool
    best_proba_test = np.asarray(best_proba_test, dtype=float)

    # ✅ Pyrefly fix: this is now ndarray -> supports astype
    best_pred_test = (best_proba_test >= best_threshold).astype(int)

    final_acc = float(accuracy_score(y_test, best_pred_test))
    final_f1 = float(f1_score(y_test, best_pred_test))
    final_rec = float(recall_score(y_test, best_pred_test))
    final_prec = float(precision_score(y_test, best_pred_test, zero_division=0))

    roc_auc = plot_roc_curve(
        y_test,
        best_proba_test,
        config.VISUALIZATION_DIR / "logreg_best_roc_curve.png",
    )

    ap = plot_pr_curve(
        y_test,
        best_proba_test,
        config.VISUALIZATION_DIR / "logreg_best_pr_curve.png",
    )

    plot_confusion_matrix(
        y_test,
        best_pred_test,
        config.VISUALIZATION_DIR / "logreg_best_confusion_matrix.png",
        title=f"Confusion Matrix (thr={best_threshold:.2f})",
    )

    plot_metrics_bar(
        {
            "Accuracy": final_acc,
            "F1": final_f1,
            "Recall": final_rec,
            "Precision": final_prec,
            "PR-AUC(AP)": ap,
        },
        config.VISUALIZATION_DIR / "logreg_best_metrics.png",
        title="Best Torch Logistic Regression Metrics",
    )

    # ----------------------------
    # PDF report
    # ----------------------------
    pdf_path = config.MODELS_DIR / "logreg_torch_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")

        # ✅ Pyrefly fix: dict must be converted to str explicitly
        best_params_str = str(best_params)

        report_text = (
            "Torch Logistic Regression Report\n\n"
            f"Best Params:\n{best_params_str}\n\n"
            f"Threshold: {best_threshold:.2f}\n"
            f"Accuracy: {final_acc:.4f}\n"
            f"F1: {final_f1:.4f}\n"
            f"Recall: {final_rec:.4f}\n"
            f"Precision: {final_prec:.4f}\n"
            f"ROC-AUC: {roc_auc:.4f}\n"
            f"PR-AUC(AP): {ap:.4f}\n\n"
            "Classification Report:\n"
            + str(classification_report(y_test, best_pred_test, zero_division=0))
        )

        ax.text(0.01, 0.5, report_text, fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n✅ PDF report saved to: {pdf_path}")

    return best_pred_test


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    X, y = load_feature_engineered_data()

    param_distributions = {
        "lr": [0.001, 0.01, 0.1],
        "num_epochs": [150, 300, 600],
    }

    regularization = "l1"  # "l1", "l2", or None
    reg_lambda = 0.01

    # ✅ IMPORTANT: use a .pt file for torch models
    model_path = config.BEST_TORCH_MODEL_FILE if hasattr(config, "BEST_TORCH_MODEL_FILE") else Path(config.BEST_MODEL_FILE)

    y_pred_test = train_logistic_regression_torch(
        X,
        y,
        param_distributions=param_distributions,
        n_iter=5,
        regularization=regularization,
        reg_lambda=reg_lambda,
        model_save_path=model_path,
    )
