import torch
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.dataset import create_dataloaders
from src.hqcnn import HQCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "hqcnn_vqc3.pth"


# ============================================================
# EVALUATE SAVED MODEL
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # --------------------------------------------------------
    # NO TRAINING
    # NO BACKPROPAGATION
    # NO OPTIMIZER
    # --------------------------------------------------------

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # Forward pass only
            outputs = model(X)

            # Convert logits to probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            # Predicted class
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Store results
            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Probability of NLOS = class 1
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    # Specificity
    if (TN + FP) > 0:
        specificity = TN / (TN + FP)
    else:
        specificity = 0.0

    # AUC
    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    except ValueError:
        auc = float("nan")

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("HQCNN-VQC3 TEST EVALUATION")
    print("=" * 60)

    print()
    print(f"Test Samples : {len(all_labels)}")

    print("\n")
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy     : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision    : {precision * 100:.2f}%"
    )

    print(
        f"Recall       : {recall * 100:.2f}%"
    )

    print(
        f"Specificity  : {specificity * 100:.2f}%"
    )

    print(
        f"F1 Score     : {f1 * 100:.2f}%"
    )

    print(
        f"AUC          : {auc:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print()
    print("                 Predicted")
    print("                 LOS   NLOS")
    print()

    print(
        f"Actual LOS      {TN:4d}  {FP:4d}"
    )

    print(
        f"Actual NLOS     {FN:4d}  {TP:4d}"
    )

    print("\n")
    print("Confusion Matrix Components:")
    print(f"TN = {TN}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"TP = {TP}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "LOS",
                "NLOS"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HQCNN-VQC3 EVALUATION ONLY")
    print("=" * 60)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print("\nDevice:")
    print(DEVICE)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print("\nDataset loaded successfully.")

    print(
        f"Test batches : {len(test_loader)}"
    )

    # --------------------------------------------------------
    # CREATE MODEL ARCHITECTURE
    # --------------------------------------------------------

    print("\nCreating HQCNN model...")

    model = HQCNN()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOAD SAVED TRAINED PARAMETERS
    # --------------------------------------------------------

    print(
        f"\nLoading trained model from: {MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    print("✓ Trained model loaded successfully.")

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print("\nEvaluating on TEST SET...")
    print("Training will NOT happen.")

    results = evaluate_model(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HQCNN-VQC3 FINAL SUMMARY")
    print("=" * 60)

    print(
        f"Test Accuracy : "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision     : "
        f"{results['precision'] * 100:.2f}%"
    )

    print(
        f"Recall        : "
        f"{results['recall'] * 100:.2f}%"
    )

    print(
        f"Specificity   : "
        f"{results['specificity'] * 100:.2f}%"
    )

    print(
        f"F1 Score      : "
        f"{results['f1'] * 100:.2f}%"
    )

    print(
        f"AUC           : "
        f"{results['auc']:.4f}"
    )

    print("\nDone.")