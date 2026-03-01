"""
04_model_evaluation.py

Comprehensive model evaluation and interpretation.

This notebook:
1. Loads trained models and test data
2. Generates ROC and Precision-Recall curves
3. Creates confusion matrices
4. Performs SHAP analysis for explainability
5. Conducts subgroup fairness analysis
6. Generates clinical decision curve analysis
7. Creates publication-ready results table

Author: Jeremy Gracey
Date: 2024
"""

import logging
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from model_utils import (
    calculate_metrics, plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrices, create_model_comparison_table, plot_calibration_curves
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DATA_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = DATA_DIR / 'models'
PLOTS_DIR = DATA_DIR / 'plots'
REPORTS_DIR = DATA_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models_and_test_data() -> tuple:
    """Load best model and test data."""
    logger.info("=" * 80)
    logger.info("STEP 1: LOAD MODELS AND TEST DATA")
    logger.info("=" * 80)

    best_model = joblib.load(MODELS_DIR / 'best_model.pkl')
    test_data = joblib.load(MODELS_DIR / 'test_data.pkl')

    X_test_scaled = test_data['X_test_scaled']
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    logger.info(f"Loaded best model and test data")
    logger.info(f"Test set size: {len(y_test):,}")
    logger.info(f"Test readmission rate: {y_test.mean():.3f}")

    return best_model, X_test_scaled, X_test, y_test


def evaluate_on_test_set(best_model, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model on test set."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TEST SET EVALUATION")
    logger.info("=" * 80)

    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_metrics(y_test, y_pred_proba)

    logger.info(f"\nTest Set Performance:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")

    return metrics, y_pred_proba


def create_roc_pr_curves(y_test: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Create ROC and Precision-Recall curves."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATE ROC AND PR CURVES")
    logger.info("=" * 80)

    models_dict = {'Best Model': (y_test, y_pred_proba)}

    # ROC curve
    fig = plot_roc_curves(models_dict, output_path=PLOTS_DIR / '16_roc_curves.png')
    plt.close()

    # Precision-Recall curve
    fig = plot_precision_recall_curves(models_dict, output_path=PLOTS_DIR / '17_pr_curves.png')
    plt.close()

    # Calibration curve
    fig = plot_calibration_curves(models_dict, output_path=PLOTS_DIR / '18_calibration_curves.png')
    plt.close()


def create_confusion_matrices(y_test: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Create confusion matrices."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CONFUSION MATRICES")
    logger.info("=" * 80)

    models_dict = {'Best Model': (y_test, y_pred_proba)}
    fig = plot_confusion_matrices(models_dict, output_path=PLOTS_DIR / '19_confusion_matrices.png')
    plt.close()

    # Detailed analysis
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives: {tn:,}")
    logger.info(f"  False Positives: {fp:,}")
    logger.info(f"  False Negatives: {fn:,}")
    logger.info(f"  True Positives: {tp:,}")

    logger.info(f"\nClinical Interpretation:")
    logger.info(f"  Sensitivity (caught readmissions): {tp / (tp + fn):.1%}")
    logger.info(f"  Specificity (correct negatives): {tn / (tn + fp):.1%}")
    logger.info(f"  Positive Predictive Value: {tp / (tp + fp):.1%}")
    logger.info(f"  Negative Predictive Value: {tn / (tn + fn):.1%}")


def shap_analysis(best_model, X_test: np.ndarray, feature_names: list) -> None:
    """SHAP analysis for model explainability."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SHAP EXPLAINABILITY ANALYSIS")
    logger.info("=" * 80)

    try:
        # Use TreeExplainer for tree-based models
        if hasattr(best_model, 'booster'):  # XGBoost
            explainer = shap.TreeExplainer(best_model)
        elif hasattr(best_model, 'estimators_'):  # Random Forest
            explainer = shap.TreeExplainer(best_model)
        else:  # Other models
            explainer = shap.KernelExplainer(best_model.predict_proba, X_test[:100])

        shap_values = explainer.shap_values(X_test)

        # Handle both binary and multiclass outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get class 1 (readmission)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / '20_shap_summary.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot")
        plt.close()

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type='bar', feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / '21_shap_bar.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP bar plot")
        plt.close()

    except Exception as e:
        logger.warning(f"SHAP analysis encountered error: {e}")
        logger.info("Continuing without SHAP analysis")


def subgroup_fairness_analysis(X_test: pd.DataFrame, y_test: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Analyze model performance across subgroups."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: SUBGROUP FAIRNESS ANALYSIS")
    logger.info("=" * 80)

    # Age subgroups
    age_col = [col for col in X_test.columns if 'age' in col.lower()][0]
    X_test_df = pd.DataFrame(X_test)

    age_medians = X_test_df[age_col].quantile([0.33, 0.67])
    age_groups = pd.cut(X_test_df[age_col],
                        bins=[X_test_df[age_col].min() - 1, age_medians.iloc[0], age_medians.iloc[1], X_test_df[age_col].max() + 1],
                        labels=['Younger', 'Middle', 'Older'])

    subgroup_results = []

    for group_name, group_mask in [('Overall', slice(None)), ('Younger', age_groups == 'Younger'),
                                    ('Middle', age_groups == 'Middle'), ('Older', age_groups == 'Older')]:
        if group_name == 'Overall':
            y_group = y_test
            y_pred_group = y_pred_proba
        else:
            y_group = y_test[group_mask]
            y_pred_group = y_pred_proba[group_mask]

        if len(y_group) > 0:
            metrics = calculate_metrics(y_group, y_pred_group)
            subgroup_results.append({
                'Subgroup': group_name,
                'N': len(y_group),
                'AUC-ROC': metrics['auc_roc'],
                'Sensitivity': metrics['recall'],
                'Specificity': metrics['specificity']
            })

    subgroup_df = pd.DataFrame(subgroup_results)
    logger.info("\nSubgroup Performance Analysis:")
    print(subgroup_df.to_string(index=False))

    # Save subgroup analysis
    subgroup_df.to_csv(REPORTS_DIR / 'subgroup_analysis.csv', index=False)


def clinical_decision_curve(y_test: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Clinical decision curve analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: CLINICAL DECISION CURVE ANALYSIS")
    logger.info("=" * 80)

    thresholds = np.linspace(0, 1, 101)
    net_benefits = []

    for threshold in thresholds:
        # Positive predictions
        tp = ((y_pred_proba >= threshold) & (y_test == 1)).sum()
        fp = ((y_pred_proba >= threshold) & (y_test == 0)).sum()
        n = len(y_test)

        # Calculate net benefit
        sensitivity = tp / (y_test == 1).sum()
        specificity = 1 - (fp / (y_test == 0).sum())

        net_benefit = sensitivity * (y_test == 1).sum() / n - (1 - specificity) * (y_test == 0).sum() / n * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, net_benefits, linewidth=2, label='Model')
    ax.plot([0, 1], [0, (y_test == 1).mean()], 'k--', linewidth=2, label='Treat All')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Treat None')
    ax.set_xlabel('Probability Threshold', fontweight='bold')
    ax.set_ylabel('Net Benefit', fontweight='bold')
    ax.set_title('Clinical Decision Curve Analysis', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '22_decision_curve.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved clinical decision curve")
    plt.close()


def generate_results_table(metrics: dict, y_test: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Generate publication-ready results table."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: GENERATE RESULTS TABLE")
    logger.info("=" * 80)

    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results_table = pd.DataFrame({
        'Metric': [
            'AUC-ROC',
            'Sensitivity (Recall)',
            'Specificity',
            'Positive Predictive Value',
            'Negative Predictive Value',
            'Accuracy',
            'F1-Score'
        ],
        'Value': [
            f"{metrics['auc_roc']:.3f}",
            f"{tp / (tp + fn):.3f}",
            f"{tn / (tn + fp):.3f}",
            f"{tp / (tp + fp):.3f}",
            f"{tn / (tn + fn):.3f}",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['f1']:.3f}"
        ],
        '95% CI': ['', '', '', '', '', '', '']  # Would need bootstrap for real CIs
    })

    logger.info("\nFinal Results Table:")
    print(results_table.to_string(index=False))

    # Save results table
    results_table.to_csv(REPORTS_DIR / 'final_results.csv', index=False)
    logger.info(f"Saved results table to {REPORTS_DIR / 'final_results.csv'}")


def main():
    """Run comprehensive model evaluation."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("30-DAY HOSPITAL READMISSION - MODEL EVALUATION")
    logger.info("=" * 80)

    # Load models and data
    best_model, X_test_scaled, X_test, y_test = load_models_and_test_data()

    # Get feature names
    data_path = DATA_DIR / 'readmission_features.csv'
    df = pd.read_csv(data_path)
    feature_names = [col for col in df.columns if col != 'readmitted_30day']

    # Evaluate
    metrics, y_pred_proba = evaluate_on_test_set(best_model, X_test_scaled, y_test)

    # Analyses
    create_roc_pr_curves(y_test, y_pred_proba)
    create_confusion_matrices(y_test, y_pred_proba)
    shap_analysis(best_model, X_test, feature_names)
    subgroup_fairness_analysis(X_test, y_test, y_pred_proba)
    clinical_decision_curve(y_test, y_pred_proba)
    generate_results_table(metrics, y_test, y_pred_proba)

    logger.info("\n" + "=" * 80)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
