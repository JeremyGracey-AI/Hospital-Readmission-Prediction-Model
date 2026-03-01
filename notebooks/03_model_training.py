"""
03_model_training.py

Model training for 30-day hospital readmission prediction.

This notebook:
1. Loads feature-engineered data
2. Splits into train/validation/test sets with stratification
3. Handles class imbalance (SMOTE and class weights)
4. Trains multiple models: Logistic Regression, Random Forest, XGBoost, Neural Network
5. Performs hyperparameter tuning with cross-validation
6. Evaluates models on validation set
7. Creates model comparison table
8. Saves best performing model

Author: Jeremy Gracey
Date: 2024
"""

import logging
import sys
from pathlib import Path
import joblib

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
from model_utils import calculate_metrics, create_model_comparison_table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)

# Define paths
DATA_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = DATA_DIR / 'models'
PLOTS_DIR = DATA_DIR / 'plots'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data() -> tuple:
    """
    Load feature-engineered data and split into train/validation/test.

    Uses stratified split to maintain class distribution.
    Typical split: 60% train, 20% validation, 20% test

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: LOAD AND SPLIT DATA")
    logger.info("=" * 80)

    # Load data
    data_path = DATA_DIR / 'readmission_features.csv'
    df = pd.read_csv(data_path)

    logger.info(f"Loaded data with shape {df.shape}")
    logger.info(f"Readmission rate: {df['readmitted_30day'].mean():.3f}")

    # Separate features and target
    X = df.drop('readmitted_30day', axis=1)
    y = df['readmitted_30day']

    # First split: 80% for train+val, 20% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: 75% for train (60% of total), 25% for val (20% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {X_train.shape[0]:,} ({X_train.shape[0]/len(df):.1%})")
    logger.info(f"  Validation: {X_val.shape[0]:,} ({X_val.shape[0]/len(df):.1%})")
    logger.info(f"  Test: {X_test.shape[0]:,} ({X_test.shape[0]/len(df):.1%})")

    logger.info(f"\nClass distribution (readmission rate):")
    logger.info(f"  Train: {y_train.mean():.3f}")
    logger.info(f"  Validation: {y_val.mean():.3f}")
    logger.info(f"  Test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple:
    """
    Apply SMOTE to balance training data.

    Clinical context: Class imbalance (83% no readmission, 17% readmission)
    requires careful handling. SMOTE creates synthetic minority examples
    to balance classes without throwing away majority examples.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target

    Returns
    -------
    tuple
        (X_train_balanced, y_train_balanced)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: HANDLE CLASS IMBALANCE WITH SMOTE")
    logger.info("=" * 80)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    logger.info(f"Before SMOTE:")
    logger.info(f"  No readmission: {(y_train == 0).sum():,}")
    logger.info(f"  Readmission: {(y_train == 1).sum():,}")
    logger.info(f"  Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

    logger.info(f"\nAfter SMOTE:")
    logger.info(f"  No readmission: {(y_train_balanced == 0).sum():,}")
    logger.info(f"  Readmission: {(y_train_balanced == 1).sum():,}")
    logger.info(f"  Ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.2f}:1")

    return X_train_balanced, y_train_balanced


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale features using StandardScaler.

    Important for distance-based models (logistic regression) and neural networks.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Train, validation, test features

    Returns
    -------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: SCALE FEATURES")
    logger.info("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Applied StandardScaler to features")
    logger.info(f"Feature means (train): {X_train_scaled.mean(axis=0)[:5]}...")  # Show first 5
    logger.info(f"Feature stds (train): {X_train_scaled.std(axis=0)[:5]}...")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    """
    Train logistic regression baseline model.

    Logistic regression is interpretable and serves as a good baseline.
    Class weights handle imbalance.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train, y_val : np.ndarray
        Training and validation targets

    Returns
    -------
    dict
        Dictionary with model, predictions, and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: LOGISTIC REGRESSION")
    logger.info("=" * 80)

    # Train baseline model
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr.fit(X_train, y_train)

    # Predictions
    y_val_pred_proba = lr.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_val_pred_proba)

    logger.info(f"Logistic Regression Performance:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")

    return {
        'model': lr,
        'y_pred_proba': y_val_pred_proba,
        'metrics': metrics,
        'name': 'Logistic Regression'
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    """
    Train Random Forest with hyperparameter tuning.

    Random forests handle non-linear relationships and interactions well.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train, y_val : np.ndarray
        Training and validation targets

    Returns
    -------
    dict
        Dictionary with model, predictions, and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: RANDOM FOREST")
    logger.info("=" * 80)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [4, 8],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation AUC-ROC: {grid_search.best_score_:.4f}")

    # Get best model
    best_rf = grid_search.best_estimator_

    # Predictions
    y_val_pred_proba = best_rf.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_val_pred_proba)

    logger.info(f"Random Forest Performance (Validation):")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")

    return {
        'model': best_rf,
        'y_pred_proba': y_val_pred_proba,
        'metrics': metrics,
        'name': 'Random Forest'
    }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    """
    Train XGBoost with hyperparameter tuning.

    XGBoost is a powerful gradient boosting algorithm with good generalization.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train, y_val : np.ndarray
        Training and validation targets

    Returns
    -------
    dict
        Dictionary with model, predictions, and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: XGBOOST")
    logger.info("=" * 80)

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Hyperparameter tuning
    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
    }

    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation AUC-ROC: {grid_search.best_score_:.4f}")

    # Get best model
    best_xgb = grid_search.best_estimator_

    # Predictions
    y_val_pred_proba = best_xgb.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_val_pred_proba)

    logger.info(f"XGBoost Performance (Validation):")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")

    return {
        'model': best_xgb,
        'y_pred_proba': y_val_pred_proba,
        'metrics': metrics,
        'name': 'XGBoost'
    }


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    """
    Train neural network (MLPClassifier).

    Neural networks can capture complex non-linear patterns.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train, y_val : np.ndarray
        Training and validation targets

    Returns
    -------
    dict
        Dictionary with model, predictions, and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: NEURAL NETWORK")
    logger.info("=" * 80)

    # Initialize model
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    )

    nn.fit(X_train, y_train)

    logger.info(f"Neural Network trained with architecture: {nn.hidden_layer_sizes}")
    logger.info(f"Converged: {nn.n_iter_} iterations")

    # Predictions
    y_val_pred_proba = nn.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_val_pred_proba)

    logger.info(f"Neural Network Performance (Validation):")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")

    return {
        'model': nn,
        'y_pred_proba': y_val_pred_proba,
        'metrics': metrics,
        'name': 'Neural Network'
    }


def compare_models(models_dict: dict, y_val: np.ndarray) -> pd.DataFrame:
    """
    Create comprehensive model comparison table.

    Parameters
    ----------
    models_dict : dict
        Dictionary of trained models and predictions
    y_val : np.ndarray
        Validation target values

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    results = []

    for model_name, model_info in models_dict.items():
        y_pred_proba = model_info['y_pred_proba']
        metrics = calculate_metrics(y_val, y_pred_proba)

        results.append({
            'Model': model_name,
            'AUC-ROC': metrics['auc_roc'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'F1-Score': metrics['f1']
        })

    comparison_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)

    logger.info("\nModel Performance Comparison (Validation Set):")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_path = DATA_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nSaved comparison table to {comparison_path}")

    return comparison_df


def visualize_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Visualize model comparison.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison results
    """
    metrics_to_plot = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.2

    for idx, metric in enumerate(metrics_to_plot):
        ax.bar(x + idx * width, comparison_df[metric], width, label=metric, edgecolor='black')

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison (Validation Set)', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '15_model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved model comparison plot to {PLOTS_DIR / '15_model_comparison.png'}")
    plt.close()


def save_best_model(models_dict: dict) -> None:
    """
    Save best performing model.

    Parameters
    ----------
    models_dict : dict
        Dictionary of trained models
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVE BEST MODEL")
    logger.info("=" * 80)

    # Find best model by AUC-ROC
    best_model_name = max(models_dict, key=lambda x: models_dict[x]['metrics']['auc_roc'])
    best_model = models_dict[best_model_name]['model']
    best_auc = models_dict[best_model_name]['metrics']['auc_roc']

    logger.info(f"Best model: {best_model_name} (AUC-ROC: {best_auc:.4f})")

    # Save model
    model_path = MODELS_DIR / 'best_model.pkl'
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to {model_path}")


def main():
    """Run complete model training pipeline."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("30-DAY HOSPITAL READMISSION - MODEL TRAINING")
    logger.info("=" * 80)

    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_balanced, X_val, X_test
    )

    # Save scaler
    scaler_path = MODELS_DIR / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Train models
    models_dict = {}

    models_dict['Logistic Regression'] = train_logistic_regression(
        X_train_scaled, y_train_balanced, X_val_scaled, y_val
    )

    models_dict['Random Forest'] = train_random_forest(
        X_train_balanced, y_train_balanced, X_val, y_val
    )

    models_dict['XGBoost'] = train_xgboost(
        X_train_balanced, y_train_balanced, X_val, y_val
    )

    models_dict['Neural Network'] = train_neural_network(
        X_train_scaled, y_train_balanced, X_val_scaled, y_val
    )

    # Compare models
    comparison_df = compare_models(models_dict, y_val.values)

    # Visualize comparison
    visualize_model_comparison(comparison_df)

    # Save best model
    save_best_model(models_dict)

    # Save test set for evaluation
    test_data = {
        'X_test_scaled': X_test_scaled,
        'X_test': X_test.values,
        'y_test': y_test.values,
        'scaler': scaler
    }
    joblib.dump(test_data, MODELS_DIR / 'test_data.pkl')
    logger.info(f"Saved test data to {MODELS_DIR / 'test_data.pkl'}")

    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
