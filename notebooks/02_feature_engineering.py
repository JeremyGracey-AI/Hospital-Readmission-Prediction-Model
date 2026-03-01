"""
02_feature_engineering.py

Feature engineering for 30-day hospital readmission prediction.

This notebook:
1. Loads raw data from EDA step
2. Creates clinical risk scores (Charlson, LACE)
3. Extracts temporal features
4. Creates aggregated features
5. Transforms lab values
6. Handles missing data with clinical logic
7. Encodes categorical variables
8. Performs feature selection
9. Saves feature-engineered dataset for modeling

Author: Jeremy Gracey
Date: 2024
"""

import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from feature_pipeline import ClinicalFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
DATA_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR = DATA_DIR / 'plots'
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """
    Load raw dataset from EDA step.

    Returns
    -------
    pd.DataFrame
        Raw patient dataset
    """
    logger.info("=" * 80)
    logger.info("STEP 1: LOAD RAW DATA")
    logger.info("=" * 80)

    data_path = DATA_DIR / 'readmission_raw_data.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape {df.shape}")
    logger.info(f"Readmission rate: {df['readmitted_30day'].mean():.3f}")

    return df


def create_clinical_risk_scores(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> pd.DataFrame:
    """
    Create clinical risk scores (Charlson, LACE).

    Clinical context:
    - Charlson Comorbidity Index: Validated predictor of mortality and complications
    - LACE Index: Specific readmission prediction tool, scores >10 indicate high risk
    These are well-established in clinical literature.

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient data
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine

    Returns
    -------
    pd.DataFrame
        Data with clinical risk scores added
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATE CLINICAL RISK SCORES")
    logger.info("=" * 80)

    df = df.copy()

    # Charlson Index
    df['charlson_index'] = df['primary_diagnosis'].apply(fe_engine.calculate_charlson_index)

    # LACE Index
    is_acute = df['admission_source'].isin(['Emergency', 'Urgent']).astype(int)
    df['lace_index'] = df.apply(
        lambda row: fe_engine.calculate_lace_index(
            row['length_of_stay'],
            is_acute[row.name],
            row['charlson_index'],
            row['ed_visits_6mo']
        ),
        axis=1
    )

    logger.info(f"Charlson Index - Mean: {df['charlson_index'].mean():.2f}, Std: {df['charlson_index'].std():.2f}")
    logger.info(f"LACE Index - Mean: {df['lace_index'].mean():.2f}, Std: {df['lace_index'].std():.2f}")

    # Visualize risk scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['charlson_index'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Charlson Index', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Charlson Comorbidity Index Distribution', fontsize=12, fontweight='bold')

    axes[1].hist(df['lace_index'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('LACE Index', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('LACE Readmission Risk Score Distribution', fontsize=12, fontweight='bold')
    axes[1].axvline(10, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_clinical_risk_scores.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved risk scores plot to {OUTPUT_DIR / '10_clinical_risk_scores.png'}")
    plt.close()

    return df


def engineer_temporal_features(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> pd.DataFrame:
    """
    Extract temporal features from admission/discharge dates.

    Clinical context: Readmission patterns vary by season and day of week.
    Winter has higher rates for respiratory conditions.
    Weekend discharges may lack follow-up coordination.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data with dates
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine

    Returns
    -------
    pd.DataFrame
        Data with temporal features
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATE TEMPORAL FEATURES")
    logger.info("=" * 80)

    df = df.copy()
    df = fe_engine.create_temporal_features(df)

    logger.info("Created temporal features:")
    logger.info(f"- Day of week (0=Mon, 6=Sun)")
    logger.info(f"- Weekend discharge flag")
    logger.info(f"- Month and season")

    # Visualize temporal patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Readmission by day of week
    dow_readmit = df.groupby('admission_day_of_week')['readmitted_30day'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 0].bar(range(7), dow_readmit.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(7))
    axes[0, 0].set_xticklabels(day_names)
    axes[0, 0].set_ylabel('Readmission Rate', fontweight='bold')
    axes[0, 0].set_title('Readmission by Admission Day of Week', fontsize=12, fontweight='bold')
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Weekend discharge effect
    weekend_readmit = df.groupby('weekend_discharge')['readmitted_30day'].mean()
    axes[0, 1].bar(['Weekday', 'Weekend'], weekend_readmit.values, color=['lightcoral', 'salmon'], edgecolor='black')
    axes[0, 1].set_ylabel('Readmission Rate', fontweight='bold')
    axes[0, 1].set_title('Weekend Discharge Effect', fontsize=12, fontweight='bold')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Readmission by season
    season_readmit = df.groupby('season')['readmitted_30day'].mean().sort_values(ascending=False)
    axes[1, 0].bar(season_readmit.index, season_readmit.values, color='seagreen', edgecolor='black')
    axes[1, 0].set_ylabel('Readmission Rate', fontweight='bold')
    axes[1, 0].set_title('Readmission by Season', fontsize=12, fontweight='bold')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Admission count by month
    month_counts = df['admission_month'].value_counts().sort_index()
    axes[1, 1].bar(month_counts.index, month_counts.values, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xlabel('Month', fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Admissions by Month', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_temporal_features.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved temporal features plot to {OUTPUT_DIR / '11_temporal_features.png'}")
    plt.close()

    return df


def engineer_aggregated_features(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> pd.DataFrame:
    """
    Create aggregated features capturing disease complexity.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine

    Returns
    -------
    pd.DataFrame
        Data with aggregated features
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CREATE AGGREGATED FEATURES")
    logger.info("=" * 80)

    df = df.copy()
    df = fe_engine.create_aggregated_features(df)

    logger.info("Created aggregated features:")
    logger.info("- Medication burden categories")
    logger.info("- Comorbidity burden categories")
    logger.info("- Total utilization (12 months)")

    # Visualize aggregated features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Medication burden vs readmission
    med_burden_readmit = df.groupby('medication_burden', observed=True)['readmitted_30day'].mean()
    axes[0, 0].bar(range(len(med_burden_readmit)), med_burden_readmit.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(med_burden_readmit)))
    axes[0, 0].set_xticklabels(med_burden_readmit.index, rotation=45)
    axes[0, 0].set_ylabel('Readmission Rate', fontweight='bold')
    axes[0, 0].set_title('Readmission by Medication Burden', fontsize=12, fontweight='bold')
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Comorbidity burden vs readmission
    comorbid_readmit = df.groupby('comorbidity_burden', observed=True)['readmitted_30day'].mean()
    axes[0, 1].bar(range(len(comorbid_readmit)), comorbid_readmit.values, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xticks(range(len(comorbid_readmit)))
    axes[0, 1].set_xticklabels(comorbid_readmit.index, rotation=45)
    axes[0, 1].set_ylabel('Readmission Rate', fontweight='bold')
    axes[0, 1].set_title('Readmission by Comorbidity Burden', fontsize=12, fontweight='bold')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Total utilization vs readmission
    total_util = df['admissions_12mo'] + df['ed_visits_12mo']
    util_readmit = df.groupby(total_util)['readmitted_30day'].mean()
    axes[1, 0].plot(util_readmit.index, util_readmit.values, marker='o', linewidth=2, markersize=8, color='seagreen')
    axes[1, 0].set_xlabel('Total Healthcare Visits (12 months)', fontweight='bold')
    axes[1, 0].set_ylabel('Readmission Rate', fontweight='bold')
    axes[1, 0].set_title('Readmission by Total Utilization', fontsize=12, fontweight='bold')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[1, 0].grid(True, alpha=0.3)

    # LACE index risk stratification
    df['lace_risk_category'] = pd.cut(
        df['lace_index'],
        bins=[0, 5, 10, 32],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    lace_readmit = df.groupby('lace_risk_category')['readmitted_30day'].mean()
    axes[1, 1].bar(range(len(lace_readmit)), lace_readmit.values, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xticks(range(len(lace_readmit)))
    axes[1, 1].set_xticklabels(lace_readmit.index)
    axes[1, 1].set_ylabel('Readmission Rate', fontweight='bold')
    axes[1, 1].set_title('Readmission by LACE Risk Category', fontsize=12, fontweight='bold')
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_aggregated_features.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved aggregated features plot to {OUTPUT_DIR / '12_aggregated_features.png'}")
    plt.close()

    return df


def engineer_lab_features(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> pd.DataFrame:
    """
    Transform lab values into clinical features.

    Creates abnormality flags and composite measures.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data with labs
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine

    Returns
    -------
    pd.DataFrame
        Data with lab-derived features
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: CREATE LAB-DERIVED FEATURES")
    logger.info("=" * 80)

    df = df.copy()
    df = fe_engine.create_lab_features(df)

    logger.info("Created lab-derived features:")
    logger.info("- Abnormality flags for each lab")
    logger.info("- Kidney disease indicator (eGFR < 60)")
    logger.info("- Anemia indicator (Hgb < 11)")
    logger.info("- Count of abnormal labs")

    # Log lab abnormality prevalence
    abnormal_cols = [col for col in df.columns if 'abnormal' in col]
    logger.info("\nLab Abnormality Prevalence:")
    for col in abnormal_cols:
        pct = df[col].mean()
        logger.info(f"  {col}: {pct:.1%}")

    # Visualize lab features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count of abnormal labs vs readmission
    abnormal_readmit = df.groupby('num_abnormal_labs')['readmitted_30day'].mean()
    axes[0].bar(abnormal_readmit.index, abnormal_readmit.values, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Number of Abnormal Labs', fontweight='bold')
    axes[0].set_ylabel('Readmission Rate', fontweight='bold')
    axes[0].set_title('Readmission by Lab Abnormalities', fontsize=12, fontweight='bold')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Kidney disease and anemia indicators
    clinical_flags = ['kidney_disease', 'anemia']
    flag_readmit = []
    flag_labels = []

    for flag in clinical_flags:
        if flag in df.columns:
            positive_rate = df[df[flag] == 1]['readmitted_30day'].mean()
            flag_readmit.append(positive_rate)
            flag_labels.append(flag.replace('_', ' ').title())

    axes[1].bar(flag_labels, flag_readmit, color='lightcoral', edgecolor='black')
    axes[1].set_ylabel('Readmission Rate (Positive Cases)', fontweight='bold')
    axes[1].set_title('Readmission Rates by Clinical Condition', fontsize=12, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '13_lab_features.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved lab features plot to {OUTPUT_DIR / '13_lab_features.png'}")
    plt.close()

    return df


def handle_missing_data(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> pd.DataFrame:
    """
    Handle missing data with clinical reasoning.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data with potential missing values
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine

    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: HANDLE MISSING DATA")
    logger.info("=" * 80)

    df = df.copy()

    # Log before
    logger.info("\nMissing data BEFORE imputation:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])

    # Apply missing data handling
    df = fe_engine.handle_missing_data(df)

    # Log after
    logger.info("\nMissing data AFTER imputation:")
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(missing_after[missing_after > 0])
    else:
        logger.info("No missing values remaining")

    return df


def encode_categorical_variables(
    df: pd.DataFrame,
    fe_engine: ClinicalFeatureEngineering,
    fit: bool = True
) -> pd.DataFrame:
    """
    Encode categorical variables using label encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data with categorical variables
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine
    fit : bool
        Whether to fit new encoders or use existing

    Returns
    -------
    pd.DataFrame
        Data with encoded categorical variables
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: ENCODE CATEGORICAL VARIABLES")
    logger.info("=" * 80)

    df = df.copy()
    df = fe_engine.encode_categorical_variables(df, fit=fit)

    logger.info(f"Encoded {len(fe_engine.label_encoders)} categorical features")

    return df


def perform_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select features for modeling based on clinical relevance and statistical association.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset

    Returns
    -------
    pd.DataFrame
        Dataset with selected features
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: FEATURE SELECTION")
    logger.info("=" * 80)

    # Define feature groups for modeling
    demographic_features = ['age', 'sex_encoded', 'race_ethnicity_encoded', 'insurance_encoded']
    clinical_features = ['charlson_index', 'lace_index', 'num_diagnoses', 'num_medications']
    temporal_features = ['admission_day_of_week', 'weekend_discharge', 'season_encoded']
    utilization_features = ['admissions_6mo', 'admissions_12mo', 'ed_visits_6mo', 'ed_visits_12mo']
    lab_features = [
        'hemoglobin', 'hematocrit', 'wbc', 'creatinine', 'egfr',
        'albumin', 'glucose', 'hba1c',
        'kidney_disease', 'anemia', 'num_abnormal_labs'
    ]
    los_features = ['length_of_stay']

    all_features = (
        demographic_features + clinical_features + temporal_features +
        utilization_features + lab_features + los_features
    )

    # Filter to only features that exist
    selected_features = [f for f in all_features if f in df.columns]

    logger.info(f"\nSelected {len(selected_features)} features for modeling:")
    logger.info(f"- Demographics: {len([f for f in demographic_features if f in df.columns])}")
    logger.info(f"- Clinical: {len([f for f in clinical_features if f in df.columns])}")
    logger.info(f"- Temporal: {len([f for f in temporal_features if f in df.columns])}")
    logger.info(f"- Utilization: {len([f for f in utilization_features if f in df.columns])}")
    logger.info(f"- Lab: {len([f for f in lab_features if f in df.columns])}")
    logger.info(f"- LOS: {len([f for f in los_features if f in df.columns])}")

    # Analyze feature correlations with target
    numeric_features = df[selected_features + ['readmitted_30day']].select_dtypes(include=[np.number]).columns
    correlations = df[numeric_features].corr()['readmitted_30day'].drop('readmitted_30day').sort_values(ascending=False)

    logger.info("\nTop 15 features by correlation with readmission:")
    logger.info(correlations.head(15))

    # Visualize feature selection
    fig, ax = plt.subplots(figsize=(10, 8))
    correlations_abs = correlations.abs().sort_values(ascending=True).tail(15)
    colors = ['green' if x > 0 else 'red' for x in correlations[correlations_abs.index]]
    ax.barh(range(len(correlations_abs)), correlations_abs.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(correlations_abs)))
    ax.set_yticklabels(correlations_abs.index)
    ax.set_xlabel('Absolute Correlation with Readmission', fontweight='bold')
    ax.set_title('Top 15 Features by Readmission Correlation', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '14_feature_selection.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature selection plot to {OUTPUT_DIR / '14_feature_selection.png'}")
    plt.close()

    return df[selected_features + ['readmitted_30day']]


def save_engineered_data(df: pd.DataFrame) -> None:
    """
    Save feature-engineered dataset for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: SAVE ENGINEERED DATA")
    logger.info("=" * 80)

    output_path = DATA_DIR / 'readmission_features.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"Saved engineered dataset to {output_path}")
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Readmission rate: {df['readmitted_30day'].mean():.3f}")


def generate_feature_report(df: pd.DataFrame, fe_engine: ClinicalFeatureEngineering) -> None:
    """
    Generate feature engineering summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset
    fe_engine : ClinicalFeatureEngineering
        Feature engineering engine
    """
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 80)

    # Get feature reference
    feature_ref = fe_engine.get_feature_importance_reference()

    report = f"""
FEATURE ENGINEERING OVERVIEW
============================
Total Features: {df.shape[1] - 1}  (excluding target)
Samples: {df.shape[0]:,}

FEATURE CATEGORIES
==================
1. Demographics (5): age, sex, race/ethnicity, insurance
2. Clinical Risk Scores (2): Charlson Index, LACE Index
3. Comorbidities (1): num_diagnoses
4. Medications (1): num_medications
5. Temporal (3): day_of_week, weekend_discharge, season
6. Prior Utilization (4): admissions_6mo/12mo, ed_visits_6mo/12mo
7. Lab Values (11): hemoglobin, hematocrit, WBC, creatinine, eGFR, albumin, glucose, HbA1c, kidney_disease, anemia, num_abnormal_labs
8. Length of Stay (1): length_of_stay

KEY CLINICAL RISK SCORES
==========================
Charlson Index:
- Mean: {df['charlson_index'].mean():.2f}
- Captures comorbidity burden
- Used in mortality prediction

LACE Index:
- Mean: {df['lace_index'].mean():.2f}
- L: Length of stay
- A: Acute admission
- C: Charlson comorbidity
- E: ED visits (past 6 months)
- Validated for 30-day readmission risk

MISSING DATA HANDLING
=====================
Strategy: Clinical domain knowledge-based imputation
- Lab values: Median imputation (robust to outliers)
- Utilization: Fill with 0 if no history
- Missing flags: Created for important missingness patterns

CLASS BALANCE
=============
No readmission: {(df['readmitted_30day'] == 0).sum():,} ({(df['readmitted_30day'] == 0).mean():.1%})
Readmission: {(df['readmitted_30day'] == 1).sum():,} ({(df['readmitted_30day'] == 1).mean():.1%})

NEXT STEPS
==========
1. Split data into train/validation/test sets
2. Apply feature scaling for distance-based models
3. Train multiple models and compare
4. Tune hyperparameters
5. Evaluate on test set with comprehensive metrics
6. Generate SHAP explanations for interpretability
"""

    logger.info(report)

    # Save report
    report_path = DATA_DIR / 'FEATURE_ENGINEERING_SUMMARY.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\nSaved feature engineering report to {report_path}")


def main():
    """Run complete feature engineering pipeline."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("30-DAY HOSPITAL READMISSION - FEATURE ENGINEERING")
    logger.info("=" * 80)

    # Load raw data
    df = load_raw_data()

    # Initialize feature engineering engine
    fe_engine = ClinicalFeatureEngineering(random_seed=42)

    # Feature engineering steps
    df = create_clinical_risk_scores(df, fe_engine)
    df = engineer_temporal_features(df, fe_engine)
    df = engineer_aggregated_features(df, fe_engine)
    df = engineer_lab_features(df, fe_engine)
    df = handle_missing_data(df, fe_engine)
    df = encode_categorical_variables(df, fe_engine, fit=True)
    df = perform_feature_selection(df)

    # Save engineered data
    save_engineered_data(df)

    # Generate report
    generate_feature_report(df, fe_engine)

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
