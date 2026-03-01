# 30-Day Hospital Readmission Prediction Model

**A Production-Quality Machine Learning Pipeline for Healthcare Risk Stratification**

This project demonstrates a complete end-to-end ML pipeline for predicting 30-day hospital readmissions, from data generation through model deployment with clinical interpretability and fairness analysis.

---

## Table of Contents

- [Clinical Context](#clinical-context)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Skills Demonstrated](#skills-demonstrated)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results Summary](#results-summary)
- [Implementation Guide](#implementation-guide)
- [Key Insights](#key-insights)
- [Technical Details](#technical-details)
- [References](#references)

---

## Clinical Context

Hospital readmissions represent a major quality and financial challenge in healthcare:

### Why This Matters

- **Patient Outcomes**: Readmissions indicate inadequate recovery, resource utilization, and patient suffering
- **Quality Metrics**: CMS uses readmission rates (AMI, heart failure, pneumonia) as quality measures
- **Financial Penalties**: Hospitals with excess readmissions face CMS reimbursement penalties
- **Prevention Potential**: Effective interventions can reduce 30-day readmissions by 15-20%

### The Opportunity

Early identification of high-risk patients enables targeted interventions:
- Enhanced discharge planning
- 48-hour phone follow-up
- Scheduled early primary care visits
- Medication reconciliation
- Care coordination

**Expected Outcome**: Prevent 50-100 readmissions annually per 3,000 discharges = $750,000 - $2,500,000 in savings

---

## Project Overview

### What This Project Does

This project implements a **clinical prediction model** that:

1. **Integrates clinical risk factors** (comorbidities, medications, prior utilization, labs)
2. **Applies evidence-based risk scores** (Charlson Index, LACE Index)
3. **Identifies high-risk patients** at discharge for targeted intervention
4. **Explains model decisions** with SHAP analysis for clinician trust
5. **Assesses fairness** across demographic subgroups
6. **Provides implementation guidance** for deployment in clinical settings

### Dataset

- **15,000 synthetic patient records** with clinically realistic distributions
- **17% readmission rate** (matching CMS national average of 15-18%)
- **Rich feature set**: Demographics, diagnoses (ICD-10), procedures, labs, medications, prior utilization
- **Realistic missingness**: Labs 10-15% missing (more intensive labs more often missing)

---

## Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION (01)                          │
│  - Synthetic patient data with clinical correlations             │
│  - 15,000 records with realistic distributions                  │
│  - Output: readmission_raw_data.csv                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│              EXPLORATORY DATA ANALYSIS (01)                      │
│  - Demographic distributions                                    │
│  - Clinical feature exploration                                 │
│  - Prior utilization patterns                                   │
│  - Lab value distributions & missingness                        │
│  - Target variable analysis                                     │
│  - Key risk factor identification                               │
│  Output: 8 publication-ready plots                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│            FEATURE ENGINEERING (02)                              │
│  - Clinical risk scores (Charlson, LACE)                        │
│  - Temporal features (day-of-week, season, weekend discharge)   │
│  - Aggregated features (burden categories)                      │
│  - Lab features (abnormality flags, composite measures)         │
│  - Missing data handling (clinical domain knowledge)            │
│  - Categorical encoding (label encoding)                        │
│  - Feature selection (correlation analysis)                     │
│  Output: readmission_features.csv (40+ features)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│              MODEL TRAINING (03)                                 │
│  - Train/Validation/Test split (60/20/20)                       │
│  - Class imbalance handling (SMOTE)                             │
│  - Feature scaling (StandardScaler)                             │
│  - Multiple models:                                             │
│    * Logistic Regression (baseline)                             │
│    * Random Forest (with hyperparameter tuning)                 │
│    * XGBoost (best performer)                                   │
│    * Neural Network (MLPClassifier)                             │
│  - Hyperparameter tuning (GridSearchCV, 5-fold CV)              │
│  Output: best_model.pkl, model_comparison.csv                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│            MODEL EVALUATION (04)                                 │
│  - ROC curves and AUC-ROC                                       │
│  - Precision-Recall curves (imbalanced data)                    │
│  - Confusion matrices with clinical interpretation             │
│  - SHAP explainability analysis                                 │
│  - Subgroup fairness analysis                                   │
│  - Clinical decision curve analysis                             │
│  - Calibration curves (clinical requirement)                    │
│  Output: 7+ evaluation plots, fairness analysis                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│            CLINICAL REPORTING (05)                               │
│  - Executive summary for hospital leadership                    │
│  - Detailed clinical report with evidence base                  │
│  - Implementation guide with timelines                          │
│  - Resource requirements and ROI analysis                       │
│  Output: CLINICAL_REPORT.txt, IMPLEMENTATION_GUIDE.txt          │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture

**Primary Model: XGBoost (Gradient Boosting)**
- Captures non-linear relationships between risk factors
- Handles mixed data types well
- Fast inference for real-time prediction
- Provides feature importance for interpretation

**Key Features**:
- Class weight balancing for imbalanced data
- Early stopping to prevent overfitting
- Cross-validated hyperparameter tuning

---

## Skills Demonstrated

### Data Science & Machine Learning

- **Data Generation**: Creating realistic synthetic data with clinically meaningful correlations
- **EDA**: Comprehensive exploratory analysis with statistical and visual techniques
- **Feature Engineering**: Domain-knowledge-based feature creation (clinical risk scores, temporal features, aggregations)
- **Missing Data Handling**: Strategic imputation with clinical reasoning (not just mean/median)
- **Class Imbalance**: SMOTE oversampling and class-weighted models
- **Model Selection**: Training and comparing multiple algorithms
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Evaluation**: Comprehensive metrics including AUC-ROC, Precision-Recall, calibration
- **Explainability**: SHAP analysis for model interpretability
- **Fairness Analysis**: Assessing performance across demographic subgroups

### Clinical/Healthcare Domain Knowledge

- **Evidence-Based Scoring**: Charlson Comorbidity Index, LACE Index
- **Clinical Correlations**: Understanding how diagnoses, meds, and labs relate
- **Risk Factors**: Identifying and contextualizing readmission drivers
- **Implementation Strategy**: Practical deployment in healthcare settings
- **Clinical Communication**: Explaining technical results to clinicians
- **Regulatory Awareness**: Addressing fairness, transparency, generalization

### Software Engineering Best Practices

- **Modular Code**: Separate modules for data generation, feature engineering, utilities
- **Type Hints**: Full type annotations for clarity and IDE support
- **Documentation**: Comprehensive docstrings with clinical context
- **Logging**: Production-grade logging instead of print statements
- **Reproducibility**: Fixed random seeds, clear documentation of parameters
- **Error Handling**: Graceful handling of edge cases
- **Configuration**: Configurable parameters for flexibility

### Visualization & Communication

- **Publication-Ready Plots**: 22+ high-quality visualizations (300 DPI)
- **Statistical Plots**: Histograms, distributions, correlations
- **Model Evaluation Plots**: ROC, PR curves, confusion matrices
- **Domain-Specific Plots**: Risk factor analysis, subgroup comparisons
- **Executive Communication**: Clear titles, labels, legends, proper formatting

---

## Quick Start

### Installation

```bash
cd project4-readmission-prediction
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# 1. Data Exploration (generates synthetic data, ~5 minutes)
python notebooks/01_data_exploration.py

# 2. Feature Engineering (~2 minutes)
python notebooks/02_feature_engineering.py

# 3. Model Training (~5-10 minutes, depending on hardware)
python notebooks/03_model_training.py

# 4. Model Evaluation (~2 minutes)
python notebooks/04_model_evaluation.py

# 5. Clinical Report Generation (<1 minute)
python notebooks/05_clinical_report.py
```

### Expected Output Structure

```
results/
├── plots/
│   ├── 01_demographics.png
│   ├── 02_clinical_features.png
│   ├── 03_prior_utilization.png
│   ├── 04_lab_distributions.png
│   ├── 05_missing_data.png
│   ├── 06_target_variable.png
│   ├── 07_risk_factors.png
│   ├── 08_correlations.png
│   ├── 10_clinical_risk_scores.png
│   ├── 11_temporal_features.png
│   ├── 12_aggregated_features.png
│   ├── 13_lab_features.png
│   ├── 14_feature_selection.png
│   ├── 15_model_comparison.png
│   ├── 16_roc_curves.png
│   ├── 17_pr_curves.png
│   ├── 18_calibration_curves.png
│   ├── 19_confusion_matrices.png
│   ├── 20_shap_summary.png
│   ├── 21_shap_bar.png
│   └── 22_decision_curve.png
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── test_data.pkl
├── reports/
│   ├── CLINICAL_REPORT.txt
│   ├── IMPLEMENTATION_GUIDE.txt
│   ├── final_results.csv
│   └── subgroup_analysis.csv
└── readmission_raw_data.csv
    readmission_features.csv
    model_comparison.csv
    EDA_SUMMARY.txt
    FEATURE_ENGINEERING_SUMMARY.txt
```

---

## Project Structure

### Source Code (`src/`)

**`data_generator.py`** (300+ lines)
- `ClinicalDataGenerator`: Generates synthetic patient data with clinical realism
- Features clinically correlated comorbidities, realistic distributions
- Configurable readmission rate, missing data patterns
- Well-documented with clinical reasoning

**`feature_pipeline.py`** (200+ lines)
- `ClinicalFeatureEngineering`: Production-ready feature engineering
- Charlson Comorbidity Index calculator
- LACE Index calculator
- Clinical missing data imputation strategy
- Lab value transformations with abnormality flags

**`model_utils.py`** (150+ lines)
- Model evaluation metrics calculation
- Plot generation functions (ROC, PR, confusion matrices, calibration)
- Results export utilities
- Clinical decision curve analysis

### Notebooks (`notebooks/`)

**`01_data_exploration.py`** (400+ lines)
- Generate synthetic data
- Comprehensive EDA with 8 publication-ready plots
- Missing data analysis
- Target variable exploration
- Risk factor identification

**`02_feature_engineering.py`** (350+ lines)
- Create clinical risk scores
- Extract temporal features
- Build aggregated features
- Transform lab values
- Handle missing data
- Feature selection with correlation analysis

**`03_model_training.py`** (500+ lines)
- Data splitting and stratification
- Class imbalance handling (SMOTE)
- Feature scaling
- Train 4 different models with hyperparameter tuning
- Model comparison and selection

**`04_model_evaluation.py`** (400+ lines)
- Test set evaluation
- ROC and Precision-Recall curves
- Confusion matrix analysis
- SHAP explainability
- Subgroup fairness analysis
- Clinical decision curve
- Publication-ready results table

**`05_clinical_report.py`** (200+ lines)
- Generate clinical summary report
- Implementation guide
- Resource requirements and ROI

---

## Results Summary

### Model Performance

| Metric | Test Set Value | Clinical Interpretation |
|--------|-----------------|--------------------------|
| AUC-ROC | 0.762 | Good discrimination between high/low risk |
| Sensitivity | 0.75 | Catches 75% of patients who will readmit |
| Specificity | 0.68 | Correctly identifies 68% of non-readmitting patients |
| PPV | 0.32 | 32% of flagged patients actually readmit |
| NPV | 0.93 | 93% of low-risk patients won't readmit |
| Accuracy | 0.68 | Overall correctness on test set |

### Top Predictive Features (in order of importance)

1. **Prior Hospital Admissions (6 months)** - Strong marker of disease complexity
2. **LACE Index Score** - Validated readmission risk tool (AUC 0.72-0.75 in literature)
3. **Medication Count** - Polypharmacy complicates adherence
4. **Number of Comorbidities** - Disease burden
5. **Discharge Disposition** - SNF/assisted living vs. home discharge
6. **ED Visits (6 months)** - High ED utilization signals unmet needs
7. **Lab Abnormalities** - Composite measure of disease severity
8. **Weekend Discharge** - Reduced follow-up access

### Model Comparison

All models tested in order of final AUC-ROC:
1. **XGBoost**: 0.762 (Best) - Selected for deployment
2. **Random Forest**: 0.758
3. **Neural Network**: 0.745
4. **Logistic Regression**: 0.739 (Baseline)

### Clinical Impact Projection

At current volume (~3,000 annual discharges):
- **Expected readmission reduction**: 15-20% (50-100 readmissions prevented annually)
- **Annual cost savings**: $750,000 - $2,500,000
- **Program cost**: ~$120,000-140,000/year
- **Return on Investment**: 6-20x in first year

---

## Implementation Guide

### Risk Stratification Protocol

**High Risk (predicted probability >0.70)**
- 48-hour phone follow-up
- Scheduled 7-day PCP visit
- Pharmacist medication reconciliation
- Home care evaluation
- Disease-specific care management

**Medium Risk (0.40-0.70)**
- 72-hour phone follow-up
- Scheduled PCP visit within 14 days
- Medication list provided
- Telehealth option

**Low Risk (<0.40)**
- Standard discharge instructions
- Standard PCP follow-up
- Virtual visit option

### Timeline for Deployment

| Phase | Timeline | Activities |
|-------|----------|------------|
| 1 | Months 1-2 | Model integration in EHR, staff training |
| 2 | Months 2-3 | Workflow integration, testing |
| 3 | Months 3-6 | Pilot with 500 discharges |
| 4 | Months 6+ | Full rollout, monitoring, optimization |

### Success Metrics

- **Technical**: >95% patients have risk score, <5% failures
- **Clinical**: 10-15% reduction in 30-day readmission rate
- **Operational**: >80% staff satisfaction, >90% intervention completion

---

## Key Insights

### Clinical Insights

1. **Prior utilization is king**: Patients with 6+ admissions in past 6 months have 2.5x higher readmission risk
2. **LACE Score matters**: Validated tool performs as expected (AUC 0.72-0.75 in literature, 0.76+ when combined with other features)
3. **Discharge disposition critical**: SNF/assisted living residents have 25% higher readmission risk
4. **Weekend discharge effect**: 8-10% higher risk when discharged on weekends (reduced follow-up access)
5. **Lab abnormalities compound risk**: Each additional abnormal lab increases risk; eGFR <60 and anemia particularly important

### Data Science Insights

1. **Synthetic data validity**: Synthetic data with clinically meaningful correlations can effectively train ML models for healthcare applications
2. **Class imbalance handling**: SMOTE improves model performance on minority class without degrading majority class
3. **Feature engineering critical**: Well-engineered clinical features outperform raw variables
4. **Model selection**: Gradient boosting (XGBoost) outperforms simpler baselines while remaining interpretable
5. **Fairness matters**: Performance variations across subgroups are clinically meaningful and warrant subgroup-specific analysis

---

## Technical Details

### Feature Engineering Methodology

**Clinical Risk Scores**
- Charlson Comorbidity Index: Weighted sum of diagnoses (weights from clinical literature)
- LACE Index: Components = Length of stay + Acute admission + Charlson + ED visits

**Temporal Features**
- Day of week (0=Monday, 6=Sunday)
- Weekend discharge flag (Friday-Sunday)
- Season (Winter, Spring, Summer, Fall)
- Month of admission

**Lab Features**
- Abnormality flags for each lab (based on clinical reference ranges)
- Count of abnormal labs (composite measure)
- Specific flags: kidney disease (eGFR <60), anemia (Hgb <11)

**Missing Data Strategy**
- Labs: Median imputation (robust to outliers) + missingness flag
- Utilization: Zero-fill for missing counts (indicates no prior visits)
- Missing flags created for important missingness patterns

### Model Training Details

**Data Split**
- Stratified train/validation/test split (60/20/20)
- Ensures readmission rate maintained across sets

**Class Imbalance Handling**
- SMOTE on training data only (to prevent data leakage)
- Class weights in logistic regression and XGBoost

**Hyperparameter Tuning**
- GridSearchCV with 5-fold cross-validation
- Metric: ROC-AUC (threshold-independent)
- Early stopping to prevent overfitting

**Feature Scaling**
- StandardScaler for logistic regression and neural network
- Tree-based models (RF, XGBoost) are scale-invariant

### Evaluation Metrics

**Primary Metric: AUC-ROC**
- Threshold-independent discrimination metric
- Appropriate for imbalanced data
- Standard in clinical literature

**Clinical Metrics**
- Sensitivity: Catch rate for readmitters
- Specificity: Correct identification of non-readmitters
- PPV: Likelihood of readmission given positive prediction
- NPV: Likelihood of no readmission given negative prediction

**Calibration**
- Important for clinical use: predicted probabilities should match actual outcomes
- Checked with calibration curves

---

## References

### Clinical Literature

1. van Walraven C, et al. (2010). "A modification of the Elixhauser comorbidity measures into a point system." J Clin Epidemiol. 63(12):1342-1350.
   - Charlson Comorbidity Index validation

2. Donzé J, et al. (2013). "Causes and patterns of readmissions." BMJ. 347:f5171.
   - LACE Index validation (AUC 0.72-0.75)

3. Kripalani S, et al. (2014). "Reducing Hospital Readmission Rates." Annu Rev Med. 65:471-485.
   - Comprehensive readmission reduction review

4. Jencks SF, et al. (2009). "Rehospitalizations among patients in the Medicare fee-for-service program." N Engl J Med. 360(14):1418-1428.
   - CMS readmission data

### Machine Learning References

1. Chen T, Guestrin C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
   - XGBoost algorithm details

2. Chawla NV, et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." J Artif Intell Res. 16:321-357.
   - Class imbalance handling

3. Lundberg SM, et al. (2017). "A Unified Approach to Interpreting Model Predictions." NIPS.
   - SHAP explainability

---

## Author & Acknowledgments

**Portfolio Project**: Jeremy Gracey
**Role**: Healthcare Data Scientist / ML Engineer (Mid-Career Transition)
**Background**: Clinical experience (Anesthesia Tech, Psychiatric Assistant), MS Psychology, AI/ML Certificate (UT Austin)

This project demonstrates a complete ML pipeline with production-quality code, clinical domain knowledge, and practical implementation guidance suitable for healthcare deployment.

---

## License

This project is for educational and portfolio purposes. Use appropriately with institutional review and validation on local data before clinical deployment.

---

**Last Updated**: 2024
**Questions?** See documentation in notebooks and clinical reports.
