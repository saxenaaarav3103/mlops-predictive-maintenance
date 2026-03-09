# MLOps Predictive Maintenance

This project builds an end-to-end **machine failure prediction pipeline** using real-world sensor data, following **MLOps best practices** such as data versioning, reproducible pipelines, and experiment tracking.

---

## 📌 Objective

The goal is to predict **machine failure events** using historical sensor readings and temporal patterns, while structuring the workflow in a way that reflects **real production ML systems**, not just a notebook-level ML experiment.

---

## 📊 Dataset

- Source: Predictive maintenance sensor dataset  
- Contains:
  - Date of observation  
  - Device identifier  
  - Binary failure label  
  - Multiple sensor metrics (`metric1` → `metric9`)  

The dataset is tracked using **DVC** instead of Git to ensure:
- reproducibility  
- scalability  
- proper data version control  

---
### 📂 Project Structure

```text
mlops-predictive-maintenance/
├── data/
│   └── raw/                # Dataset tracked with DVC
├── notebooks/
│   └── eda.ipynb           # Exploratory analysis
├── .dvc/                   # DVC metadata
├── .dvcignore
├── .gitignore
└── README.md
```

---

## 🔎 Exploratory Data Analysis (EDA)

EDA focused on understanding **temporal behavior** and **sensor relationships** before model training.

### Key findings

#### 1. Temporal patterns
- Failure frequency varies across **months** and **weekdays**.
- Peaks align with **high operational workload periods** and drop toward **weekends**, suggesting workload influence rather than random degradation.

#### 2. Sensor behavior
- Certain sensor metrics show **distributional differences** between failure events and normal operation.
- Strong **multicollinearity** detected between:
  - `metric7` and `metric8`  
  → To avoid:
    1. redundant information
    2. unstable model coefficients
    3. inflated variance
    
    metric8 will be removed during modeling.

#### 3. Correlation structure
- Most sensors are exhibit weak pairwise correlation.
- A few moderate relationships indicate **localized dependency**, not global redundancy.


---

## 🧠 Modeling Strategy (High-Level)

Because machine failures are extremely rare, standard accuracy-focused modeling is misleading. Our roadmap focuses on handling class imbalance and non-linear interactions:

1. **Baseline Logistic Regression**
   - Establish interpretable reference performance.
   - Reveal class-imbalance challenges.

2. **Imbalance-Aware Training**
   - Improve recall for rare failures.
   - Optimize meaningful metrics like ROC-AUC, PR-AUC, and Recall.

3. **Gradient Boosting Models (LightGBM / XGBoost)**
   - Capture nonlinear sensor interactions.
   - Improve predictive discrimination.

4. **Probability Calibration**
   - Convert raw scores into true failure risk probabilities.
   - Enable real-world decision thresholds.


---

## 🔁 Data Version Control (DVC)

- Dataset removed from Git tracking  
- Dataset tracked using DVC
- Local DVC remote storage configured
- This helps with:
    1. Experiment Reproducibility
    2. clean Git History
    3. Production-Style Data Management  

---

## Modeling Evolution

1) Initial Baselines and Their Limits

   The project started with three scripts:

  * src/train_logreg.py (Logistic Regression)
  * src/train_xgb.py (XGBoost)
  * src/train_hybrid.py (Isolation Forest + XGBoost)

These models used the original 1 day target and random stratified train-test evaluation. Despite imbalance handling, results were not operationally useful due to the following reasons:

  * very unstable precision-recall tradeoff
  * high false-alert behavior for recall-focused thresholds
  * weak future-facing reliability when evaluated more strictly

In short, exact same day failure prediction with the raw setup was too hard for this dataset.


2) Problem Reframing Window-Based Failure Target

    To address these issues, the pipeline was redesigned in src/train_window_target.py:

    * target changed from “failure today” to “failure within next 14 days”
      
  This pivot in direction increased positive examples due to increased prediction window and made the task closer to real maintenance planning (aiming for early warning instead of exact day guessing).


3) Temporal Signal Engineering
   
   Device-level temporal features were added to capture degradation dynamics:

    * lag features (lag1, lag3)
    * delta/change features
    * rolling means/std over multiple windows (3, 7, 14)
    * relative change and calendar context (day, month, dayofweek)

This moved the model away from one time snapshots and made it learn how each machine’s behavior changes over time.
     
4) Evaluation Hardening (More Realistic, Less Inflated)
   
   The old random split approach was replaced with a chronological split:

    * train = older period
    * validation = middle period
    * test = latest period

Threshold selection is now performed on validation and then fixed for final test evaluation.
This removed optimistic bias from tuning directly on test predictions and gave a more honest view of generalization.

5) Current Outcome

   Compared with earlier scripts, train_window_target.py provides a more credible and practical framework:

    * better aligned target definition
    * stronger temporal feature representation
    * production-like time-aware validation logic
    
   Even where metrics remain challenging, results are now reliable enough for meaningful iteration, unlike earlier pipelines that did not produce suitable results


---


## Current Status

✅ EDA completed  
✅ Dataset tracked with DVC  
✅ Local DVC remote configured  
✅ Baseline models implemented (`train_logreg.py`, `train_xgb.py`, `train_hybrid.py`)  
✅ Window-based target modeling implemented (`train_window_target.py`, currently 14-day window)  
✅ Time-aware train/validation/test split implemented  
✅ Validation-based threshold tuning implemented (test set kept for final evaluation)  
✅ Temporal feature engineering added (lags, deltas, rolling stats)  
🔄 Precision–recall improvement experiments ongoing (SMOTE vs class-weight setups)  
⏳ PR-AUC and recall@top-k operational metrics pending in training logs  
⏳ MLflow experiment tracking integration pending  


---

## 📜 License

For educational and portfolio use.
