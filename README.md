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

🧠 Modeling Strategy (High-Level)

Because machine failures are extremely rare, standard accuracy-focused modeling is misleading in this case.

The modeling roadmap is therefore:
	1.	Baseline Logistic Regression
			- Establish interpretable reference performance
			- Reveal class-imbalance challenges
	2.	Imbalance-Aware Training
			- Improve recall for rare failures
			- Optimize meaningful metrics (ROC-AUC, PR-AUC, Recall)
	3.	Gradient Boosting Models (LightGBM / XGBoost)
			- Capture nonlinear sensor interactions
			- Improve predictive discrimination
	4.	Probability Calibration
			- Convert raw scores into true failure risk probabilities
		 	- Enable real-world decision thresholds


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

## 🚀 Upcoming Pipeline Stages

The following production stages will be implemented:

1. Data ingestion  
2. Data preprocessing  
3. Feature engineering  
4. Model training  
5. Evaluation  
6. Experiment tracking (MLflow)  
7. Deployment-ready inference pipeline  

---

## 🏁 Current Status

✅ EDA completed  
✅ Dataset tracked with DVC  
✅ Local DVC Remote Configured
✅ Baseline training pipeline implemented 
🔄 Imbalance-aware modeling in progress
⏳ Gradient boosting + calibration pending
⏳ Full MLOps orchestration pending
---

## 📜 License

For educational and portfolio use.
