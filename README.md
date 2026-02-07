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

## 🔎 Exploratory Data Analysis (EDA)

EDA focused on understanding **temporal behavior** and **sensor relationships** before model training.

### Key findings

#### 1. Temporal patterns
- Failure rates vary across **months** and **weekdays**.
- Failures peak during **high-activity operational periods** and drop toward **weekends**, suggesting workload influence rather than random degradation.

#### 2. Sensor behavior
- Certain sensor metrics show **distributional differences** between failure and non-failure events.
- Strong **multicollinearity** detected between:
  - `metric7` and `metric8`  
  → metric8 will be removed during modeling.

#### 3. Correlation structure
- Most sensors are weakly correlated.
- A few moderate relationships indicate **localized dependency**, not global redundancy.

---


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

## 🔁 Data Version Control (DVC)

- Dataset removed from Git tracking  
- Added to **DVC pipeline**  
- Enables:
  - reproducible experiments  
  - remote storage  
  - scalable collaboration  

Next step: **connect cloud remote storage**.

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

## 🏁 Status

✅ EDA completed  
✅ Dataset tracked with DVC  
🔄 Cloud remote setup in progress  
⏳ ML pipeline implementation pending  

---

## 📜 License

For educational and portfolio use.
