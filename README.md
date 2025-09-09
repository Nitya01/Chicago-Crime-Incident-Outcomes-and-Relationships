# IS 597 – Machine Learning in the Cloud (MLC) Final Project

**Title:** Exploring Crime Incident Outcomes and Relationships through Machine Learning and Network Analysis  
**Course:** IS 597 – MLC  


## Project Overview
This project analyzes Chicago crime patterns using **three complementary approaches**:

1. **Arrest Prediction (Supervised Classification)**  
   Predict whether an incident leads to an arrest using Random Forest, XGBoost, and MLP.

2. **Sequential Theft Forecasting (RNN/LSTM)**  
   Forecast theft in the *next hour* from the prior 12 hours of crime activity.

3. **Crime Graph Analysis (Community Detection)**  
   Build a spatial–temporal crime network and detect communities of related incidents.

Together, these address **event outcomes**, **temporal risk**, and **relational structure**.

---

## 📂 Repository Structure

```text
IS-597-MLC-Project/
├─ Problem_Statement1_Arrest Prediction_Code/    # ML pipelines for arrest prediction
├─ Problem_Statement2_RNN_Model_Code/            # Sequential forecasting (RNN/LSTM)
└─ Problem_Statement3_Graph_Analysis_Code/       # Graph-based analysis & community detection
```

## Data & Environment

- **Dataset:** Chicago Crimes (filtered; ~2015–2025) stored on **AWS S3**  
- **Compute:** AWS SageMaker (`ml.m5.large` for ML; `ml.t2.large` for graph analysis)  
- **Core Libraries:**  
  - `scikit-learn`, `xgboost`, `pandas`, `numpy`  
  - `tensorflow`/`keras` (RNN/LSTM)  
  - `networkx`, `igraph`, `leidenalg`, `haversine`, `pyg` (for graph/community analysis)

> Update S3 paths in each folder’s scripts before running.

## 📊 Key Results

| Problem Statement     | Best Model / Approach   | Score(s)                              | Highlights |
|-----------------------|-------------------------|---------------------------------------|------------|
| Arrest Prediction     | **XGBoost**             | Accuracy: 80.47% • ROC-AUC: 0.8985    | Strong discriminative ability & calibrated probabilities |
| Theft Forecasting     | **RNN (LSTM)**          | F1: ~0.75 • AUC: ~0.75                | Theft risk peaks 10 AM–12 PM; clear temporal patterns |
| Graph Analysis        | **Community Detection** | Modularity: 0.710                     | Spatial–temporal clusters of related crimes uncovered |

## ⚙️ Methodology at a Glance

- **Preprocessing:**  
  Cleaning, datetime feature expansion (Hour/Weekday/Month/Year), category consolidation, correlation checks.

- **Imbalance Handling (Arrest Prediction):**  
  Heavy imbalance (~17.6% arrests) → **downsampling** (balanced classes).  
  SMOTE was tested but introduced null artifacts.

- **Modeling:**
  - RF/XGBoost/MLP on SageMaker (scripted training, `argparse` hyperparams).
  - RNN/LSTM with sliding 12-hour windows; class weighting for rare events.
  - Graph construction with Haversine + BallTree (500 m, 1-hour edge rule); Greedy Modularity for communities.

- **Dimensionality Reduction (PCA):**
  - **Tree models:** slight performance drop → not used in finals.
  - **MLP:** negligible effect; no final benefit.

- **Monitoring:**  
  SageMaker/CloudWatch tracked CPU/memory/IO; XGBoost showed high CPU utilization consistent with parallelization.

## 🔮 Future Work

- Enrich with **demographic/geospatial/socioeconomic** covariates  
- Scale **hyperparameter optimization** (efficient HPO in SageMaker)  
- Extend graph methods with **GNNs** (PyG/DGL) for dynamic link prediction  
- Deploy **real-time inference** for proactive resource allocation  

## 📚 References

- AWS SageMaker Docs – Algorithms & Training  
- XGBoost on SageMaker – Example scripts and patterns  
- DGL on SageMaker – Graph learning workflows  
- PCA & tree models – mixed evidence on benefits for tabular/tree pipelines  
- LSTM foundations for sequence modeling  
