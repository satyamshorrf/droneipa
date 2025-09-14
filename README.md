

# 🚀 Drone Flight Anomaly Detection – Datathon 2025



---

## 📌 Introduction
This project was developed for **Terrier Cyber Quest Datathon 2025**, where the challenge is focused on **predictive threat intelligence and anomaly detection**.  
Our problem statement:  
**Detect anomalies in drone flight telemetry data (altitude, velocity, yaw, pitch, battery, and GPS drift) to prevent mission failures or hijacks.**

---

## 🎯 Objectives
- Build a **predictive anomaly detection model** using drone telemetry data.  
- Identify **malfunction patterns**, **signal jamming**, and **unauthorized diversions**.  
- Provide **real-time anomaly alerts** for defense/security operations.  

---



## 📂 Dataset  

For our **Drone Flight Anomaly Detection** theme, we worked with a **simulated telemetry dataset** inspired by real-world drone logs.  

📌 **Features used (time-series):**  
- `time` – Timestamp of drone flight  
- `altitude` – Drone height variation  
- `velocity` – Forward motion speed  
- `yaw` – Angular rotation (left/right drift)  
- `pitch` – Forward/backward tilt  
- `battery` – Remaining energy %  
- `gps_drift` – Positional error (used to simulate GPS jamming)  

📁 **Files Generated/Used:**  
- `drone_train.csv` – Training data (normal + simulated anomalies)  
- `drone_test.csv` – Test dataset for evaluation  
- `anomaly_output.csv` – Final results with anomaly flags  

---

## 🛠️ Methodology

### 1. Data Preprocessing
- Loaded datasets in **Google Colab**.  
- Handled missing values and normalized features using **MinMaxScaler**.  
- Converted time-series logs into structured format for ML models.  

### 2. Models Used
- **Isolation Forest** → Detects anomalies by splitting feature space.  
- **AutoEncoder (Deep Learning)** → Learns reconstruction patterns of normal flight.  
- **Rule-based Checks** → Flag sudden drops in altitude, excessive GPS drift, or unusual yaw/pitch values.  

### 3. Ensemble Strategy
We combined results from all three methods to form a **final anomaly score**:
- If multiple methods flagged a data point → High-confidence anomaly.  
- Ensures balance between **false positives** and **missed anomalies**.  

---

## 📊 Results & Visualizations

### Altitude Anomalies
- Red markers → Isolation Forest  
- Green markers → AutoEncoder  
- Orange markers → Rule-based anomalies  

### GPS Drift Anomalies
- Similar visualization applied to GPS drift readings.  

📌 All visualizations were built using **Plotly** and saved as PNGs (`altitude_anomalies.png`, `gps_drift_anomalies.png`).  

---

## ✅ Evaluation
- Compared detected anomalies with **ground truth labels** from test dataset.  
- Metrics: **Precision, Recall, F1-score**.  
- Ensemble improved detection robustness compared to individual models.  

---

## 🖥️ Tech Stack
- **Google Colab** – Development & execution  
- **Python** – Core programming  
- **Libraries**: pandas, numpy, scikit-learn, tensorflow/keras, plotly  
- **Visualization**: Interactive anomaly plots  

---
----

## 📌 How to Run  

1. **Open the Colab Notebook** in your browser (Google Colab).  

2. **Clone / Create dataset (simulated telemetry):**  
   ```python
   import pandas as pd
   import numpy as np

   def create_simulated_data(n=600, inject=True):
       t = np.arange(n)
       altitude = 100 + np.cumsum(np.random.normal(0, 0.4, n))
       velocity = 10 + np.random.normal(0, 0.3, n)
       yaw = np.cumsum(np.random.normal(0, 0.2, n))
       pitch = np.random.normal(0, 0.1, n)
       battery = np.linspace(100, 20, n) + np.random.normal(0, 1, n)
       gps_drift = np.random.normal(0, 0.05, n)

       if inject:
           altitude[120:125] += 30      # sudden climb
           velocity[300:305] += 8       # speed burst
           battery[500:505] -= 35       # abnormal drain
           gps_drift[400:405] += 0.5    # GPS jamming

       return pd.DataFrame({
           "time": t,
           "altitude": altitude,
           "velocity": velocity,
           "yaw": yaw,
           "pitch": pitch,
           "battery": battery,
           "gps_drift": gps_drift
       })

   df = create_simulated_data()
   df.to_csv("drone_train.csv", index=False)
   print("✅ Drone telemetry dataset created: drone_train.csv")


3. **Run preprocessing + anomaly detection cells.**

   * Standardization of features.
   * Train Isolation Forest + Autoencoder.
   * Save results → `anomaly_output.csv`.

4. **Visualize results.**
   Interactive Plotly graphs will highlight anomalies in **altitude** and **GPS drift** with color markers.

5. **(Optional)** Export plots as images (if `kaleido` is installed).


3. Install dependencies:

   ```python
   !pip install scikit-learn tensorflow plotly kaleido
   ```
4. Run preprocessing, anomaly detection, and visualization cells.
5. Check results in `/content/results/`.


## 🎯 Impact

* Helps **defense & security teams** monitor drone fleets in real-time.
* Reduces risk of **mid-air mission failure** or **hijacking**.
* Scalable to large fleets with streaming telemetry data.

---

## 👥 Team & Contribution  

**🛡️ Team Name:** `AeroDefender`  

📌 **Member Details:**  

| 👤 Name          | 🎯 Role & Responsibility                                | 🛠️ Tools & Tech Used                                  |
|------------------|---------------------------------------------------------|-------------------------------------------------------|
| **Satyam Kumar** | Lead Developer – ML, Visualization & System Design      | Python, Google Colab, Scikit-learn, TensorFlow, Plotly, React (Frontend Dashboard Idea) |

---

## 🏆 Conclusion

This solution demonstrates how **data-driven anomaly detection** can safeguard national defense operations by:

* Detecting unusual drone behaviors early,
* Ensuring mission reliability,
* And providing a scalable framework for real-world deployment.



  

