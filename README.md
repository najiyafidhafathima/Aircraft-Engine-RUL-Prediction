# ✈️ Aircraft Engine Remaining Useful Life (RUL) Prediction Using LSTM

An end-to-end Deep Learning project that predicts the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS FD001 dataset. The trained LSTM model is deployed through a Flask web application.

## 🚀 Live Application

**Try the application here:**

[Aircraft Engine RUL Prediction – Live Demo](https://aircraft-engine-rul-prediction-ceis.onrender.com)

## 📌 Project Overview

Aircraft engines generate sensor readings throughout their operational life. As engine components degrade, their performance changes, eventually leading to failure.

Predictive maintenance helps estimate the remaining operational life of an engine so that maintenance can be planned before failure occurs.

This project uses a Long Short-Term Memory (LSTM) neural network to analyse sequential engine sensor measurements and predict the remaining operating cycles before engine failure.

## 🎯 Project Objectives

- Analyse aircraft engine sensor data.
- Perform Exploratory Data Analysis (EDA).
- Identify relevant sensor features.
- Calculate Remaining Useful Life (RUL).
- Develop an LSTM-based time-series regression model.
- Evaluate model performance using regression metrics.
- Deploy the model through a Flask web application.

## 📂 Dataset Description

**Dataset:** NASA C-MAPSS  
**Subset:** FD001

| Description | Details |
|---|---|
| Training engines | 100 |
| Training observations | 20,631 |
| Test engines | 100 |
| Test observations | 13,096 |
| Original sensor measurements | 21 |
| Selected sensor features | 14 |

The dataset contains engine operational settings and sensor measurements collected over multiple operating cycles.

## ⚙️ Project Workflow

1. Dataset loading and exploration
2. Data preprocessing
3. Exploratory Data Analysis
4. RUL calculation and target capping
5. Sensor feature selection
6. Feature scaling
7. Time-series sequence generation
8. LSTM model development
9. Model training and validation
10. Model evaluation
11. Flask application development
12. Model deployment

## 🧠 Deep Learning Model Architecture

The project uses an LSTM (Long Short-Term Memory) neural network to learn temporal patterns from engine sensor data.

**Model Architecture:**

- Input Layer: 30 time steps × 14 sensor features
- LSTM Layer: 64 units
- Dropout Layer: 0.2
- Dense Layer: 32 neurons with ReLU activation
- Output Layer: 1 neuron for RUL regression

**Optimizer:** Adam  
**Loss Function:** Mean Squared Error (MSE)  
**Evaluation Metric:** Mean Absolute Error (MAE)

The model was developed and trained using TensorFlow/Keras. For lightweight deployment, its trained weights were converted into a NumPy-based inference implementation.

## 📊 Model Performance

### Validation Results

| Metric | Score |
|---|---:|
| MAE | 8.70 cycles |
| MSE | 138.14 |
| RMSE | 11.75 cycles |
| R² Score | 0.921 |

### Test Results

| Metric | Score |
|---|---:|
| MAE | 10.57 cycles |
| MSE | 202.49 |
| RMSE | 14.23 cycles |
| R² Score | 0.883 |

The model achieved an R² score of approximately 0.883 on the test dataset.

## 🌐 Flask Web Application
### Application Screenshot

### Application Screenshot

![Aircraft Engine RUL Prediction Application](<Screenshot 2026-11-25 065723.png>)

The trained model is integrated into a Flask web application.

### Key Features

- Upload engine sensor data through CSV files.
- Predict Remaining Useful Life in operating cycles.
- Display engine condition based on predicted RUL.
- Maintain prediction history using SQLite.
- Download prediction history as a CSV file.
- Clear saved prediction records.

### Deployment

The application is deployed on Render.

The deployment uses a lightweight NumPy-based LSTM inference implementation to reduce memory and computational requirements.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras, LSTM |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Database | SQLite |
| Deployment | Render |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
Aircraft-Engine-RUL-Prediction/
│
├── EDA_Plots/
├── Model_Evaluation_Plots/
├── templates/
│
├── Aircraft_Engine_RUL_Prediction.ipynb
├── app.py
├── numpy_model.py
├── rul_lstm_weights.npz
├── rul_preprocessing.pkl
├── requirements.txt
├── .gitignore
├── .python-version
└── README.md
```

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/najiyafidhafathima/Aircraft-Engine-RUL-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Aircraft-Engine-RUL-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open the application

Visit:

```text
http://127.0.0.1:5000
```

## ⚠️ Limitations

- The project uses only the FD001 subset of NASA C-MAPSS.
- The dataset is based on simulated engine degradation data.
- The model is designed for a single operating condition and fault mode.
- The predictions are experimental estimates and are not certified for real aircraft maintenance decisions.

## 🔮 Future Improvements

- Evaluate the model on additional C-MAPSS subsets.
- Explore advanced deep learning architectures.
- Incorporate real-world engine sensor data.
- Improve model generalization.
- Develop more advanced predictive maintenance capabilities.

## 🙏 Acknowledgement

Dataset source: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation).

## 👩‍💻 Author

**Najiya Fidha Fathima**

[GitHub Profile](https://github.com/najiyafidhafathima)

[Project Repository](https://github.com/najiyafidhafathima/Aircraft-Engine-RUL-Prediction)

[Live Application](https://aircraft-engine-rul-prediction-ceis.onrender.com)
