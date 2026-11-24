# Aircraft Engine Remaining Useful Life (RUL) Prediction Using LSTM

## Project Overview
This project develops a deep learning-based predictive maintenance system to estimate the Remaining Useful Life (RUL) of aircraft engines using sensor data from the NASA C-MAPSS FD001 dataset.

An LSTM (Long Short-Term Memory) neural network is used to learn temporal patterns in engine sensor measurements and predict the remaining operating cycles before engine failure.

## Objectives
- Analyse aircraft engine sensor data.
- Perform exploratory data analysis (EDA).
- Identify relevant sensor features.
- Develop an LSTM-based RUL prediction model.
- Deploy the trained model using Flask.

## Dataset
- **Dataset:** NASA C-MAPSS
- **Subset:** FD001
- **Training engines:** 100
- **Training observations:** 20,631
- **Test engines:** 100
- **Test observations:** 13,096
- **Input features:** 14 selected sensor measurements.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- SQLite

## Model Architecture
- Input Layer: 30 time steps × 14 sensor features
- LSTM Layer: 64 units
- Dropout: 0.2
- Dense Layer: 32 neurons
- Output Layer: 1 neuron

## Model Performance

| Metric | Test Result |
|---|---:|
| MAE | 10.57 cycles |
| MSE | 202.49 |
| RMSE | 14.23 cycles |
| R² Score | 0.883 |

## Flask Web Application
The Flask application allows users to:
- Upload engine sensor data through CSV files.
- Generate RUL predictions.
- View engine status based on predicted RUL.
- Store prediction history.
- Download prediction history as a CSV file.
- Clear saved prediction records.

## Project Structure

```text
Aircraft-Engine-RUL-Prediction/
│
├── app.py
├── Aircraft_Engine_RUL_Prediction.ipynb
├── rul_lstm_model.keras
├── rul_preprocessing.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
├── EDA_Plots/
└── Model_Evaluation_Plots/
```

## How to Run

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

4. Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Limitations
- Uses only the FD001 subset of the NASA C-MAPSS dataset.
- Based on simulated engine data.
- Designed for a single operating condition and fault mode.

## Future Improvements
- Evaluate the model on additional C-MAPSS subsets.
- Explore advanced deep learning architectures.
- Incorporate real-world engine sensor data.
- Develop real-time predictive maintenance capabilities.

## Acknowledgement
Dataset source: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation).