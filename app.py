import numpy as np
import pandas as pd
import joblib
import sqlite3
import io

from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file
from numpy_model import predict_rul

app = Flask(__name__)


def init_db():
    conn = sqlite3.connect("prediction_history.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction REAL,
            status TEXT,
            datetime TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


preprocessing = joblib.load("rul_preprocessing.pkl")

scaler = preprocessing["scaler"]
selected_sensors = preprocessing["selected_sensors"]
sequence_length = preprocessing["sequence_length"]
rul_cap = preprocessing["rul_cap"]

print("NumPy LSTM model loaded successfully!")
print("Preprocessing objects loaded successfully!")


@app.route("/")
def home():

    conn = sqlite3.connect("prediction_history.db")
    conn.row_factory = sqlite3.Row

    history = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "index.html",
        history=history
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():

    conn = sqlite3.connect("prediction_history.db")

    conn.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    return redirect("/")


@app.route("/download_history")
def download_history():

    conn = sqlite3.connect("prediction_history.db")

    history = pd.read_sql_query(
        "SELECT prediction, status, datetime FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    csv_file = io.BytesIO()

    history.to_csv(
        csv_file,
        index=False,
        encoding="utf-8"
    )

    csv_file.seek(0)

    return send_file(
        csv_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="RUL_Prediction_History.csv"
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]

    if file.filename == "":
        return "Please select a CSV file!"

    if not file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file!"

    try:

        data = pd.read_csv(file)

        print("CSV file uploaded successfully!")
        print("Dataset shape:", data.shape)

        missing_columns = [
            col for col in selected_sensors
            if col not in data.columns
        ]

        if missing_columns:
            return f"Missing sensor columns: {missing_columns}"

        sensor_data = data[selected_sensors].tail(sequence_length)

        if len(sensor_data) < sequence_length:
            return f"Please upload at least {sequence_length} rows of sensor data."

        sensor_data = sensor_data.apply(
            pd.to_numeric, errors="coerce"
        )

        if sensor_data.isnull().values.any():
            return "The CSV contains missing or invalid sensor values."

        scaled_data = scaler.transform(sensor_data)

        scaled_data = scaled_data.astype(np.float32)

        prediction = predict_rul(scaled_data)

        prediction = np.clip(prediction, 0, rul_cap)

        prediction = round(float(prediction), 2)

        if prediction > 75:
            status = "High Remaining Life"

        elif prediction > 30:
            status = "Moderate Remaining Life"

        else:
            status = "Low Remaining Life"

        print("Predicted RUL:", prediction)

        conn = sqlite3.connect("prediction_history.db")

        conn.execute(
            "INSERT INTO predictions (prediction, status, datetime) VALUES (?, ?, ?)",
            (
                prediction,
                status,
                datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
            )
        )

        conn.commit()
        conn.close()

        conn = sqlite3.connect("prediction_history.db")
        conn.row_factory = sqlite3.Row

        history = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()

        conn.close()

        return render_template(
            "index.html",
            prediction=prediction,
            status=status,
            history=history
        )

    except Exception as e:

        print("Prediction error:", str(e))

        return f"Error processing file: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)