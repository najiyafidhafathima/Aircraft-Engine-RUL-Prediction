import numpy as np

weights = np.load("rul_lstm_weights.npz")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_rul(sequence):

    kernel = weights["lstm_kernel"]
    recurrent_kernel = weights["lstm_recurrent_kernel"]
    bias = weights["lstm_bias"]

    dense_kernel = weights["dense_kernel"]
    dense_bias = weights["dense_bias"]

    output_kernel = weights["output_kernel"]
    output_bias = weights["output_bias"]

    h = np.zeros(64, dtype=np.float32)
    c = np.zeros(64, dtype=np.float32)

    for x in sequence:

        z = np.dot(x, kernel) + np.dot(h, recurrent_kernel) + bias

        i, f, candidate, o = np.split(z, 4)

        i = sigmoid(i)
        f = sigmoid(f)
        candidate = np.tanh(candidate)
        o = sigmoid(o)

        c = f * c + i * candidate
        h = o * np.tanh(c)

    dense_output = np.maximum(
        np.dot(h, dense_kernel) + dense_bias, 0
    )

    prediction = (
        np.dot(dense_output, output_kernel) + output_bias
    )

    return float(prediction[0])