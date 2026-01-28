
# Digit-Predictor

A **handwritten digit classifier** built with **TensorFlow** using a simple three-layer neural network trained on the **MNIST dataset** (from TensorFlow). The model takes grayscale 28×28 images of handwritten digits (0–9) and predicts the digit class.

---

## 📌 Overview

This project implements a neural network with the following architecture:

* **Input Layer:** Flattened 28×28 pixel images (784 inputs).
* **Hidden Layer 1:** Fully connected **Dense** layer with **64 units** and **ReLU** activation.
* **Hidden Layer 2:** Fully connected **Dense** layer with **32 units** and **ReLU** activation.
* **Output Layer:** Fully connected **Dense** layer with **10 units** (for digits 0–9) and **softmax** activation.

This feed-forward neural network is trained to classify handwritten digits from the MNIST dataset, which contains 60,000 training images and 10,000 testing images of digits labeled 0 through 9. ([GitHub][1])

---

## 🧠 Architecture

```
Input (28×28 image → flattened)
        ↓
Dense (64 units, ReLU)
        ↓
Dense (32 units, ReLU)
        ↓
Dense (10 units, Softmax)
```

---

## 🛠️ Setup

### 📦 Requirements

Install Python (3.7+ recommended) and required packages:

```bash
pip install tensorflow numpy matplotlib
```

*Note:* You may choose to add a `requirements.txt` with these dependencies.

---

## 📥 Data

MNIST is loaded directly using TensorFlow datasets or Keras API:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Before training, **normalize** pixel values to the range [0, 1] and one-hot encode labels:

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

---

## 🧩 Model Building (Example)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 📈 Training

Train the model using:

```python
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

Adjust epochs, batch size, and validation split as needed. Evaluate performance after training:

```python
model.evaluate(x_test, y_test)
```

---

## 🎯 Prediction

Predict on new images:

```python
predictions = model.predict(x_test)
predicted_digit = predictions[i].argmax()
```

The model outputs a probability distribution over 10 classes and selects the most likely class as the predicted digit.

---

## 📌 Results (Optional)

You can include a summary of performance metrics such as:

```
Test accuracy: 97–99% on MNIST test set (typical for simple MLP). :contentReference[oaicite:2]{index=2}
```

---

## 📁 File Structure

```
.
├── Digits.ipynb                 # Notebook with training & prediction code
├── README.md                   # Project doc (this file)

```

---

## 🚀 How to Use

1. Clone the repo:

   ```bash
   git clone https://github.com/YohanJaya/Digit-Predictor.git
   cd Digit-Predictor
   ```



2. Run the training/prediction notebook or script.

---




