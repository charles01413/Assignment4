# Assignment4
# 🧠 Multidisciplinary AI Project Suite

This repository contains three focused machine learning and deep learning tasks, each demonstrating essential techniques across classical ML, convolutional neural networks, and natural language processing. Each task is implemented in Python using industry-standard libraries with annotated code and practical evaluation.

---

## 📘 Task 1: Iris Classification (Classical ML)

**🔹 Objective:**  
Train a Decision Tree classifier to predict iris species.

**📊 Dataset:**  
[Iris Species Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

**📌 Workflow:**
- Preprocess the feature data (handle missing values)
- Encode target labels
- Train and evaluate a decision tree model
- Report accuracy, precision, and recall

**🛠 Tools:**  
`Scikit-learn`, `Pandas`, `NumPy`

**📁 Deliverables:**  
- Jupyter Notebook or `.py` script with inline comments
- Printed evaluation metrics for model performance

---

## 🧮 Task 2: MNIST Digit Recognition (Deep Learning)

**🔹 Objective:**  
Classify handwritten digits using a CNN and achieve ≥95% accuracy.

**🧾 Dataset:**  
[MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

**📌 Workflow:**
- Build a convolutional neural network (CNN) with PyTorch
- Train over multiple epochs using the Adam optimizer
- Evaluate test accuracy
- Visualize predictions on 5 sample images

**🛠 Tools:**  
`PyTorch`, `Torchvision`, `Matplotlib`

**📁 Deliverables:**  
- Python code with model architecture, training loop, and evaluation
- Plots showing sample images and predicted labels

---

## 💬 Task 3: Amazon Reviews – NER & Sentiment Analysis (NLP)

**🔹 Objective:**  
Use NLP to extract named entities and analyze sentiment from user reviews.

**📋 Text Source:**  
Sample reviews from Amazon product listings

**📌 Workflow:**
- Apply spaCy’s NER to identify product names and brand entities
- Use TextBlob to assign sentiment polarity labels
- Display named entities and sentiment summaries

**🛠 Tools:**  
`spaCy`, `TextBlob`, `re` (optional for rule tuning)

**📁 Deliverables:**  
- Python script or notebook output showing named entities and sentiment
- Sample text annotations with labels

---

## 🧭 Optional Extension: Ethics & Debugging Challenge

**Ethical Consideration:**  
- Explore fairness in the MNIST model and bias in Amazon review interpretation  
- Use tools like TensorFlow Fairness Indicators or custom rule-based augmentations with spaCy

**Bug Fixes Task:**  
- Debug a provided TensorFlow script containing common issues like input shape errors or incorrect loss function

---

## 🛠 Setup Instructions

```bash
# Clone the repo


# Install dependencies
pip install -r requirements.txt
