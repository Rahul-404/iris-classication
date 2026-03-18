# 🌸 Autonomous Flower Classification System

### *Production-Grade ML System with Probabilistic Decision Intelligence*

> **An end-to-end machine learning system designed to power autonomous bots for flower classification using probabilistic inference — enabling smarter, uncertainty-aware decisions instead of naive predictions.**

---

## 📌 Problem Statement

Traditional classification systems output a **single hard label**, which is often insufficient in real-world environments where **uncertainty matters**.

### 🚨 Real-World Scenario (Hypothetical but Practical)

Autonomous bots are deployed in a botanical environment to:

* Identify flower species
* Assist in sorting, tagging, and monitoring
* Operate under **uncertain and noisy conditions**

A wrong classification can lead to:

* Incorrect labeling
* Faulty downstream decisions
* Reduced system reliability

---

## 💡 Solution

Instead of returning just a class label, this system provides:

✅ **Class probability distribution**
✅ **Top-K predictions**
✅ **Confidence-aware decision making**

This allows bots to:

* Act only when confidence is high
* Escalate uncertain cases
* Improve reliability in production environments

---

## 🧠 Core Idea

> **“Don’t just predict — quantify confidence.”**

We transform a simple ML model into a **decision-support system**.

---

## 🏗️ System Architecture

```text
User / Bot Request
        ↓
API Gateway
        ↓
Inference Service (FastAPI / Flask)
        ↓
ML Model (Iris Classifier)
        ↓
Probability Engine
        ↓
Response (Top-K + Confidence Scores)
```

---

## ⚙️ Tech Stack

### 🔹 Machine Learning

* Scikit-learn (Baseline model)
* Probability calibration (Softmax / Predict_proba)

### 🔹 Backend

* FastAPI / Flask (Inference API)
* RESTful architecture

### 🔹 Deployment

* Docker (Containerization)
* AWS ECR (Image registry)
* AWS Lambda / EC2 (Serving layer)
* API Gateway (Routing)

### 🔹 CI/CD

* GitHub Actions
* Automated build & push to ECR
* Deployment pipeline

### 🔹 Monitoring & Observability

* Logging (Structured JSON logs)
* Prometheus + Grafana (metrics)
* Loki (log aggregation)

---

## 🔍 Features

### 🧾 Probabilistic Predictions

* Returns full probability distribution across classes
* Example:

```json
{
  "setosa": 0.02,
  "versicolor": 0.85,
  "virginica": 0.13
}
```

---

### 🧠 Top-K Decision Support

* Instead of 1 label:

```json
Top-2 Predictions:
1. versicolor (85%)
2. virginica (13%)
```

---

### ⚠️ Confidence-Aware Logic

* High confidence → Auto decision
* Low confidence → Flag for review

---

### 📊 Production Logging

* Request logs
* Prediction logs
* Confidence tracking

---

### 🔁 Scalable API Design

* Stateless inference service
* Easily scalable via containers

---

## 📁 Project Structure

```bash
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   ├── api/
│   └── utils/
│
├── notebooks/
│   └── experimentation.ipynb
│
├── docker/
│   └── Dockerfile
│
├── .github/workflows/
│   └── cicd.yml
│
├── tests/
│
├── requirements.txt
├── app.py / main.py
└── README.md
```

---

## 🚀 API Endpoints

### 🔹 Health Check

```
GET /
```

---

### 🔹 Predict Flower Class

```
POST /predict
```

#### Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Response

```json
{
  "top_prediction": "setosa",
  "confidence": 0.98,
  "top_k": [
    {"class": "setosa", "probability": 0.98},
    {"class": "versicolor", "probability": 0.01}
  ]
}
```

---

## 🧪 Model Details

* Dataset: Iris Dataset
* Model: Logistic Regression / Random Forest
* Output: `predict_proba()`

### 📈 Why Probability Matters?

| Approach       | Output Type | Limitation                    |
| -------------- | ----------- | ----------------------------- |
| Traditional ML | Label       | No uncertainty awareness      |
| This System    | Probability | Enables decision intelligence |

---

## ⚔️ Challenges & Solutions

### 🔴 Challenge 1: Overconfidence in Predictions

* Models tend to output high confidence even when wrong

✅ **Solution:**

* Probability calibration
* Threshold-based decision logic

---

### 🔴 Challenge 2: Making Toy Dataset Production-Ready

✅ **Solution:**

* API abstraction
* Logging & monitoring
* Containerization

---

### 🔴 Challenge 3: Scalability

✅ **Solution:**

* Stateless API
* Docker-based deployment
* Cloud-ready architecture

---

## 📊 Observability Strategy

* **Metrics:** Request latency, prediction distribution
* **Logs:** Structured logs (JSON)
* **Monitoring Stack:**

  * Prometheus
  * Grafana
  * Loki

---

## 🔁 CI/CD Pipeline

```text
GitHub Push
    ↓
Run Tests
    ↓
Build Docker Image
    ↓
Push to AWS ECR
    ↓
Deploy to Cloud (Lambda / EC2)
```

---

## 🎯 Key Learnings

* Transitioning from **ML model → Production system**
* Importance of **uncertainty in decision-making**
* Designing **scalable ML APIs**
* Implementing **real-world MLOps practices**

---

## 🌟 Future Improvements

* Model retraining pipeline
* Drift detection
* Feature store integration
* A/B testing for models
* Human-in-the-loop feedback system

---

## 👨‍💻 Author

**Rahul Shelke**
Data Scientist | ML Engineer | MLOps Enthusiast

---

## ⭐ Final Note

> This project demonstrates that even a simple dataset like Iris can be transformed into a **production-grade intelligent system** when engineered with the right mindset.

---