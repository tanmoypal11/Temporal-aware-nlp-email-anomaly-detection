# 📧 Temporal-Aware NLP Email Anomaly Detection

An end-to-end NLP pipeline to detect anomalous communication patterns in corporate emails using unsupervised learning, behavioral analytics, and temporal features.

---

## 🚀 Overview

This project analyzes unstructured email data (Enron dataset) to identify unusual or potentially non-compliant communication patterns.

It combines:
- Natural Language Processing (NLP)
- Behavioral analysis
- Temporal pattern detection
- Unsupervised machine learning (Isolation Forest)

---

## 🧠 Key Features

- 🔍 Text processing using TF-IDF / BERT embeddings  
- 📊 Behavioral features (email length, recipients, keywords)  
- ⏱ Temporal features (hour, day, rolling activity)  
- ⚠️ Anomaly detection using Isolation Forest  
- 📈 Interactive Streamlit dashboard  
- 🔮 Real-time email anomaly prediction  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Sentence Transformers (BERT)  
- Streamlit  
- Matplotlib  

---

## 📂 Project Structure

```
project/
│
├── data/                      # Dataset
├── notebooks/                 # EDA & experimentation
├── models/
│   ├── isolation_forest_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│
├── streamlit_app/
│   ├── app.py                # Dashboard
│   ├── pages/               # Prediction page
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Pipeline Flow

```
Data → EDA → Cleaning → Feature Engineering → Model → Insights → Dashboard
```

---

## 📊 Dashboard Features

- 📈 Email activity vs anomaly timeline  
- ⚠️ Top risky users  
- 🔑 Keyword trends  
- 🚨 Flagged emails table  
- 🔍 Real-time anomaly prediction  

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run streamlit_app/app.py
```

---

## 📦 Model Artifacts

- isolation_forest_model.pkl → trained model  
- scaler.pkl → feature scaling  
- feature_cols.pkl → feature schema  

---

## 📌 Dataset

- Enron Email Dataset  
- Contains real-world corporate email communications  

---

## 🎯 Use Cases

- Compliance monitoring  
- Fraud detection  
- Insider threat detection  
- Risk analytics in financial institutions  

---

## 🧠 Future Improvements

- Explainable AI (“Why flagged?”)  
- User risk scoring system  
- Alert system integration  
- Advanced NLP models (LLMs)  

---

## 👤 Author

**Tanmoy Pal**  
MBA (Business Analytics) | Data & Risk Analytics Enthusiast  