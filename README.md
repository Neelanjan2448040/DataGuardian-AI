# 🛡️ DataGuardian AI

**DataGuardian AI** is a deployed AI tool that audits dataset quality using statistical analysis and explains risks using Generative AI before any machine learning model is trained.

It helps students, analysts, and practitioners **avoid common but critical data mistakes** such as missing values, multicollinearity, outliers, and data leakage.

---

## 🚀 Why DataGuardian AI?

In real-world machine learning projects, **poor data quality silently breaks models**.  
Most errors happen *before* training even begins.

**DataGuardian AI acts as a gatekeeper**, ensuring your dataset is **model-ready**.

---

## ✨ Key Features

- 📊 **Data Quality Audit**
  - Missing value analysis
  - Constant & identifier column detection
  - High correlation (multicollinearity) detection
  - Outlier detection (IQR-based)

- 🤖 **Generative AI Explanations**
  - Explains *why* each issue is harmful
  - Suggests best practices (without auto-modifying data)

- 💬 **Chat with Your Dataset**
  - Ask questions like:
    - *“Which columns should I drop?”*
    - *“Is this dataset safe for regression?”*
    - *“Why is multicollinearity a problem here?”*

- 🌐 **Deployed Streamlit App**
  - Live, accessible, and demo-ready

---

## 🧠 AI Design Philosophy (Important)

This project follows **correct AI engineering principles**:

| Component | Responsibility |
|--------|----------------|
| Pandas / NumPy | Deterministic statistical analysis |
| Generative AI (Groq LLM) | Explanation, reasoning, interaction |
| Streamlit | User interface |

❌ The LLM does **not** compute statistics  
✅ The LLM **explains and reasons about results**

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Pandas & NumPy**
- **Groq LLM (LLaMA 3.1)**
- **python-dotenv**

---

## 📂 Project Structure

```text
DataGuardian-AI/
│
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
