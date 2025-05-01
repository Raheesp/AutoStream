# 📊 AutoStream: Streamlit-Based Data Science Assistant

![Image](https://github.com/user-attachments/assets/2808e317-9f5d-4a45-80ff-43264ce442a4)

# 🚀 AutoStream: Your All-in-One Streamlit-Based Data Science Assistant

**AutoStream** is a no-code/low-code platform built with Streamlit that empowers data professionals to perform end-to-end data analysis, visualization, and AutoML — all in an intuitive web interface.

Whether you're a beginner, analyst, or data scientist, AutoStream offers powerful features to interact with your data, build ML models, and explain results with ease — without writing a single line of code.

---

## 📚 Table of Contents

1. [✨ Features](#-features)  
2. [🏗️ Project Structure](#-project-structure)  
3. [📦 Tech Stack](#-tech-stack)  
4. [📌 Use Cases](#-where-autostream-is-useful)  
5. [✅ Benefits](#-benefits-of-using-autostream)  
6. [📸 Screenshots](#-screenshots)  
7. [🔧 Installation](#-installation)  
 
---

## ✨ Features

- **📂 Data Upload & Exploration:**  
  Upload multiple CSV files and explore them interactively in the UI.

- **💬 Chat with Data (LLM-Powered):**  
  Use Mistral (via Ollama) to ask natural language questions about your dataset.

- **📊 Exploratory Data Analysis (EDA):**  
  Instantly generate stunning reports with Sweetviz and view them in-app.

- **⚙️ AutoML Workflows (Regression & Classification):**  
  Automatically train, compare, and save the best models using PyCaret.

- **🔍 SHAP Model Explainability:**  
  Visualize feature importance and model decision logic through SHAP plots.

- **🌐 Local & Secure:**  
  Runs fully offline; great for sensitive data and internal use.

---

## 🏗️ Project Structure

```
AutoStream/
├── streamlit_app.py
├── requirements.txt
├── vectorstore/
│   └── db_faiss/
├── dataset.csv
├── best_model.pkl
└── sweetviz_report.html
```

---

## 📦 Tech Stack

- **Frontend:** Streamlit  
- **LLM Interface:** PandasAI with Mistral (via Ollama)  
- **AutoML:** PyCaret  
- **EDA Reports:** Sweetviz  
- **Model Explainability:** SHAP  
- **Visualization:** Matplotlib  
- **Data Handling:** Pandas

---

## 🌟 Where AutoStream is Useful

### 1. 🔬 Rapid Prototyping for Data Scientists
- Quickly test ideas without coding.
- Compare models, check SHAP values, and iterate rapidly.

### 2. 🧮 Business Analysts & Domain Experts
- Upload data and ask questions like “What is the trend of sales over time?”
- Zero programming required to gain deep insights.

### 3. 🎓 Educational Use
- Ideal for learning and teaching machine learning concepts.
- Students can explore real datasets, visualize patterns, and train models interactively.

### 4. 🕵️ Model Transparency in Regulated Domains
- Explain your model predictions with SHAP values.
- Build trust in AI/ML decisions for healthcare, finance, etc.

### 5. ⚡ Startups, Hackathons & Internal Tools
- Lightweight yet powerful analytics dashboard.
- Use AutoStream instead of building full pipelines from scratch.

---

## ✅ Benefits of Using AutoStream

### 🔧 End-to-End Workflow in One App
From uploading datasets to generating models and explanations — everything happens in one place.

### 🧠 Human-Like Interaction with Data
Ask questions like "Which product had the highest revenue in Q1?" and get answers instantly.

### 💡 Instant Visual EDA
Get clean, informative EDA reports with one click — no need to code plots or summaries.

### 📈 SHAP Integration
Deep dive into how each feature influences model predictions — crucial for debugging and trust-building.

### 🔒 Private & Offline
LLM and AutoML components run locally — perfect for organizations with strict data governance.

### ⚙️ Easy Customization
Extend the platform with your own models, analytics logic, or additional ML libraries.

---

## 📸 Screenshots

![Image](https://github.com/user-attachments/assets/1541f6b5-e611-43f2-bf55-121dcad515a0)
![Image](https://github.com/user-attachments/assets/96d6212b-30a1-4ad3-b201-23c71cb1c262)
![Image](https://github.com/user-attachments/assets/869098f8-1832-4f1c-8660-62c4b1c187af)
![Image](https://github.com/user-attachments/assets/bfc8f27e-fe39-4955-80b4-fa782a63939e)
![Image](https://github.com/user-attachments/assets/292c2666-ea49-47e9-9a24-776c15aec6d8)
![Image](https://github.com/user-attachments/assets/214f29e9-92f4-43dd-9d25-c269cd183d22)


## 🔧 Installation

1. **Clone the Repo**

```bash
git clone https://github.com/your-username/AutoStream.git
cd AutoStream
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run streamlit_app.py
```

> 💡 Make sure you have [Ollama](https://ollama.com/) installed and running the Mistral model locally for the chat features to work.



Made with ❤️ using Streamlit, PyCaret, and LLMs.


