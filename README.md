# 📊 AutoStream: Streamlit-Based Data Science Assistant

**AutoStream** is updated version of DataQuest is an all-in-one interactive data science assistant built using Streamlit. It provides an intuitive interface for:

- 📂 Uploading and exploring datasets  
- 💬 Chatting with your data using an LLM (Mistral via Ollama)  
- 📊 Generating automated EDA reports using Sweetviz  
- 🤖 Running AutoML workflows (classification & regression) with PyCaret  
- 🔍 Explaining models using SHAP visualizations  

This project is perfect for data analysts and ML enthusiasts who want a no-code/low-code interface for common data science tasks.

---

## 🚀 Features

- **Data Analysis**  
  Upload CSV files, explore them in a friendly UI, and ask questions powered by an LLM.

- **Exploratory Data Analysis (EDA)**  
  Visualize dataset structure and insights with embedded Sweetviz reports.

- **AutoML**  
  Automatically build and compare models using PyCaret, with easy export of the best model.

- **Chat with Mistral (Ollama)**  
  Ask natural language queries about your dataset or general questions with a local Mistral model.

---

## 🧱 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sweetviz_report.html    # Auto-generated report (created at runtime)
├── best_model.pkl          # Output: Trained PyCaret model
└── vectorstore/
    └── db_faiss            # (Optional) Local vector database for advanced features
```

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dataquest.git
cd dataquest
```

### 2. Set Up a Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 🧠 LLM Setup (Ollama + Mistral)

To enable chatbot and CSV Q&A features using the **Mistral model**:

1. **Install Ollama**  
   👉 https://ollama.com

2. **Pull the Mistral model**

```bash
ollama pull mistral
```

3. **Run Ollama Server**

```bash
ollama serve
```

Then start the Streamlit app, and the chatbot features will be available.

---

## 📦 Dependencies

Some core packages used:

- `streamlit`
- `pandas`
- `streamlit-option-menu`
- `pandasai`
- `pycaret`
- `sweetviz`
- `matplotlib`
- `shap`
- `langchain-community`
- External: `Ollama` for running the Mistral model locally

All dependencies are listed in `requirements.txt`.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io)
- [PyCaret](https://pycaret.org)
- [Sweetviz](https://github.com/fbdesignpro/sweetviz)
- [PandasAI](https://github.com/gventuri/pandas-ai)
- [Ollama](https://ollama.com)
- [LangChain](https://www.langchain.com)

---

## 👨‍💻 Author

**Your Name**  
GitHub: [@Raheesp](https://github.com/Raheesp)
