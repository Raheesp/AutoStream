import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
import shap
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
DB_FAISS_PATH = "vectorstore/db_faiss"

def streamlit_ui():
    with st.sidebar:
        st.image("shutterstock_1166533285-Converted-02.png")
        choice = option_menu('Navigation', ['Home', 'Data Analysis', 'Chat with Mistral', 'EDA', 'AutoML'], default_index=0)

    if choice == 'Home':
        st.markdown("## 👋 Welcome to **AutoStream**")
        st.image("./the stream.gif", use_column_width=True)
    
        st.markdown("""
        AutoStream empowers you to analyze data, run automated machine learning models, and explore insights effortlessly using powerful tools and LLMs (Large Language Models).
        """)
    
        st.markdown("### 🚀 Get Started in 3 Steps:")
        st.markdown("""
        1. 📁 Upload your dataset in the appropriate section (Data Analysis, EDA, or AutoML).
        2. 💡 Explore your data with visualizations or chat directly with it using Mistral LLM.
        3. 🤖 Let AutoML handle the modeling for you — classification or regression!
        """)
    
        st.markdown("### 🔍 Key Features:")
        st.markdown("""
        - 🧠 **LLM-Powered Data Chat** with Mistral
        - 📊 **EDA with Sweetviz** for visual summaries
        - ⚙️ **AutoML** using PyCaret (Classification & Regression)
        - 📈 **SHAP Explainability** for model insights
        """)
    
        st.markdown("---")
        if st.button("🚀 Jump into Data Analysis"):
            st.switch_page("Data Analysis")


    elif choice == 'Data Analysis':
        st.title("Data Analysis Dashboard")
        def chat_with_csv(df, query):
            llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral")
            pandas_ai = SmartDataframe(df, config={"llm": llm})
            return pandas_ai.chat(query)

        uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

        if uploaded_files:
            selected_file = st.selectbox("Choose a CSV", [f.name for f in uploaded_files])
            df = pd.read_csv(next(f for f in uploaded_files if f.name == selected_file))
            st.dataframe(df.head(), use_container_width=True)

            query = st.text_area("Ask about the data:")
            if st.button("Ask"):
                result = chat_with_csv(df, query)
                st.success(result)

    elif choice == "EDA":
        st.title("Exploratory Data Analysis")

        uploaded_file = st.file_uploader("Upload CSV for EDA", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)

            import sweetviz as sv
            report = sv.analyze(df)
            report.show_html("sweetviz_report.html", open_browser=False)

            # Display full-screen-like Sweetviz report
            with open("sweetviz_report.html", "r", encoding='utf-8') as f:
                html = f.read()
                st.components.v1.html(html, height=1500, scrolling=True)


    elif choice == "Chat with Mistral":
        from langchain_community.llms import Ollama 
        llm = Ollama(model="mistral")
        st.title("Chatbot")
        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate"):
            st.write_stream(llm.stream(prompt, stop=['<|eot_id|>']))

    elif choice == "AutoML":
        task = st.radio("Choose a step", ["Upload", "Modeling", "Download"])
        st.info("Automated ML: Upload your data, select a target, and build models!")

        if task == "Upload":
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)
                df.to_csv("dataset.csv", index=False)
                st.session_state["df_loaded"] = True
                st.session_state["df"] = df
                st.dataframe(df)

        if task == "Modeling":
            if "df_loaded" not in st.session_state:
                st.warning("Please upload a dataset first.")
                return

            df = st.session_state["df"]
            st.write("Your dataset:")
            st.dataframe(df.head(), use_container_width=True)
            model_type = st.selectbox("Type of Task", ["Classification", "Regression"])
            target = st.selectbox("Choose target column", df.columns)

            if st.button("Run AutoML"):
                if model_type == "Classification":
                    clf_setup(df, target=target, session_id=123)
                    st.dataframe(clf_pull(), use_container_width=True)
                    model = clf_compare()
                    clf_save(model, "best_model")
                else:
                    reg_setup(df, target=target, session_id=123)
                    st.dataframe(reg_pull(), use_container_width=True)
                    model = reg_compare()
                    reg_save(model, "best_model")

                st.success("Model training complete!")
                explainer = shap.Explainer(model.predict, df.drop(columns=[target]))
                shap_values = explainer(df.drop(columns=[target]))
                st.subheader("SHAP Summary Plot")
                fig = plt.figure()
                shap.summary_plot(shap_values, df.drop(columns=[target]), show=False)
                st.pyplot(fig)

        if task == "Download":
            try:
                with open("best_model.pkl", "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="best_model.pkl")
            except FileNotFoundError:
                st.error("No model found. Please run modeling first.")
    


if __name__ == "__main__":
    streamlit_ui()
