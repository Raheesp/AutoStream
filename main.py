import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
import shap
import matplotlib.pyplot as plt
import sklearn
import scipy
import re
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    pull as reg_pull,
    save_model as reg_save,
    predict_model as reg_predict
)
from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare,
    pull as clf_pull,
    save_model as clf_save,
    predict_model as clf_predict
)


st.set_page_config(layout="wide")
DB_FAISS_PATH = "vectorstore/db_faiss"


def loose_column_match(columns, keyword):
    keyword = keyword.lower().replace(" ", "").replace("_", "")
    cleaned = [col.lower().replace(" ", "").replace("_", "") for col in columns]
    matches = get_close_matches(keyword, cleaned, n=1, cutoff=0.8)
    if matches:
        original_index = cleaned.index(matches[0])
        return columns[original_index]
    return None

# 🤖 Core function
def chat_with_csv(df, query):
    llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral")
    pandas_ai = SmartDataframe(df, config={
        "llm": llm,
        "whitelisted_libraries": ["pandas", "numpy", "matplotlib", "seaborn", "pandasql", "scipy"]
    })

    query_lower = query.lower()
    if any(kw in query_lower for kw in ["number of rows", "number of columns", "how many rows", "how many columns", "shape", "size of data"]):
        num_rows, num_cols = df.shape
        return {
            "type": "list",
            "value": [
                {"type": "number", "value": num_rows, "label": "Rows"},
                {"type": "number", "value": num_cols, "label": "Columns"}
            ]
        }

    # 🧮 Grouped aggregations (with support for time-based grouping)
    if " by " in query_lower and any(op in query_lower for op in ["average", "mean", "sum", "count", "max", "min", "median", "std"]):
        match = re.search(r"(average|mean|sum|count|maximum|max|minimum|min|median|std|standard deviation)\s+(\w+)\s+by\s+(\w+)", query_lower)
        
        if match:
            op, value_col, group_col = match.groups()
            
            # Match column names loosely
            value_match = loose_column_match(df.columns, value_col)
            group_match = loose_column_match(df.columns, group_col)
            
            if value_match and group_match:
                agg_func = {
                    "average": "mean", "mean": "mean", "sum": "sum", "count": "count",
                    "maximum": "max", "max": "max", "minimum": "min", "min": "min",
                    "median": "median", "std": "std", "standard deviation": "std"
                }.get(op, "mean")

                # Check if the value column is numeric before applying aggregation
                if pd.api.types.is_numeric_dtype(df[value_match]):
                    result_df = df.groupby(group_match)[value_match].agg(agg_func).reset_index()
                    summary = f"The table below shows the {agg_func} of '{value_match}' grouped by '{group_match}'."
                    return {"data": result_df, "summary": summary}
                else:
                    return f"Cannot apply '{agg_func}' to non-numeric column '{value_match}'."
            else:
                return "Couldn't match column names in grouped query."
        
        # Attempt to handle date-based grouping
        group_key = handle_date_grouping(query_lower, df, group_match, value_match)
        if group_key is None:
            return f"Column '{group_match}' could not be converted to a valid date format."

        # Apply aggregation
        agg_func = {
            "average": "mean", "mean": "mean", "sum": "sum", "count": "count",
            "maximum": "max", "max": "max", "minimum": "min", "min": "min",
            "median": "median", "std": "std", "standard deviation": "std"
        }.get(op, "mean")
        
        if not pd.api.types.is_numeric_dtype(df[value_match]):
            return f"Column '{value_match}' is not numeric and cannot be aggregated."

        result_df = df.groupby(group_key)[value_match].agg(agg_func).reset_index()
        summary = f"The table below shows the {agg_func} of '{value_match}' grouped by '{group_key}'."
        return {"data": result_df, "summary": summary}

    # 🔎 Filtered aggregations (e.g., mean of sales where quantity > 100)
    filter_match = re.search(r"(count|average|mean|sum|median|max|min|std)\s+of\s+(\w+)\s+where\s+(\w+)\s*([<>=!]+)\s*([\d.]+)", query_lower)
    if filter_match:
        op, value_col, filter_col, operator, threshold = filter_match.groups()
        value_match = loose_column_match(df.columns, value_col)
        filter_match_col = loose_column_match(df.columns, filter_col)

        if value_match and filter_match_col:
            try:
                threshold = float(threshold)
                filtered_df = df.query(f"`{filter_match_col}` {operator} {threshold}")
                if op == "count":
                    return filtered_df[value_match].count()
                elif op in ["average", "mean"]:
                    return filtered_df[value_match].mean()
                elif op == "sum":
                    return filtered_df[value_match].sum()
                elif op == "median":
                    return filtered_df[value_match].median()
                elif op == "max":
                    return filtered_df[value_match].max()
                elif op == "min":
                    return filtered_df[value_match].min()
                elif op == "std":
                    return filtered_df[value_match].std()
            except Exception as e:
                return f"Error in filtering: {e}"
        else:
            return "Couldn't match column names in filtered query."

    # 🔢 Simple column-level aggregation (e.g., mean of sales)
    if any(keyword in query_lower for keyword in ["average", "mean", "maximum", "minimum", "sum", "total", "count", "median", "std", "standard deviation"]):
        matched_column = None
        for col in df.columns:
            if col.lower().replace(" ", "").replace("_", "") in query_lower.replace(" ", ""):
                matched_column = col
                break

        if matched_column:
            if not is_numeric_dtype(df[matched_column]):
                return f"Column '{matched_column}' is not numeric and cannot be used in aggregation."

            if "average" in query_lower or "mean" in query_lower:
                return df[matched_column].mean()
            elif "maximum" in query_lower or "max" in query_lower:
                return df[matched_column].max()
            elif "minimum" in query_lower or "min" in query_lower:
                return df[matched_column].min()
            elif "sum" in query_lower or "total" in query_lower:
                return df[matched_column].sum()
            elif "count" in query_lower:
                return df[matched_column].count()
            elif "median" in query_lower:
                return df[matched_column].median()
            elif "std" in query_lower or "standard deviation" in query_lower:
                return df[matched_column].std()
            else:
                return "Sorry, I couldn't determine the correct operation."
        return "Couldn't match any column in the query. Please check column name spelling."

    # 🧠 Fallback to LLM + PandasAI
    try:
        result = pandas_ai.chat(query)
    except Exception as e:
        return f"LLM Error: {e}"

    # 📦 Return result appropriately
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], pd.DataFrame):
            st.markdown(result.get("summary", ""))
            st.dataframe(result["data"])
        elif result.get("type") == "number" and isinstance(result.get("value"), dict):
            st.metric("Rows", result["value"].get("Rows"))
            st.metric("Columns", result["value"].get("Columns"))
        elif result.get("type") == "number":
            st.metric("Result", result.get("value"))
        else:
            st.json(result) # if it's something like {'Rows': ..., 'Columns': ...}
    elif isinstance(result, (str, int, float)):
        return result
    elif isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, dict) and "data" in result:
        return result
    elif hasattr(result, "savefig"):
        return result
    elif isinstance(result, list):
        return str(result)
    else:
        return f"Unknown result format from LLM: {type(result)} - {result}"

def handle_date_grouping(query_lower, df, group_match, value_match):
    """ Helper function to extract date parts (month, year, day) if applicable """
    # Try to convert group column to datetime
    if not is_datetime64_any_dtype(df[group_match]):
        try:
            df[group_match] = pd.to_datetime(df[group_match], errors="coerce")
        except:
            return None

    # Extract date parts if applicable
    if "month" in query_lower:
        df["__month__"] = df[group_match].dt.month
        return "__month__"
    elif "year" in query_lower:
        df["__year__"] = df[group_match].dt.year
        return "__year__"
    elif "day" in query_lower:
        df["__day__"] = df[group_match].dt.day
        return "__day__"
    else:
        return group_match

def streamlit_ui():
    with st.sidebar:
        st.image("shutterstock_1166533285-Converted-02.png")
        options = ['Home', 'Data Analysis', 'Chat with Csv', 'Chatbot', 'EDA', 'AutoML']
        default_idx = options.index(st.session_state.get("option", "Home")) if "option" in st.session_state else 0
        choice = option_menu('Navigation', options, default_index=default_idx)


    if choice == 'Home':
        st.markdown("## 👋 Welcome to **AutoStream**")
        st.image("the stream.gif",use_container_width=True)
    
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
            st.session_state["option"] = "Data Analysis"

    elif choice == 'Data Analysis':
        st.title("🤖 Chat with your Data (Mistral-powered)")
        st.markdown("Ask questions or generate custom charts from your uploaded CSV file.")

        uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

        if uploaded_files:
            selected_file = st.selectbox("Choose a CSV", [f.name for f in uploaded_files])
            df = pd.read_csv(next(f for f in uploaded_files if f.name == selected_file))
            st.dataframe(df.head(), use_container_width=True)

            query = st.text_area("Ask about the data:")
            if st.button("Ask"):
                result = chat_with_csv(df, query)

                if isinstance(result, dict) and "data" in result:
                    st.write(result.get("summary", ""))
                    st.dataframe(result["data"])
                elif isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif hasattr(result, "savefig"):
                    st.pyplot(result)
                elif isinstance(result, (str, int, float)):
                    st.write(f"Result: {result}")
                elif isinstance(result, list):
                    st.write("Result:")
                    st.write(result)
                else:
                    st.info(f"Unknown response format: {type(result)}")
                    st.write(result)

    elif choice == "Chatbot":
        from langchain_community.llms import Ollama 
        llm = Ollama(model="mistral")
        st.title("Chatbot")
        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate"):
            st.write_stream(llm.stream(prompt, stop=['<|eot_id|>']))

        
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


    elif choice == "Chat with Csv":
        from langchain_community.llms import Ollama 
        llm = Ollama(model="mistral")
        st.title("🧠 Chat with Mistral LLM")

        st.sidebar.header("🔎 Data Filter Panel")
        file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.write("### Full Data Preview", df)

            # Dynamic filter system
            filter_cols = st.sidebar.multiselect("Select columns to filter:", df.columns.tolist())
            filter_conditions = {}

            for col in filter_cols:
                st.sidebar.markdown(f"**Filter: {col}**")
                if df[col].dtype == "object":
                    options = df[col].dropna().unique().tolist()
                    selected = st.sidebar.multiselect(f"Choose values for '{col}'", options, default=options)
                    filter_conditions[col] = selected
                elif pd.api.types.is_numeric_dtype(df[col]):
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    selected_range = st.sidebar.slider(f"Select range for '{col}'", min_val, max_val, (min_val, max_val))
                    filter_conditions[col] = selected_range
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    min_date, max_date = df[col].min(), df[col].max()
                    selected_range = st.sidebar.date_input(f"Select date range for '{col}'", [min_date, max_date])
                    if len(selected_range) == 2:
                        filter_conditions[col] = selected_range

            # Apply filters
            filtered_df = df.copy()
            for col, condition in filter_conditions.items():
                if isinstance(condition, list) and df[col].dtype == "object":
                    filtered_df = filtered_df[filtered_df[col].isin(condition)]
                elif isinstance(condition, tuple) and pd.api.types.is_numeric_dtype(df[col]):
                    filtered_df = filtered_df[(filtered_df[col] >= condition[0]) & (filtered_df[col] <= condition[1])]
                elif isinstance(condition, list) and pd.api.types.is_datetime64_any_dtype(df[col]):
                    filtered_df = filtered_df[(df[col] >= condition[0]) & (df[col] <= condition[1])]

            st.write("### Filtered Data", filtered_df)
            st.sidebar.success(f"{len(filtered_df)} rows after filtering.")

            prompt = st.text_area("🗣️ Ask a question about your data:")
            if st.button("Generate Explanation"):
                if not filtered_df.empty:
                    # Use filtered data in prompt (up to 10 rows for context)
                    context = filtered_df.head(10).to_markdown(index=False)
                    full_prompt = f"Here is some data:\n{context}\n\nAnswer the following question:\n{prompt}"
                    st.write("🧠 Generating response...")
                    st.write_stream(llm.stream(full_prompt, stop=["<|eot_id|>"]))
                else:
                    st.warning("⚠️ No data to analyze after filtering. Please adjust filters.")
        else:
            st.warning("📂 Please upload a CSV file to begin.")

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
                st.success("File uploaded successfully!")
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
            metric = st.selectbox("Choose metric", ["Accuracy", "AUC", "F1", "Recall", "Precision", "MAE", "MSE", "RMSE", "R2"])
            
            if st.button("Run AutoML"):
                X = df.drop(columns=[target])
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.write("Train size:", X_train.shape)
                st.write("Test size:", X_test.shape)

                if model_type == "Classification":
                    if df[target].value_counts().min() < 2:
                        st.error("❌ The least populated class in the target column has fewer than 2 samples. Please choose a different target column or clean your data.")
                        return
                    clf_setup(data=df, target=target, session_id=123)
                    st.subheader("Model Comparison Results")
                    st.dataframe(clf_pull(), use_container_width=True)

                    model = clf_compare(sort=metric)
                    clf_save(model, "best_model")

                    predictions = clf_predict(model, data=X_test)
                    st.write(predictions.columns)
                    acc = accuracy_score(y_test, predictions["prediction_label"])
                    st.metric(label=f"{metric} on Test Set", value=round(acc, 4))

                else:  # Regression
                    reg_setup(data=df, target=target, session_id=123)
                    st.subheader("Model Comparison Results")
                    st.dataframe(reg_pull(), use_container_width=True)

                    model = reg_compare(sort=metric)
                    reg_save(model, "best_model")

                    predictions = reg_predict(model, data=X_test)
                    pred_col = [col for col in predictions.columns if col not in df.columns and col != target][0]
                    y_pred = predictions[pred_col]

                    if metric == "RMSE":
                        score = mean_squared_error(y_test, y_pred, squared=False)
                    elif metric == "MAE":
                        score = mean_absolute_error(y_test, y_pred)
                    elif metric == "R2":
                        score = r2_score(y_test, y_pred)
                    else:
                        score = model.score(X_test, y_test)

                    st.metric(label=f"{metric} on Test Set", value=round(score, 4))

                st.success("Model training complete!")

                # SHAP explainability
                try:
                    explainer = shap.Explainer(model.predict, X)
                    shap_values = explainer(X)

                    st.subheader("SHAP Plot")
                    plot_type = st.selectbox("SHAP Plot Type", ["Summary", "Bar", "Beeswarm"])

                    if plot_type == "Summary":
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X, show=False)
                        st.pyplot(fig)

                    elif plot_type == "Bar":
                        fig = shap.plots.bar(shap_values, show=False)
                        st.pyplot(fig)

                    elif plot_type == "Beeswarm":
                        fig, ax = plt.subplots()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)

                except Exception as e:
                    st.warning(f"SHAP could not generate the plot for this model: {e}")
                    

        if task == "Download":
            try:
                with open("best_model.pkl", "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="best_model.pkl")
            except FileNotFoundError:
                st.error("No model found. Please run modeling first.")
    


if __name__ == "__main__":
    streamlit_ui()
