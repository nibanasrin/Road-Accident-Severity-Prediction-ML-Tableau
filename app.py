import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Road Accident Severity Prediction System", layout="wide")

st.title("🚦 Road Accident Severity Prediction System")
st.markdown("Upload a dataset file (CSV / XLSX / XLS / JSON) to predict accident severity using the trained Machine Learning model.")

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "best_road_accident_severity_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model file not found. Please make sure 'best_road_accident_severity_model.pkl' is in the same folder as app.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload accident dataset file",
    type=["csv", "xlsx", "xls", "json"]
)

# -------------------------------
# Function to read uploaded file
# -------------------------------
def load_file(file):
    file_name = file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    elif file_name.endswith(".xls"):
        df = pd.read_excel(file, engine="xlrd")
    elif file_name.endswith(".json"):
        df = pd.read_json(file)
    else:
        return None

    return df

# -------------------------------
# Prediction process
# -------------------------------
if uploaded_file is not None:
    try:
        df_input = load_file(uploaded_file)

        if df_input is None:
            st.error("Unsupported file format.")
            st.stop()

        st.subheader("Uploaded Data Preview")
        st.dataframe(df_input.head())

        # Predict button
        if st.button("Predict Accident Severity"):
            predictions = model.predict(df_input)

            # Add prediction column
            df_result = df_input.copy()
            df_result["Predicted_Accident_Severity"] = predictions

            st.success("Prediction completed successfully!")

            st.subheader("Prediction Results Preview")
            st.dataframe(df_result.head())

            # Save predictions to CSV for download
            output_file = "predicted_accident_severity.csv"
            df_result.to_csv(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 Download Predicted File",
                    data=f,
                    file_name="predicted_accident_severity.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error while processing file: {e}")