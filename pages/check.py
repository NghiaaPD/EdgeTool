import streamlit as st
import pandas as pd

st.set_page_config(page_title="Check", page_icon="✔️")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    uploaded_file_type = uploaded_file.name.split('.')[-1]

    if uploaded_file_type in ['csv', 'xlsx']:
        if uploaded_file_type == 'csv':
            st.write("File uploaded is currently a CSV.")
        else:
            data = pd.read_excel(uploaded_file)
            csv_uploaded_file_name = uploaded_file.name.replace('.xlsx', '.csv')

            data_cleaned = data.dropna()

            data_cleaned.to_csv(csv_uploaded_file_name, index=False)

            st.success(f"File has been saved as: {csv_uploaded_file_name}")
    else:
        st.error("Please choose CSV or XLSX file format.")
