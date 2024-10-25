import pandas as pd
import streamlit as st


def Load(file):
    if file is not None:
        file_type = file.name.split('.')[-1]
        if file_type in ['csv', 'xlsx']:
            if file_type == 'xlsx':
                data = pd.read_excel(file)
                csv_file_name = file.name.replace('.xlsx', '.csv')
                data.to_csv(csv_file_name, index=False)
            elif file_type == 'csv':
                data = pd.read_csv(file)

            columns_to_keep = ['longitude', 'latitude']
            data = data[columns_to_keep]

            st.write("Tải file thành công")

            choice = st.selectbox("How many ingredients do you want?", ("All", "Custom"))
            if choice == "Custom":
                num = st.number_input("Enter the quantity of ingredients you want to get", min_value=1,
                                      max_value=len(data), step=1)
                value = st.selectbox("What style do you want to get?", ("Random", "Head", "Tail"))

                if num > 0:
                    if value == "Random":
                        random_rows = data.sample(n=int(num))
                        st.write(random_rows)
                    elif value == "Head":
                        head_rows = data.head(int(num))
                        st.write(head_rows)
                    elif value == "Tail":
                        tail_rows = data.tail(int(num))
                        st.write(tail_rows)
            else:
                st.write(data)
        else:
            st.write("File format not supported.")