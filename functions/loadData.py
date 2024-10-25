import duckdb
import streamlit as st
import pandas as pd


def Load(file):
    if file is not None:
        file_type = file.name.split('.')[-1]

        if file_type == 'xlsx':
            st.write("Please convert to csv file for performance problem")
            return None

        elif file_type == 'csv':
            data = pd.read_csv(file)

            data.columns = data.columns.str.lower()

            if 'longitude' in data.columns and 'latitude' in data.columns:
                data = data[['longitude', 'latitude']]
            else:
                st.write("The file does not contain the columns 'longitude' and 'latitude'.")
                return None

            st.write("Download file successfully")
            choice = st.selectbox("How many ingredients do you want?", ("All", "Custom"))

            if choice == "All":
                return data

            if choice == "Custom":
                num = st.number_input("Enter the quantity of ingredients you want to get", min_value=1,
                                      max_value=len(data), step=1)
                value = st.selectbox("What style do you want to get?", ("Random", "Head", "Tail"))

                if num > 0:
                    con = duckdb.connect()
                    con.register('data', data)

                    if value == "Random":
                        random_rows = con.execute(f"SELECT * FROM data USING SAMPLE {num}").fetchdf()
                        return random_rows
                    elif value == "Head":
                        head_rows = data.head(int(num))
                        return head_rows
                    elif value == "Tail":
                        tail_rows = data.tail(int(num))
                        return tail_rows

        else:
            st.write("File format not supported.")
            return None
