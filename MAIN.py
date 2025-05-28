import streamlit as st

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="chart_with_upwards_trend",
)

st.title("Stock Prediction App")

PAGES = {
    "LSTM Prediction": "Pages/lstm_page.py",
    "GRU Prediction": "Pages/gru_page.py",
    "Meta-Learner Prediction": "Pages/meta_page.py",
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if selection == "LSTM Prediction":
    with open("Pages/lstm_page.py", "r") as f:
        code = f.read()
        exec(code)
elif selection == "GRU Prediction":
    with open("Pages/gru_page.py", "r") as f:
        code = f.read()
        exec(code)
elif selection == "Meta-Learner Prediction":
    with open("Pages/meta_page.py", "r") as f:
        code = f.read()
        exec(code)
