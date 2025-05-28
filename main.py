import streamlit as st
import os

st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Prediction using Deep Learning and Ensembling")

st.sidebar.title("Navigation")
st.sidebar.success("Select a model from above.")

st.markdown(
    """
    This application demonstrates stock price prediction using various deep learning models (LSTM, GRU)
    and ensembling techniques (Meta-Learner).
    """
)

# You can add more content to the main page if needed.
# For example, a brief overview of the project or instructions.

# To ensure the Pages directory is recognized, you might want to explicitly
# list the pages in the sidebar if Streamlit's automatic page discovery
# isn't working as expected, though it usually does for files in 'pages/'.
