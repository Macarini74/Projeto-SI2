import streamlit as st
from .front import big_numbers, graficos


def render_dash():

    st.markdown(
    "<h1 style='text-align: center;'>📊 Dashboard Geral 📊</h1>", unsafe_allow_html=True)

    st.divider()
    big_numbers()

    st.divider()
    graficos()
    