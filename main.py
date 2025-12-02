import streamlit as st
from pages.dash_geral import render_dash
from pages.dash_pred import render_pred
from backenzo import insert_data

insert_data()

st.set_page_config(
    page_title="Agora V-AI",
    page_icon='🏥',
    layout="wide",
)


pg = st.navigation([
    st.Page(render_dash, title="Dashboard Geral", icon="📊"),
    st.Page(render_pred, title="Predições", icon="📈"),
])

pg.run()