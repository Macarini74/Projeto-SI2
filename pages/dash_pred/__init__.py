import streamlit as st
from .front import scatter_grad, reg_lin, ran_for, ser_temp, matriz_knn, matriz_lg_reg, matriz_rand_for, class_rank, pred

def render_graphs():

    st.markdown("<h1 style='text-align: center;'>🔮 Predições 🔮</h1>", unsafe_allow_html=True)

    st.divider()

    with st.container(border=True):

        st.markdown("<h3 style='text-align: center;'>Algoritmos de Classificação</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            matriz_knn()

        with col2:
            matriz_lg_reg()

        with col3:
            matriz_rand_for()
        
        class_rank()

    with st.container(border=True):

        st.markdown("<h3 style='text-align: center;'>Algoritmos de Regressão</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            scatter_grad()
        
        with col2:
            reg_lin()
        
        with col3:
            ran_for()

        col4, col5 = st.columns(2)

        with col4:
            ser_temp()

        with col5:
            pred()
    
    
    