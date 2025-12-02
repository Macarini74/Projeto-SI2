from pages import execute_query
import streamlit as st

@st.cache_data
def quantidade_total_vendida():
    query = """
            SELECT SUM(quantidade) AS total
            FROM registros_vendas
            """
    
    return execute_query(query)

@st.cache_data
def media_mensal_vendas():
    query = """
            SELECT
                AVG(Vendas_Mensais) AS media
            FROM
                (
                    SELECT
                        ANO,
                        MES,
                        SUM(QUANTIDADE) AS Vendas_Mensais
                    FROM
                        registros_vendas
                    GROUP BY
                        ANO,
                        MES
                ) AS Subconsulta_Vendas_Mensais;
            """
    return execute_query(query)

@st.cache_data
def dept_maior_venda():
    query = """
        SELECT
            DEPARTAMENTO,
            SUM(QUANTIDADE) AS qtd
        FROM
            registros_vendas
        GROUP BY
            DEPARTAMENTO
        ORDER BY
            qtd DESC
        LIMIT 1;
            """
    
    return execute_query(query)

@st.cache_data
def produto_mais_vendido():
    query = """
        SELECT
            PRODUTO,
            SUM(QUANTIDADE) AS qtd_vend
        FROM
            registros_vendas
        GROUP BY
            PRODUTO
        ORDER BY
            qtd_vend DESC
        LIMIT 1;
            """

    return execute_query(query)

@st.cache_data
def loja_mais_rentavel():
    query = """
            SELECT
                LOJA,
                SUM(QUANTIDADE) AS qtd_total
            FROM
                registros_vendas
            GROUP BY
                LOJA
            ORDER BY
                qtd_total DESC
            LIMIT 1;
            """
    
    return execute_query(query)

@st.cache_data
def graf_evolucao():
    query = """
        SELECT
            ANO,
            MES,
            SUM(QUANTIDADE) AS Vendas_Totais
        FROM
            registros_vendas
        GROUP BY
            ANO,
            MES
        ORDER BY
            ANO ASC,
            MES ASC;
            """
    
    return execute_query(query)

@st.cache_data
def graf_top_cinco():
    query = """
        SELECT
            PRODUTO,
            LOJA,
            ANO,
            SUM(QUANTIDADE) AS Total_Vendido
        FROM
            registros_vendas
        GROUP BY
            PRODUTO,
            LOJA,
            ANO;
        """
    
    return execute_query(query)

@st.cache_data
def vendas_dep():
    query = """
        SELECT
            DEPARTAMENTO,
            ANO,
            SUM(QUANTIDADE) AS Total_Vendido
        FROM
            registros_vendas
        GROUP BY
            DEPARTAMENTO,
            ANO;
        """
    
    return execute_query(query)

@st.cache_data
def evoluc_dep():
    query = """
        SELECT
            ANO,
            MES,
            DEPARTAMENTO,
            SUM(QUANTIDADE) AS Total_Vendido
        FROM
            registros_vendas
        GROUP BY
            ANO,
            MES,
            DEPARTAMENTO
        ORDER BY
            ANO,
            MES;
        """
    
    return execute_query(query)