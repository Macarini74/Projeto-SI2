from .sql import quantidade_total_vendida, media_mensal_vendas, dept_maior_venda, produto_mais_vendido, loja_mais_rentavel
from .sql import graf_evolucao, graf_top_cinco, vendas_dep, evoluc_dep
import streamlit as st
import plotly.express as px
import pandas as pd
import calendar

def big_numbers():
    with st.container():

        col1, col2= st.columns(2)

        with col1.container(border=True):
            df = quantidade_total_vendida()
            qtd_vendida = df["total"].values[0]

            st.metric(
                label="Quantidade Total Vendida",
                value=f"{qtd_vendida:,.0f}" 
            )

        with col2.container(border=True):
            df = media_mensal_vendas()
            media = df['media'].values[0]
            
            st.metric(
                label="Média Mensal de Vendas",
                value=f"{media:,.2f}" 
            )

        col3, col4, col5 = st.columns(3)

        with col3.container(border=True):
            df = dept_maior_venda()
            dpt = df['DEPARTAMENTO'].values[0]
            qtd_dpt = df['qtd'].values[0]

            display = f'{dpt} ({qtd_dpt:,.0f} uni.)'

            st.metric(
                label="Departamento de Maior Venda",
                value=display 
        )

        with col4.container(border=True):
            df = produto_mais_vendido()
            prod = df['PRODUTO'].values[0]
            qtd_prod = df['qtd_vend'].values[0]

            display = f'{prod} \n ({qtd_prod:,.0f} uni.)'

            st.metric(
                label="Produto Mais Vendido",
                value=display
            )

        with col5.container(border=True):
            df = loja_mais_rentavel()
            loja = df['LOJA'].values[0]
            qtd_loja = df['qtd_total'].values[0]

            display = f'{loja} ({qtd_loja:.0f} uni.)'

            st.metric(
                label="Loja Mais Rentável",
                value=display
            )


def graficos():

    AZUIS_VIBRANTES_CUSTOM = [
        '#00BFFF',  # Azul Ciano Vibrante
        '#1E90FF',  # Azul Médio
        '#4169E1',  # Azul Royal
        '#0000FF',  # Azul Puro
        '#0000CD',  # Azul Escuro
        '#0077B6',  # Azul Forte
        '#003566',  # Azul Marinho
        '#483D8B',  # Azul Ametista
    ]

    # --- PRIMEIRO GRÁFICO ---

    with st.container(border=True):
        df = graf_evolucao()

        anos_disponiveis = sorted(df['ANO'].unique().tolist(), reverse=True)
        opcoes_selectbox = ['Todos'] + anos_disponiveis

        ano_selecionado = st.selectbox(
            'Selecione o Ano para Análise',
            options=opcoes_selectbox,
            index=0,
            key='filtro_principal_evolucao'
        )

        if ano_selecionado == 'Todos':            
            df_sazonalidade = df.groupby('MES', as_index=False)['Vendas_Totais'].sum()
            df_sazonalidade['MES'] = df_sazonalidade['MES'].astype(str).str.zfill(2)

            fig1 = px.line(
                df_sazonalidade,
                x='MES',
                y='Vendas_Totais',
                title='Soma das Vendas Mensais em Todo o Período Histórico',
                labels={
                    'MES': 'Mês do Ano',
                    'Vendas_Totais': 'Total de Vendas Acumuladas'
                }
            )
            fig1.update_traces(line=dict(color='#0077B6', width=3))
            
        else:            
            df_analise = df[df['ANO'] == ano_selecionado].copy()
            
            if df_analise.empty:
                st.warning(f'Nenhum dado encontrado para o ano {ano_selecionado}.')
                st.stop()
                
            df_analise['Data_Mensal'] = (
                df_analise['ANO'].astype(str) + '-' + df_analise['MES'].astype(str).str.zfill(2)
            )
            
            fig1 = px.line(
                df_analise,
                x='Data_Mensal',
                y='Vendas_Totais',
                title=f'Evolução Mensal das Vendas no Ano {ano_selecionado}',
                labels={
                    'Data_Mensal': 'Mês/Ano',
                    'Vendas_Totais': 'Total de Vendas (Quantidade)'
                }
            )
            fig1.update_xaxes(tickangle=45)
            fig1.update_traces(line=dict(color='#0077B6', width=3))

        fig1.update_layout(hovermode="x unified")

        # --- SEGUNDO GRÁFICO ---
        df = graf_top_cinco()

        PRODUTO_COL = 'PRODUTO' 


        if ano_selecionado != 'Todos':
            df_filtrado = df[df['ANO'] == ano_selecionado].copy()
        else:
            df_filtrado = df.copy()

        df_top_geral = df_filtrado.groupby(PRODUTO_COL)['Total_Vendido'].sum().reset_index()

        top_10_produtos_nomes = df_top_geral.sort_values(by='Total_Vendido', ascending=False).head(10)[PRODUTO_COL].tolist()

        df_para_grafico = df_filtrado[df_filtrado[PRODUTO_COL].isin(top_10_produtos_nomes)].copy()

        df_ranking = df_top_geral[df_top_geral[PRODUTO_COL].isin(top_10_produtos_nomes)]
        ranking_map = df_ranking.set_index(PRODUTO_COL)['Total_Vendido'].to_dict()
        df_para_grafico['Ranking_Order'] = df_para_grafico[PRODUTO_COL].map(ranking_map)

        fig2 = px.bar(
            df_para_grafico.sort_values(by='Ranking_Order', ascending=True),
            x='Total_Vendido',
            y=PRODUTO_COL,
            orientation='h', 
            color='LOJA', 
            color_discrete_sequence=AZUIS_VIBRANTES_CUSTOM,
            title=f'Top 10 Produtos Mais Vendidos por Loja - {ano_selecionado}',
            labels={'Total_Vendido': 'Quantidade Vendida', PRODUTO_COL: 'Produto', 'LOJA': 'Loja'},
            height=600,
            hover_data=[PRODUTO_COL, 'LOJA', 'Total_Vendido']
        )

        fig2.update_layout(
            xaxis_title='Quantidade Vendida',
            yaxis_title='Produto',
            legend_title='Lojas',
            template='plotly_white'
        )
        fig2.update_yaxes(categoryorder='array', categoryarray=top_10_produtos_nomes[::-1])
    
        # --- TERCEIRO GRÁFICO ---

        df = vendas_dep()

        DEP_COL = 'DEPARTAMENTO' 
        VALOR_COL = 'Total_Vendido'

        if ano_selecionado != 'Todos':
            df_filtrado_dep = df[df['ANO'] == ano_selecionado].copy()
        else:
            df_filtrado_dep = df.copy()

        df_dep_participacao = df_filtrado_dep.groupby(DEP_COL)[VALOR_COL].sum().reset_index()

        fig_pizza = px.pie(
            df_dep_participacao,
            values=VALOR_COL,
            names=DEP_COL,
            title=f'Participação Percentual por Departamento - {ano_selecionado}',
            hole=.3, 
            color_discrete_sequence=AZUIS_VIBRANTES_CUSTOM, 
            hover_data=[VALOR_COL],
        )

        fig_pizza.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            sort=True 
        )

        # --- QUARTO GRÁFICO ---

        df = evoluc_dep()

        DEP_COL = 'DEPARTAMENTO' 
        MES_COL = 'MES' 
        VALOR_COL = 'Total_Vendido'
        ANO_COL = 'ANO'

        if ano_selecionado != 'Todos':
            df_filtrado_evolucao = df[df[ANO_COL].astype(str) == str(ano_selecionado)].copy()
        else:
            df_filtrado_evolucao = df.copy()
        
        if ano_selecionado == 'Todos':
            group_cols = [ANO_COL, MES_COL, DEP_COL]
            titulo_ano = "Evolução ao Longo dos Anos"
        else:
            group_cols = [MES_COL, DEP_COL]
            titulo_ano = f"Evolução em {ano_selecionado}"

        df_evolucao = df_filtrado_evolucao.groupby(group_cols)[VALOR_COL].sum().reset_index()

        if df_evolucao[MES_COL].dtype in ['int64', 'int32']:
            df_evolucao['Nome_Mes'] = df_evolucao[MES_COL].apply(lambda x: calendar.month_name[x])
            meses_ordenados = [calendar.month_name[i] for i in range(1, 13) if i in df_evolucao[MES_COL].unique()]
        else:
            df_evolucao['Nome_Mes'] = df_evolucao[MES_COL]
            meses_ordenados = df_evolucao['Nome_Mes'].unique().tolist() 

        fig_evolucao = px.line(
            df_evolucao,
            x='Nome_Mes', 
            y=VALOR_COL, 
            color=DEP_COL,
            color_discrete_sequence=AZUIS_VIBRANTES_CUSTOM,
            title=f'Evolução Mensal de Vendas por Departamento - {titulo_ano}',
            labels={
                'Nome_Mes': 'Mês', 
                VALOR_COL: 'Quantidade Vendida (Unidades)', 
                DEP_COL: 'Departamento'
            },
            line_shape='spline',
            markers=True,
            height=550
        )

        fig_evolucao.update_xaxes(
            categoryorder='array', 
            categoryarray=meses_ordenados 
        )
        fig_evolucao.update_layout(
            template='plotly_white',
            legend_title_text='Departamentos'
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1, use_container_width=True, key='chart_evolucao_vendas') 

        with col2:
            st.plotly_chart(fig_evolucao, use_container_width=True)

        col3, col4 = st.columns(2)

        st.divider()

        with col3:
            st.plotly_chart(fig_pizza, use_container_width=True)

        with col4:
            st.plotly_chart(fig2, use_container_width=True)