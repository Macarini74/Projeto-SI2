import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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


def scatter_grad():
    file_name = 'dados/predicao/dados_grafico_scatter_Gradient_Boosting.csv'

    df = pd.read_csv(file_name)

    fig = px.scatter(
        df,
        x='Real',
        y='Previsto',
        title='Real vs. Previsto (Gradient Boosting)',
        labels={
            'Real': 'Valor Real',
            'Previsto': 'Valor Previsto'
        },
        template='plotly_white',
        color_continuous_scale= AZUIS_VIBRANTES_CUSTOM
    )
    

    max_val = max(df['Real'].max(), df['Previsto'].max())
    min_val = min(df['Real'].min(), df['Previsto'].min())

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", width=2, dash="dash"),
        name='Linha de Idealidade (y=x)'
    )

    fig.update_xaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_yaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def reg_lin():
    file_name = 'dados/predicao/dados_grafico_scatter_Linear_Regression.csv'

    df = pd.read_csv(file_name)

    fig = px.scatter(
        df,
        x='Real',
        y='Previsto',
        title='Real vs. Previsto (Linear Regression)',
        labels={
            'Real': 'Valor Real',
            'Previsto': 'Valor Previsto'
        },
        template='plotly_white',
        color_continuous_scale= AZUIS_VIBRANTES_CUSTOM
    )
    

    max_val = max(df['Real'].max(), df['Previsto'].max())
    min_val = min(df['Real'].min(), df['Previsto'].min())

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", width=2, dash="dash"),
        name='Linha de Idealidade (y=x)'
    )

    fig.update_xaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_yaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def ran_for():
    file_name = 'dados/predicao/dados_grafico_scatter_Random_Forest.csv'

    df = pd.read_csv(file_name)

    fig = px.scatter(
        df,
        x='Real',
        y='Previsto',
        title='Real vs. Previsto (Random Forest)',
        labels={
            'Real': 'Valor Real',
            'Previsto': 'Valor Previsto'
        },
        template='plotly_white',
        color_continuous_scale= AZUIS_VIBRANTES_CUSTOM
    )
    

    max_val = max(df['Real'].max(), df['Previsto'].max())
    min_val = min(df['Real'].min(), df['Previsto'].min())

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", width=2, dash="dash"),
        name='Linha de Idealidade (y=x)'
    )

    fig.update_xaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_yaxes(range=[min_val * 0.9, max_val * 1.1])
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

def ser_temp():

    paleta = {
        'Real': '#00BFFF',  # Azul Ciano Vibrante
        'Linear Regression': '#1E90FF',  # Azul Médio
        'Random Forest': '#4169E1',  # Azul Royal
        'Gradient Boosting': '#0000FF',  # Azul Puro
    }

    file_name = 'dados/predicao/dados_grafico_serie_temporal_total.csv'

    df = pd.read_csv(file_name)

    id_vars = ['Data']
    value_vars = ['Real', 'Linear Regression', 'Random Forest', 'Gradient Boosting']

    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in value_vars):
        missing_cols = [col for col in value_vars if col not in df.columns]
        raise KeyError(f"Colunas ausentes para o gráfico: {missing_cols}. Verifique se o cabeçalho está exatamente como: Data,Real,Linear Regression,Random Forest,Gradient Boosting")

    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Modelo',
        value_name='Valor'
    )

    fig = px.line(
        df_long,
        x='Data',
        y='Valor',
        color='Modelo',
        title='Previsão de Vendas Totais',
        labels={
            'Data': 'Data',
            'Valor': 'Y (Valor de Vendas)',
            'Modelo': 'Modelo'
        },
        line_dash='Modelo', 
        color_discrete_map=paleta,
        line_dash_map={
            'Real': 'solid',
            'Linear Regression': 'dash',
            'Random Forest': 'dash',
            'Gradient Boosting': 'dash'
        },
        template='plotly_white',
    )

    fig.update_traces(
        line=dict(width=4),
        selector=dict(name='Real')
    )

    fig.update_traces(
        line=dict(width=2),
        selector=dict(name__in=['Linear Regression', 'Random Forest', 'Gradient Boosting'])
    )

    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=8, symbol='circle'),
        selector=dict(name='Real')
    )
    fig.update_traces(
        mode='lines',
        selector=dict(name__in=['Linear Regression', 'Random Forest', 'Gradient Boosting'])
    )

    st.plotly_chart(fig, use_container_width=True)

def matriz_knn():

    file_name = 'dados/predicao/dados_matriz_confusao_KNN.csv'

    df_cm = pd.read_csv(file_name, index_col=0)

    print("DataFrame carregado:")
    print(df_cm)

    conf_matrix_data = df_cm.values.astype(int)
    y_labels = df_cm.index.tolist() 
    x_labels = df_cm.columns.tolist() 

    fig = px.imshow(
        conf_matrix_data,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Matriz de Confusão KNN',
        labels=dict(x="Valores Previstos", y="Valores Reais", color="Contagem")
    )

    annotations_text = [
        ["Verdadeiro Negativo (TN)", "Falso Positivo (FP)"],
        ["Falso Negativo (FN)", "Verdadeiro Positivo (TP)"]
    ]

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=annotations_text[i][j],
                showarrow=False,
                font=dict(color="black", size=10),
                yref="y", xref="x", yshift=20,
            )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text='Valores Previstos')
    fig.update_yaxes(title_text='Valores Reais')

    st.plotly_chart(fig, use_container_width=True)


def matriz_lg_reg():
    file_name = 'dados/predicao/dados_matriz_confusao_Logistic_Regression.csv'

    df_cm = pd.read_csv(file_name, index_col=0)

    print("DataFrame carregado:")
    print(df_cm)

    conf_matrix_data = df_cm.values.astype(int)
    y_labels = df_cm.index.tolist() 
    x_labels = df_cm.columns.tolist() 

    fig = px.imshow(
        conf_matrix_data,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Matriz de Confusão Logistic Regression',
        labels=dict(x="Valores Previstos", y="Valores Reais", color="Contagem")
    )

    annotations_text = [
        ["Verdadeiro Negativo (TN)", "Falso Positivo (FP)"],
        ["Falso Negativo (FN)", "Verdadeiro Positivo (TP)"]
    ]

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=annotations_text[i][j],
                showarrow=False,
                font=dict(color="black", size=10),
                yref="y", xref="x", yshift=20,
            )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text='Valores Previstos')
    fig.update_yaxes(title_text='Valores Reais')

    st.plotly_chart(fig, use_container_width=True)

def matriz_rand_for():
    file_name = 'dados/predicao/dados_matriz_confusao_Random_Forest.csv'

    df_cm = pd.read_csv(file_name, index_col=0)

    print("DataFrame carregado:")
    print(df_cm)

    conf_matrix_data = df_cm.values.astype(int)
    y_labels = df_cm.index.tolist() 
    x_labels = df_cm.columns.tolist() 

    fig = px.imshow(
        conf_matrix_data,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Matriz de Confusão Random Forest',
        labels=dict(x="Valores Previstos", y="Valores Reais", color="Contagem")
    )

    annotations_text = [
        ["Verdadeiro Negativo (TN)", "Falso Positivo (FP)"],
        ["Falso Negativo (FN)", "Verdadeiro Positivo (TP)"]
    ]

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=annotations_text[i][j],
                showarrow=False,
                font=dict(color="black", size=10),
                yref="y", xref="x", yshift=20,
            )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text='Valores Previstos')
    fig.update_yaxes(title_text='Valores Reais')

    st.plotly_chart(fig, use_container_width=True)

def class_rank():
    file_name = 'dados/predicao/dados_ranking_classificacao.csv'

    df_acuracia = pd.read_csv(file_name)

    fig = px.bar(
        df_acuracia,
        x='Modelo',
        y='Acuracia',
        title='Acurácia dos Modelos de Classificação',
        labels={
            'Acuracia': 'Acurácia (%)',
            'Modelo': 'Modelo de Classificação'
        },
        color='Modelo', 
        text='Acuracia', 
        template='plotly_white',
        color_discrete_sequence=AZUIS_VIBRANTES_CUSTOM
    )

    fig.update_traces(
        texttemplate='%{y:.2%}', 
        textposition='outside'    
    )

    max_acuracia = df_acuracia['Acuracia'].max()
    fig.update_yaxes(
        range=[0, max_acuracia * 1.1],
        tickformat=".1%" 
    )

    st.plotly_chart(fig, use_container_width=True)

def pred():
    file_predicao = 'dados/predicao/dados_grafico_previsao_2025_total.csv'
    file_vendas = 'dados/2023.csv'

    df_predicao = pd.read_csv(file_predicao)

    df_predicao = df_predicao.rename(columns={'Qtd_Clean': 'Quantidade'})
    df_predicao['Data'] = pd.to_datetime(df_predicao['Data'], errors='coerce') # 'coerce' lida com datas inválidas
    df_predicao = df_predicao.dropna(subset=['Data']) # Remove linhas com datas inválidas
    df_predicao['Mês_Label'] = df_predicao['Data'].dt.strftime('%m/%y')
    df_predicao['Mês_Order'] = df_predicao['Data'].dt.month
    df_predicao_final = df_predicao.sort_values(by='Mês_Order').reset_index(drop=True)



    df_vendas = pd.read_csv(file_vendas, sep=';', encoding='latin-1')

    df_vendas['QUANTIDADE'] = df_vendas['QUANTIDADE'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_vendas['QUANTIDADE'] = pd.to_numeric(df_vendas['QUANTIDADE'], errors='coerce') 
    df_vendas = df_vendas.dropna(subset=['QUANTIDADE']) 

    df_vendas['MES'] = pd.to_datetime(df_vendas['MES'], errors='coerce') 
    df_vendas = df_vendas.dropna(subset=['MES']) 

    df_2023 = df_vendas.groupby('MES')['QUANTIDADE'].sum().reset_index()
    df_2023 = df_2023.rename(columns={'QUANTIDADE': 'Quantidade', 'MES': 'Data'})

    df_2023['Mês_Label'] = df_2023['Data'].dt.strftime('%m/%y')
    df_2023['Mês_Order'] = df_2023['Data'].dt.month
    df_2023_final = df_2023.sort_values(by='Mês_Order').reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_predicao_final['Mês_Order'],
        y=df_predicao_final['Quantidade'],
        mode='lines+markers',
        name='Predição 2025',
        line=dict(color='#1E90FF'),
        customdata=df_predicao_final['Mês_Label'],
        hovertemplate="Mês: %{customdata}<br>Quantidade: %{y:,.0f}<br>Série: %{fullData.name}<extra></extra>"
    ))

    if not df_2023_final.empty:
        fig.add_trace(go.Scatter(
            x=df_2023_final['Mês_Order'],
            y=df_2023_final['Quantidade'],
            mode='lines+markers',
            name='Real 2023',
            line=dict(color='#483D8B'),
            customdata=df_2023_final['Mês_Label'],
            hovertemplate="Mês: %{customdata}<br>Quantidade: %{y:,.0f}<br>Série: %{fullData.name}<extra></extra>"
        ))

    month_labels = df_predicao_final['Mês_Label'].str.split('/', expand=True)[0].unique().tolist()

    fig.update_layout(
        title='Comparação Mês a Mês: Predição 2025 vs Real 2023',
        xaxis=dict(
            title='Mês',
            tickvals=list(range(1, max(df_predicao_final['Mês_Order'].max(), df_2023_final['Mês_Order'].max() if not df_2023_final.empty else 12) + 1)),
            ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][:len(month_labels)],

        ),
        yaxis_title='Quantidade',
        template='plotly_white',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)