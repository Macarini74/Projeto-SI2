import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, mean_absolute_percentage_error

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Pasta para salvar os resultados
OUTPUT_DIR = "resultados_vendas_final"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Carrega e padroniza os dados
def carregar_dados():
    print("--- Carregando arquivos ---")
    arquivos = ['../dados/2022.csv', '../dados/2023.csv', '../dados/2024.csv']
    dfs = []
    
    for arq in arquivos:
        if os.path.exists(arq):
            df_temp = pd.read_csv(arq, sep=';', encoding='latin1', on_bad_lines='skip')
            dfs.append(df_temp)
            print(f"   -> {arq} OK")

    df = pd.concat(dfs, ignore_index=True)

    df.columns = [c.upper() for c in df.columns]
    
    # Usa SECAO como DEPARTAMENTO caso necessário
    if 'DEPARTAMENTO' not in df.columns and 'SECAO' in df.columns:
        df.rename(columns={'SECAO': 'DEPARTAMENTO'}, inplace=True)
    
    # Cria coluna de data a partir do mês
    if 'MES' in df.columns:
        df['Data'] = pd.to_datetime(df['MES'], format='%Y-%m', errors='coerce')
    else:
        raise ValueError("Coluna MES não encontrada")

    df = df.dropna(subset=['Data'])
    
    # Converte virgula para ponto
    if df['QUANTIDADE'].dtype == 'object':
        df['Qtd_Clean'] = df['QUANTIDADE'].astype(str).str.replace(',', '.').astype(float)
    else:
        df['Qtd_Clean'] = df['QUANTIDADE'].astype(float)
    
    print(f"   Total de registros carregados: {len(df)}")
    return df

# Calcula lags e médias móveis
def calcular_lags_e_medias(df_target):
    df_target = df_target.sort_values('Data')
    
    df_target['Mes'] = df_target['Data'].dt.month
    df_target['Ano'] = df_target['Data'].dt.year
    
    # Lags usados para capturar histórico
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        df_target[f'Lag_{lag}'] = df_target.groupby('PRODUTO')['Qtd_Clean'].shift(lag)
        
    # Média dos últimos 3 meses
    df_target['Media_Movel_3m'] = df_target.groupby('PRODUTO')['Qtd_Clean'].transform(
        lambda x: x.rolling(3).mean()
    )
    
    return df_target

# Etapa principal de engenharia de features
def engenharia_features(df):
    print("--- Engenharia de Features ---")
    
    # Agrupa vendas mensais por produto
    df_agrupado = df.groupby(['Data', 'PRODUTO']).agg({'Qtd_Clean': 'sum'}).reset_index()
    
    df_agrupado = calcular_lags_e_medias(df_agrupado)
    
    df_agrupado.dropna(inplace=True)
    return df_agrupado

# Modelos de regressão para prever quantidade vendida
def executar_regressao(train, test, features, target):
    print("\n--- Executando Regressão ---")
    
    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    resultados_detalhados = test[['Data', 'PRODUTO', target]].copy()
    resultados_detalhados.rename(columns={target: 'Real'}, inplace=True)
    
    df_grafico_total = test.groupby('Data')[target].sum().reset_index().rename(columns={target: 'Real'})
    
    melhor_modelo_obj = None
    menor_rmse = float('inf')
    nome_melhor_modelo = ""

    metricas_lista = []

    for nome, modelo in modelos.items():
        print(f"   Treinando {nome}...")
        modelo.fit(train[features], train[target])
        pred = modelo.predict(test[features])
        
        # Impede previsões negativas
        pred = np.maximum(pred, 0)
        
        resultados_detalhados[nome] = pred
        
        # Calcula métricas
        mse = mean_squared_error(test[target], pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test[target], pred)
        
        print(f"     -> RMSE: {rmse:.2f} | MSE: {mse:.2f} | MAPE: {mape:.2%}")
        
        metricas_lista.append({'Modelo': nome, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape})
        
        if rmse < menor_rmse:
            menor_rmse = rmse
            melhor_modelo_obj = modelo
            nome_melhor_modelo = nome

        soma_pred_mes = resultados_detalhados.groupby('Data')[nome].sum().reset_index()
        df_grafico_total = df_grafico_total.merge(soma_pred_mes, on='Data')
        
        # Exporta dados para análise de dispersão
        df_scatter = pd.DataFrame({'Real': test[target], 'Previsto': pred})
        df_scatter.to_csv(f"{OUTPUT_DIR}/dados_grafico_scatter_{nome.replace(' ', '_')}.csv", index=False)

        # Scatter plot
        plt.figure(figsize=(6, 6))
        amostra = df_scatter.sample(min(2000, len(pred)))
        sns.scatterplot(x='Real', y='Previsto', data=amostra, alpha=0.5)
        plt.plot([0, amostra['Real'].max()], [0, amostra['Real'].max()], 'r--')

        plt.title(f'{nome}\nRMSE={rmse:.1f} | MSE={mse:.1f} | MAPE={mape:.1%}')
        plt.savefig(f"{OUTPUT_DIR}/regressao_scatter_{nome.replace(' ', '_')}.png")
        plt.close()

    pd.DataFrame(metricas_lista).to_csv(f"{OUTPUT_DIR}/comparativo_metricas_modelos.csv", index=False)

    # Gera série temporal
    df_grafico_total.to_csv(f"{OUTPUT_DIR}/dados_grafico_serie_temporal_total.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df_grafico_total['Data'], df_grafico_total['Real'], label='REAL', color='black', linewidth=3, marker='o')
    cores = ['red', 'green', 'blue']
    for i, nome in enumerate(modelos.keys()):
        plt.plot(df_grafico_total['Data'], df_grafico_total[nome], label=nome, color=cores[i], linestyle='--')
        
    plt.title('Previsão de Vendas Totais (2024)')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/regressao_serie_temporal_total.png")
    plt.close()
    
    resultados_detalhados.to_csv(f"{OUTPUT_DIR}/previsoes_regressao_detalhado.csv", index=False)
    print("   -> Gráficos e métricas exportados.")
    
    return melhor_modelo_obj, nome_melhor_modelo

# Modelos de classificação para identificar meses acima/abaixo da média
def executar_classificacao(train, test, features):
    print("\n--- Executando Classificação (Acima/Abaixo da Média) ---")
    
    def get_target(df):
        return (df['Qtd_Clean'] > df['Media_Movel_3m']).astype(int)
    
    y_train = get_target(train)
    y_test = get_target(test)
    X_train = train[features]
    X_test = test[features]
    
    modelos = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    acuracias = {}
    
    for nome, modelo in modelos.items():
        print(f"   Treinando {nome}...")
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        acuracias[nome] = acc
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, pred)
        df_cm = pd.DataFrame(cm, index=['Real_0', 'Real_1'], columns=['Pred_0', 'Pred_1'])
        df_cm.to_csv(f"{OUTPUT_DIR}/dados_matriz_confusao_{nome.replace(' ', '_')}.csv")
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusão: {nome}')
        plt.savefig(f"{OUTPUT_DIR}/classificacao_confusao_{nome.replace(' ', '_')}.png")
        plt.close()
        
    df_ranking = pd.DataFrame(list(acuracias.items()), columns=['Modelo', 'Acuracia'])
    df_ranking.to_csv(f"{OUTPUT_DIR}/dados_ranking_classificacao.csv", index=False)

    # Ranking de acurácia
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Modelo', y='Acuracia', data=df_ranking, palette='viridis')
    plt.title('Ranking de Acurácia dos Modelos')
    plt.savefig(f"{OUTPUT_DIR}/classificacao_ranking_acuracia.png")
    plt.close()

# Gera previsões para os 12 meses de 2025
def gerar_previsao_2025(modelo, df_historico, features):
    print("\n--- Gerando Previsão para 2025 ---")
    
    produtos = df_historico['PRODUTO'].unique()
    
    df_total = df_historico.copy()
    df_total = df_total.sort_values('Data')
    
    ano_target = 2025
    previsoes_2025 = []
    
    for mes in range(1, 13):
        nova_data = pd.Timestamp(year=ano_target, month=mes, day=1)
        print(f"   Previsto para: {nova_data.strftime('%Y-%m')}")
        
        # Cria placeholders para o novo mês
        df_mes = pd.DataFrame({'PRODUTO': produtos})
        df_mes['Data'] = nova_data
        df_mes['Qtd_Clean'] = 0 
        
        # Adiciona ao histórico para recalcular lags
        df_temp = pd.concat([df_total, df_mes], ignore_index=True)
        df_temp = calcular_lags_e_medias(df_temp)
        
        df_to_predict = df_temp[df_temp['Data'] == nova_data].copy()
        df_to_predict = df_to_predict.fillna(0)
        
        preds = modelo.predict(df_to_predict[features])
        preds = np.maximum(preds, 0)
        
        df_to_predict['Qtd_Clean'] = preds
        
        previsoes_2025.append(df_to_predict[['Data', 'PRODUTO', 'Qtd_Clean']])
        
        # Atualiza histórico
        df_total = pd.concat([df_total, df_to_predict[['Data', 'PRODUTO', 'Qtd_Clean']]], ignore_index=True)

    df_2025_final = pd.concat(previsoes_2025, ignore_index=True)
    df_2025_final.to_csv(f"{OUTPUT_DIR}/previsao_detalhada_2025.csv", index=False)
    
    # Série temporal consolidada de 2025
    df_grafico_2025 = df_2025_final.groupby('Data')['Qtd_Clean'].sum().reset_index()
    df_grafico_2025.to_csv(f"{OUTPUT_DIR}/dados_grafico_previsao_2025_total.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_grafico_2025['Data'], df_grafico_2025['Qtd_Clean'], marker='o', color='purple')
    plt.title('Projeção de Vendas Totais - 2025')
    plt.xlabel('Mês')
    plt.ylabel('Quantidade Prevista')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/previsao_2025_temporal.png")
    plt.close()
    
# Analisa sazonalidade por departamento
def analise_por_departamento(df_raw):
    print("\n--- Sazonalidade por Departamento ---")
    
    if 'DEPARTAMENTO' not in df_raw.columns:
        print("   Coluna 'DEPARTAMENTO' não encontrada.")
        print(f"   Colunas existentes: {list(df_raw.columns)}")
        return

    df_dept = df_raw.groupby(['Data', 'DEPARTAMENTO'])['Qtd_Clean'].sum().reset_index()
    
    df_dept.to_csv(f"{OUTPUT_DIR}/dados_vendas_por_departamento_historico.csv", index=False)
    
    top_depts = df_dept.groupby('DEPARTAMENTO')['Qtd_Clean'].sum().nlargest(5).index
    print(f"   Top 5 departamentos: {list(top_depts)}")

    plt.figure(figsize=(12, 6))
    
    for dept in top_depts:
        subset = df_dept[df_dept['DEPARTAMENTO'] == dept]
        plt.plot(subset['Data'], subset['Qtd_Clean'], label=dept)
        
        nome_arq = str(dept).replace(' ', '_').replace('/', '-')
        subset.to_csv(f"{OUTPUT_DIR}/dados_dept_{nome_arq}.csv", index=False)
    
    plt.title('Sazonalidade: Top 5 Departamentos')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/sazonalidade_departamentos_top5.png")
    plt.close()
    
    print("   -> Dados por departamento exportados.")

# main
if __name__ == "__main__":
    df_raw = carregar_dados()
        
    analise_por_departamento(df_raw)

    df_proc = engenharia_features(df_raw)
        
    # Separa dados de treino e teste
    mask_train = df_proc['Data'].dt.year < 2024
    mask_test = df_proc['Data'].dt.year == 2024
        
    train = df_proc[mask_train]
    test = df_proc[mask_test]
        
    features = ['Mes', 'Ano', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_6', 'Lag_12']
    target = 'Qtd_Clean'
        
    # Executa modelos de regressão
    melhor_modelo, nome_modelo = executar_regressao(train, test, features, target)
    executar_classificacao(train, test, features)
            
    if melhor_modelo is not None:
        print(f"  Melhor modelo: {nome_modelo}")
        print("   Re-treinando para prever 2025...")
                
        melhor_modelo.fit(df_proc[features], df_proc[target])
                
        gerar_previsao_2025(melhor_modelo, df_proc, features)
            
        print(f"\nResultados salvos em '{OUTPUT_DIR}'")
