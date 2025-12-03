import pandas as pd
import os
import sqlite3
from .bd import NOME_BANCO, NOME_TABELA, criar_banco_e_tabela 

def extract_data():
    lista_arquivos = ['./dados/2022.csv', './dados/2023.csv', './dados/2024.csv']
    dataframes = []

    for nome_do_arquivo in lista_arquivos:

        df_temp = pd.read_csv(nome_do_arquivo, encoding='latin-1', sep=';')
        dataframes.append(df_temp)

    if dataframes:
        df_final = pd.concat(dataframes, ignore_index=True)
        return df_final
    else:
        print("\nNão foi possível ler nenhum arquivo. O DataFrame final não foi criado.")
        return None


def transform_data(df):
    if df is None:
        return None
    
    print("\n--- Iniciando Transformação de Dados ---")
    
    df.drop_duplicates(inplace=True)
    print(f"✅ Duplicatas internas do lote removidas. Novo tamanho: {len(df)}")
    
    df['MES_TEMP'] = pd.to_datetime(df['MES'], format='%Y-%m', errors='coerce')
    df['ANO'] = df['MES_TEMP'].dt.year.astype('Int64')
    df['MES'] = df['MES_TEMP'].dt.month.astype('Int64')
    df.drop(columns=['MES_TEMP'], inplace=True, errors='ignore')
    
    return df


def insert_df_into_bd(df_para_inserir):
    df_transformado = transform_data(df_para_inserir)
    
    if df_transformado is None or df_transformado.empty:
        print("❌ Inserção cancelada: DataFrame vazio ou transformação falhou.")
        return
        
    criar_banco_e_tabela() 
    
    conn = None
    try:
        conn = sqlite3.connect(NOME_BANCO)
        cursor = conn.cursor()
        
        NOME_TABELA_TEMP = 'temp_insert_batch'
        

        df_transformado.to_sql(
            name=NOME_TABELA_TEMP, 
            con=conn, 
            if_exists='replace', 
            index=False 
        )
        print(f"\nDados inseridos na tabela temporária '{NOME_TABELA_TEMP}'.")
        
        colunas = ", ".join(df_transformado.columns)
        
        sql_insert_or_ignore = f"""
        INSERT OR IGNORE INTO {NOME_TABELA} ({colunas})
        SELECT {colunas} FROM {NOME_TABELA_TEMP};
        """
        
        cursor.execute(sql_insert_or_ignore)
        conn.commit()
        print(f"Dados únicos movidos para a tabela principal '{NOME_TABELA}'.")

        cursor.execute(f"DROP TABLE IF EXISTS {NOME_TABELA_TEMP}")
        
        cursor.execute(f"SELECT COUNT(*) FROM {NOME_TABELA}")
        total_registros = cursor.fetchone()[0]
        print(f"Total de registros na tabela após a inserção: {total_registros}")
        
    except sqlite3.Error as e:
        print(f"Erro ao inserir dados no banco: {e}")
    finally:
        if conn:
            conn.close()