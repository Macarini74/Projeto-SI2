import pandas as pd
import os
import sqlite3
from bd import NOME_BANCO, NOME_TABELA, criar_banco_e_tabela 

def extract_data():
    lista_arquivos = ['./Data/2022.csv', './Data/2023.csv', './Data/2024.csv']

    dataframes = []

    for nome_do_arquivo in lista_arquivos:
        df_temp = pd.read_csv(nome_do_arquivo, encoding='latin-1', sep=';')
        
        dataframes.append(df_temp)

    if dataframes:
        df_final = pd.concat(dataframes, ignore_index=True)
        
        return df_final
    else:
        print("\nNão foi possível ler nenhum arquivo. O DataFrame final não foi criado.")

def insert_df_into_bd(df_para_inserir):
    criar_banco_e_tabela() 
    
    try:

        conn = sqlite3.connect(NOME_BANCO)
        
        df_para_inserir.to_sql(
            name=NOME_TABELA, 
            con=conn, 
            if_exists='append', 
            index=False 
        )
        
        print(f"\nDados inseridos com sucesso na tabela '{NOME_TABELA}'.")
        
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {NOME_TABELA}")
        total_registros = cursor.fetchone()[0]
        print(f"Total de registros na tabela após a inserção: {total_registros}")
        
    except sqlite3.Error as e:
        print(f"Erro ao inserir dados no banco: {e}")
    finally:
        if conn:
            conn.close()

# Exemplo de como você usaria isso no seu código principal:
if __name__ == "__main__":
    
    df = extract_data()

    insert_df_into_bd(df)