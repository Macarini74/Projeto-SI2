import sqlite3
import os # Importar 'os' é boa prática, mas não estritamente necessário para esta função SQLite

NOME_BANCO = 'vendas.db'
NOME_TABELA = 'registros_vendas'

def get_db_connection():
    """
    Cria e retorna uma conexão com o banco de dados SQLite.
    (Análoga à função de conexão do PostgreSQL, mas simplificada para SQLite.)
    """
    try:
        conn = sqlite3.connect(NOME_BANCO)
        return conn
    except sqlite3.Error as e:
        print(f"❌ Erro ao conectar ao banco de dados SQLite: {e}")
        raise e

def criar_banco_e_tabela():
    """
    Garante que o banco de dados e a tabela 'registros_vendas' com o esquema atualizado existam.
    """
    conn = None 
    try:
        conn = get_db_connection() 
        cursor = conn.cursor()

        sql_create_table = f"""
        CREATE TABLE IF NOT EXISTS {NOME_TABELA} (
            LOJA TEXT,
            CODIGO TEXT,
            PRODUTO TEXT,
            DEPARTAMENTO TEXT,
            QUANTIDADE INTEGER,
            MES INTEGER,
            ANO INTEGER,
            UNIQUE (LOJA, CODIGO, MES, ANO) ON CONFLICT IGNORE
        );
        """
        
        cursor.execute(sql_create_table)
        
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"Erro ao criar o banco de dados/tabela: {e}")
    finally:
        # Fecha a conexão se ela foi estabelecida
        if conn:
            conn.close()