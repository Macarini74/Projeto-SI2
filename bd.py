import sqlite3

NOME_BANCO = 'vendas.db'
NOME_TABELA = 'registros_vendas'

def criar_banco_e_tabela():
    try:
        conn = sqlite3.connect(NOME_BANCO)
        cursor = conn.cursor()

        sql_create_table = f"""
        CREATE TABLE IF NOT EXISTS {NOME_TABELA} (
            LOJA TEXT,
            CODIGO TEXT,
            PRODUTO TEXT,
            DEPARTAMENTO TEXT,
            QUANTIDADE INTEGER,
            MES TEXT
        );
        """
        
        cursor.execute(sql_create_table)
        
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"❌ Erro ao criar o banco de dados/tabela: {e}")
    finally:
        # Fecha a conexão
        if conn:
            conn.close()

if __name__ == "__main__":
    criar_banco_e_tabela()