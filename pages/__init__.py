from banco.bd import get_db_connection
import pandas as pd

def execute_query(query, fetch_results=True):
    """
    Executa uma query SQL no banco de dados SQLite e retorna os resultados, se houver.
    """
    conn = None
    try:
        # 1. Abre a conexão usando a função
        conn = get_db_connection()
        
        # 2. Cria o cursor manualmente (não como gerenciador de contexto)
        cur = conn.cursor()
        
        # 3. Executa a query
        cur.execute(query)
        
        # 4. Comita (salva) se a query for de escrita (INSERT, UPDATE, DELETE)
        # O SQLite exige commit manual
        conn.commit()
        
        # 5. Retorna os dados se for uma query de leitura (SELECT)
        if fetch_results:
            # Pega todos os resultados e os nomes das colunas
            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            # Converte para DataFrame do Pandas (útil para dashboards)
            return pd.DataFrame(data, columns=columns)
        
    except Exception as e:
        # Se houver erro, garante que a transação seja revertida (rollback)
        if conn:
            conn.rollback()
        print(f"Erro ao executar query SQL: {e}")
        return pd.DataFrame() # Retorna DataFrame vazio em caso de erro
        
    finally:
        # 6. Fecha a conexão
        if conn:
            conn.close()

    return None

def get_month_name(df, coluna_mes='month'):
    """
    Converte uma coluna numérica de meses (1-12) para nomes em português.
    
    Args:
        df: DataFrame pandas.
        coluna_mes: Nome da coluna com os números dos meses.
    
    Returns:
        DataFrame com a coluna modificada e ordenada corretamente.
    """
    # Mapeamento número -> nome do mês
    meses_pt = {
        1: 'Janeiro',
        2: 'Fevereiro',
        3: 'Março',
        4: 'Abril',
        5: 'Maio',
        6: 'Junho',
        7: 'Julho',
        8: 'Agosto',
        9: 'Setembro',
        10: 'Outubro',
        11: 'Novembro',
        12: 'Dezembro'
    }
    
    # Converte os números para nomes
    df[coluna_mes] = df[coluna_mes].map(meses_pt)
    
    # Garante a ordem cronológica
    ordem_meses = list(meses_pt.values())
    df[coluna_mes] = pd.Categorical(
        df[coluna_mes],
        categories=ordem_meses,
        ordered=True
    )
    
    return df.sort_values(coluna_mes)