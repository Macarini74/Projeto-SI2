from .extract_data import extract_data, insert_df_into_bd

def insert_data():
    return insert_df_into_bd(extract_data())
