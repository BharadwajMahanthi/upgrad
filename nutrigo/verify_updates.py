import sqlite3
import pandas as pd

def check_schema(db_path):
    print(f"\n--- Checking Schema: {db_path} ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables = ['user', 'recipe', 'interaction']
    for table in tables:
        print(f"\nTable: {table}")
        cursor.execute(f"PRAGMA table_info({table});")
        cols = {col[1]: col[2] for col in cursor.fetchall()}
        for name, dtype in cols.items():
            print(f"  {name}: {dtype}")
            
    conn.close()

def check_data_samples(db_path):
    print(f"\n--- Checking Data Samples: {db_path} ---")
    conn = sqlite3.connect(db_path)
    
    print("\nRecipe Instruction Sample (mapping check):")
    df_recipe = pd.read_sql_query("SELECT recipe_id, recipe_name, instructions FROM recipe LIMIT 1;", conn)
    print(df_recipe)
    
    print("\nInteraction Timestamp Sample (parsing check):")
    df_inter = pd.read_sql_query("SELECT user_id, recipe_id, timestamp FROM interaction WHERE timestamp IS NOT NULL LIMIT 1;", conn)
    print(df_inter)
    
    conn.close()

if __name__ == "__main__":
    db_path = 'nutrigo.db'
    try:
        check_schema(db_path)
        check_data_samples(db_path)
    except Exception as e:
        print(f"Error: {e}")
