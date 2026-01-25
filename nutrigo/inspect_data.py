import pandas as pd
import sqlite3
import os

def inspect_csv(file_path):
    print(f"\n--- {file_path} ---")
    df = pd.read_csv(file_path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
    print("Full Shape estimate (file size):", os.path.getsize(file_path))

def inspect_db(db_path):
    print(f"\n--- {db_path} ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
    conn.close()

if __name__ == "__main__":
    csv_files = [
        'core-data-train_rating.csv',
        'core-data_recipe.csv',
        'raw-data_interaction.csv'
    ]
    for f in csv_files:
        if os.path.exists(f):
            inspect_csv(f)
    
    if os.path.exists('nutrigo.db'):
        inspect_db('nutrigo.db')
    elif os.path.exists('instance/nutrigo.db'):
        inspect_db('instance/nutrigo.db')
