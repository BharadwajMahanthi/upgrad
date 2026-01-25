import sqlite3
import os

DB_PATH = 'data/nutrigo.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    columns = [
        ('sodium', 'FLOAT'),
        ('sugars', 'FLOAT'),
        ('cholesterol', 'FLOAT'),
        ('saturated_fat', 'FLOAT')
    ]
    
    for col_name, col_type in columns:
        try:
            print(f"Adding column {col_name}...")
            cur.execute(f"ALTER TABLE recipe ADD COLUMN {col_name} {col_type}")
            print(f"Successfully added {col_name}.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding {col_name}: {e}")
    
    conn.commit()
    conn.close()
    print("Migration finished.")

if __name__ == "__main__":
    migrate()
