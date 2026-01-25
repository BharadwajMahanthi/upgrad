import pandas as pd
import ast
import os

CSV_PATH = 'data/raw-data_recipe.csv'

def check_reviews():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Read a few rows that are likely to have reviews
    df = pd.read_csv(CSV_PATH, usecols=['recipe_id', 'reviews'], nrows=10)
    
    for idx, row in df.iterrows():
        raw_reviews = row['reviews']
        print(f"\n--- Recipe ID: {row['recipe_id']} ---")
        if not raw_reviews or raw_reviews == '[]' or pd.isnull(raw_reviews):
            print("No reviews.")
            continue
            
        try:
            # Standardize string for literal_eval
            clean_str = raw_reviews.replace("u'", "'")
            r_data = ast.literal_eval(clean_str)
            
            print(f"Type: {type(r_data)}")
            if isinstance(r_data, dict):
                print(f"Keys (first 2): {list(r_data.keys())[:2]}")
                first_key = list(r_data.keys())[0]
                first_val = r_data[first_key]
                print(f"Sample review text: {first_val.get('text', 'N/A')[:50]}...")
            elif isinstance(r_data, list):
                print(f"Length: {len(r_data)}")
                print(f"Sample review text: {r_data[0].get('text', 'N/A')[:50]}...")
        except Exception as e:
            print(f"Error parsing: {e}")

if __name__ == "__main__":
    check_reviews()
