import pandas as pd
import ast
import os

CSV_PATH = 'data/core-data_recipe.csv'

def check_keys():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH, usecols=['nutritions'], nrows=1)
    raw_nutritions = df['nutritions'].iloc[0]
    
    # Standardize string for literal_eval
    clean_str = raw_nutritions.replace("u'", "'")
    n_dict = ast.literal_eval(clean_str)
    
    print("Nutritional keys found:", sorted(n_dict.keys()))
    
    # Check for specific interesting ones
    targets = ['sodium', 'sugars', 'cholesterol', 'saturatedFat', 'fat']
    for t in targets:
        if t in n_dict:
            print(f"Found {t}: {n_dict[t].get('amount')} {n_dict[t].get('unit')}")
        else:
            print(f"Missing {t}")

if __name__ == "__main__":
    check_keys()
