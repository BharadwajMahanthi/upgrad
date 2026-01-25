import pandas as pd
import sqlite3
import ast
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

DB_PATH = 'data/nutrigo.db'
RAW_CSV_PATH = 'data/raw-data_recipe.csv'

def get_nutrient_amount(n_dict, key):
    if not n_dict or key not in n_dict:
        return None
    data = n_dict[key]
    if isinstance(data, dict):
        return data.get('amount')
    return data

def enrich():
    if not os.path.exists(DB_PATH):
        logging.error(f"Database not found at {DB_PATH}")
        return
    if not os.path.exists(RAW_CSV_PATH):
        logging.error(f"CSV not found at {RAW_CSV_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=300000") # 5 minutes

    logging.info("Starting comprehensive enrichment via direct SQLite access...")

    # Use chunking for large file
    chunk_size = 1000
    count = 0
    
    # We include nutritions now
    cols_to_use = ['recipe_id', 'aver_rate', 'review_nums', 'reviews', 'nutritions']
    
    try:
        for chunk in pd.read_csv(RAW_CSV_PATH, chunksize=chunk_size, usecols=cols_to_use):
            for _, row in chunk.iterrows():
                recipe_id = str(row['recipe_id'])
                rating = float(row['aver_rate']) if not pd.isnull(row['aver_rate']) else 0.0
                review_nums = int(row['review_nums']) if not pd.isnull(row['review_nums']) else 0
                
                # 1. Parse Nutritions
                sodium = None
                sugars = None
                cholesterol = None
                saturated_fat = None
                
                raw_nutritions = row.get('nutritions')
                if raw_nutritions and isinstance(raw_nutritions, str) and raw_nutritions.strip():
                    try:
                        clean_n_str = raw_nutritions.replace("u'", "'")
                        n_dict = ast.literal_eval(clean_n_str)
                        if isinstance(n_dict, dict):
                            sodium = get_nutrient_amount(n_dict, 'sodium')
                            sugars = get_nutrient_amount(n_dict, 'sugars')
                            cholesterol = get_nutrient_amount(n_dict, 'cholesterol')
                            saturated_fat = get_nutrient_amount(n_dict, 'saturatedFat')
                    except:
                        pass

                # 2. Update recipe info
                cur.execute(
                    """UPDATE recipe SET 
                       rating = ?, 
                       review_nums = ?, 
                       sodium = ?, 
                       sugars = ?, 
                       cholesterol = ?, 
                       saturated_fat = ? 
                       WHERE recipe_id = ?""",
                    (rating, review_nums, sodium, sugars, cholesterol, saturated_fat, recipe_id)
                )

                # 3. Parse and Insert Reviews (Dictionary parsing)
                raw_reviews = row.get('reviews')
                if raw_reviews and isinstance(raw_reviews, str) and raw_reviews.strip() and raw_reviews != '[]':
                    try:
                        clean_reviews_str = raw_reviews.replace("u'", "'")
                        # Handle literal strings that might be inside the dict
                        reviews_data = ast.literal_eval(clean_reviews_str)
                        
                        # The data is a dict of {review_id: {text, rating, userName, ...}}
                        if isinstance(reviews_data, dict):
                            for rev_id, rev in reviews_data.items():
                                comment = rev.get('text', '')
                                if not comment: continue
                                
                                user_name = rev.get('userName', 'Anonymous')
                                rating_val = float(rev.get('rating', 0))
                                ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Insert if not already present
                                user_id = f"legacy_{user_name}"[:50]
                                comment_prefix = f"{comment[:50]}%"
                                
                                cur.execute(
                                    "SELECT 1 FROM review WHERE recipe_id = ? AND user_id = ? AND comment LIKE ? LIMIT 1",
                                    (recipe_id, user_id, comment_prefix)
                                )
                                if not cur.fetchone():
                                    cur.execute(
                                        "INSERT INTO review (recipe_id, user_id, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?)",
                                        (recipe_id, user_id, rating_val, comment, ts)
                                    )
                        elif isinstance(reviews_data, list):
                            # Just in case some rows are lists
                            for rev in reviews_data:
                                comment = rev.get('text', '')
                                if not comment: continue
                                user_name = rev.get('userName', 'Anonymous')
                                rating_val = float(rev.get('rating', 0))
                                ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                                user_id = f"legacy_{user_name}"[:50]
                                comment_prefix = f"{comment[:50]}%"
                                cur.execute(
                                    "SELECT 1 FROM review WHERE recipe_id = ? AND user_id = ? AND comment LIKE ? LIMIT 1",
                                    (recipe_id, user_id, comment_prefix)
                                )
                                if not cur.fetchone():
                                    cur.execute(
                                        "INSERT INTO review (recipe_id, user_id, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?)",
                                        (recipe_id, user_id, rating_val, comment, ts)
                                    )
                    except Exception as e:
                        # logging.warning(f"Error parsing reviews for {recipe_id}: {e}")
                        pass # Skip malformed lists

                count += 1
            
            conn.commit()
            if count % 5000 == 0:
                logging.info(f"Processed {count} entries...")
            
    except Exception as e:
        logging.error(f"Enrichment failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    enrich()
