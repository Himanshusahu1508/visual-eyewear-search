import sqlite3
import pandas as pd
import os

DB_PATH = "data/eyewear.db"
METADATA_PATH = "data/dataset_v1/metadata.csv"


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("metadata.csv not found")

    df = pd.read_csv(METADATA_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0

    for _, row in df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO catalog_items (
            image_id,
            image_path,
            brand,
            shape,
            material,
            color,
            price
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["image_id"]),
            row["image_path"],
            row.get("brand"),
            row.get("shape"),
            row.get("material"),
            row.get("color"),
            int(row["price"]) if not pd.isna(row["price"]) else None
        ))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"Loaded {inserted} catalog items into SQLite")


if __name__ == "__main__":
    load_metadata()
