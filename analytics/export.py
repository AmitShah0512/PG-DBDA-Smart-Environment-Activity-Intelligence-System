import sqlite3
import csv
from utils.config import DB_PATH, EXPORT_CSV

def export_actions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT * FROM actions")
    rows = cur.fetchall()

    with open(EXPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "timestamp", "action", "confidence"])
        writer.writerows(rows)

    conn.close()
    print("Exported to:", EXPORT_CSV)

# Alias so realtime.py can call it
export_csv = export_actions
