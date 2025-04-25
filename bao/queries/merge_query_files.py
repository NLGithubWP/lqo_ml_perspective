import os
import csv


def merge_sql_queries(input_folder):
    rows = []

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".sql"):
            query_id = os.path.splitext(filename)[0]
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip().replace('\n', ' ')
                rows.append((query_id, content))

    # Write to output file
    for ele in rows:
        print(ele)



# === Example usage ===
# Set your folder path and output file name here
input_folder = "./join-order-benchmark"
merge_sql_queries(input_folder)
