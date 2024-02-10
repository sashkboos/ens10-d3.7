import pandas as pd
import sys

def process_csv(file_path):
    # Load the CSV file
    print(f"Loading file: {file_path}")
    data = pd.read_csv(file_path)

    # Iterate through each column and calculate averages, min, and max
    for column in data.columns:
        try:
            # Calculate statistics only for numeric data
            avg = data[column].mean()
            min_val = data[column].min()
            max_val = data[column].max()
            print(f"Column: {column}\n Average: {avg}\n Min: {min_val}\n Max: {max_val}\n")
        except TypeError:
            print(f"Column: {column} is non-numeric and will be skipped.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    process_csv(file_path)
