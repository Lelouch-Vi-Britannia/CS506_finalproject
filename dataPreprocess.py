import pandas as pd
import numpy as np
import os
import chardet
from sklearn.preprocessing import StandardScaler


def detect_file_encoding(file_path, num_bytes=100000):
    """
    Detect the file encoding.

    Parameters:
    - file_path: file path
    - num_bytes: number of bytes to detect (default 100000)

    Returns:
    - encoding name (str)
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"Detected encoding: {encoding} (Confidence: {confidence * 100:.2f}%)")
    return encoding


def inspect_error_position(file_path, position):
    """
    Read the file in binary, locate and display the byte at a specific position.

    Parameters:
    - file_path: file path
    - position: byte position (int)
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(position)
            byte = f.read(1)
            print(f"Byte at position {position}: {byte} (Hex: {byte.hex()})")
    except Exception as e:
        print(f"Error reading file: {e}")


def process_new_weather_data(input_file, output_file):
    """
    Process the weather data and convert the raw data into a format suitable for machine learning models.

    Parameters:
    - input_file: path to the input CSV file.
    - output_file: path to the output processed CSV file.
    """
    # Detect file encoding
    try:
        detected_encoding = detect_file_encoding(input_file)
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        detected_encoding = 'utf-8'  # Default to UTF-8

    # Read CSV file
    try:
        df = pd.read_csv(input_file, encoding=detected_encoding)
        print("CSV file read successfully.")
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}")
        print("Attempting to read file using 'latin1' encoding.")
        try:
            df = pd.read_csv(input_file, encoding='latin1')
            print("CSV file read successfully using 'latin1' encoding.")
        except Exception as e2:
            print(f"Error reading CSV file again: {e2}")
            return
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if required columns exist
    required_columns = ['DATE', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlySeaLevelPressure']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return

    # Extract required columns
    df_extracted = df[required_columns].copy()
    print("Required columns extracted.")

    # Drop rows where 'HourlyDryBulbTemperature' is empty
    initial_row_count = df_extracted.shape[0]
    df_extracted = df_extracted.dropna(subset=['HourlyDryBulbTemperature'])
    final_row_count = df_extracted.shape[0]
    print(f"After dropping empty rows, {final_row_count} rows remain (removed {initial_row_count - final_row_count} rows).")

    # Print first few rows to check data format
    print("Sample of first few rows:")
    print(df_extracted.head())

    # Ensure 'DATE' column is a string
    df_extracted['DATE'] = df_extracted['DATE'].astype(str)

    # Process 'DATE' column, extract date, time, and month
    try:
        # Convert 'DATE' column to datetime
        df_extracted['Datetime'] = pd.to_datetime(df_extracted['DATE'], format='%Y-%m-%dT%H:%M:%S', errors='raise')
        print("Datetime conversion successful.")
    except ValueError as ve:
        print(f"Datetime conversion error: {ve}")
        print("Please check if the 'DATE' column format is 'YYYY-MM-DDTHH:MM:SS'.")
        # Optional: locate error position
        error_position = 6558  # Obtained from the error message
        inspect_error_position(input_file, error_position)
        return
    except Exception as e:
        print(f"Unknown error occurred during datetime conversion: {e}")
        return

    # Extract Date, Time (est), Month
    df_extracted['Date'] = df_extracted['Datetime'].dt.day
    df_extracted['Time (est)'] = df_extracted['Datetime'].dt.strftime('%H:%M')
    df_extracted['Month'] = df_extracted['Datetime'].dt.month
    # df_extracted['DayOfWeek'] = df_extracted['Datetime'].dt.dayofweek  # Monday=0, Sunday=6

    # Print new columns to confirm
    print("Extracted Date, Time (est), Month columns sample:")
    print(df_extracted[['Date', 'Time (est)', 'Month']].head())

    # Rename columns
    df_extracted = df_extracted.rename(columns={
        'HourlyDryBulbTemperature': 'Temperature',
        'HourlyRelativeHumidity': 'Relative Humidity',
        'HourlySeaLevelPressure': 'Dwpt'
    })
    print("Columns renamed.")
    print("Current column names:", df_extracted.columns.tolist())

    # Filter rows where 'Time (est)' ends with ':54'
    df_filtered = df_extracted[df_extracted['Time (est)'].str.endswith(':54')]
    print(f"{df_filtered.shape[0]} rows remain after filtering (keeping only data where Time (est) ends with ':54').")

    # Select and reorder columns
    output_columns = ['Date', 'Time (est)', 'Month', 'Temperature', 'Relative Humidity', 'Dwpt']
    df_output = df_filtered[output_columns].copy()

    # Add 'Datetime' column for sorting
    df_output['Datetime'] = df_extracted.loc[df_filtered.index, 'Datetime']

    # Sort by 'Datetime'
    df_sorted = df_output.sort_values('Datetime')
    print("Data sorting completed.")

    # Drop 'Datetime' column if not needed
    df_sorted = df_sorted.drop(columns=['Datetime'])

    # Save sorted data to new CSV file
    try:
        df_sorted.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


input_csv = 'data/five_year.csv'
output_csv = 'data/2.csv'
process_new_weather_data(input_csv, output_csv)
