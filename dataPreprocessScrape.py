import pandas as pd

# Define the CSV file path
file_path = 'data/weather_data.csv'

# Define the known year and month
YEAR = 2024
MONTH = 11  # November

# Read the CSV file
try:
    df = pd.read_csv(file_path)
    print("CSV file read successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Check if required columns exist
required_columns = ['Date', 'Time (est)', 'Temperature (ºF)', 'Relative Humidity', 'Dwpt']
df.rename(columns={'Temperature (ºF)':'Temperature'})
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    exit(1)

# Extract required columns
df_extracted = df[required_columns].copy()
print("Required columns extracted.")

# Print first few rows to check the data format
print("Sample of first few rows:")
print(df_extracted.head())

# Ensure 'Date' and 'Time (est)' columns are strings
df_extracted['Date'] = df_extracted['Date'].astype(str)
df_extracted['Time (est)'] = df_extracted['Time (est)'].astype(str)

# Print data types for confirmation
print("\nData types:")
print(df_extracted.dtypes)

# Handle possible time format issues (such as removing extra spaces)
df_extracted['Time (est)'] = df_extracted['Time (est)'].str.strip()

# Process the 'Date' column, assuming it only contains the day number
# Combine the day number with the known year and month to form a full date
# Format: 'YYYY-MM-DD'
def construct_full_date(day_str, year, month):
    try:
        day = int(day_str)
        # Check the validity of the day number
        if not 1 <= day <= 31:
            raise ValueError(f"Invalid day number: {day}")
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError as ve:
        print(f"Date construction error: {ve}")
        return None

df_extracted['Full_Date'] = df_extracted['Date'].apply(lambda x: construct_full_date(x, YEAR, MONTH))

# Check if there are any dates that could not be constructed
if df_extracted['Full_Date'].isnull().any():
    print("There are records for which a full date could not be constructed. Please check the 'Date' column data.")
    print(df_extracted[df_extracted['Full_Date'].isnull()])
    exit(1)

# Combine 'Full_Date' and 'Time (est)' columns into 'Datetime'
try:
    df_extracted['Datetime'] = pd.to_datetime(df_extracted['Full_Date'] + ' ' + df_extracted['Time (est)'],
                                             format='%Y-%m-%d %H:%M',
                                             errors='raise')
    print("Date and time combined and converted successfully.")
except ValueError as ve:
    print(f"Date-time conversion error: {ve}")
    print("Please check if the formats of 'Full_Date' and 'Time (est)' are correct.")
    exit(1)
except Exception as e:
    print(f"An unknown error occurred during date-time conversion: {e}")
    exit(1)

df_extracted['Month'] = 11
# Sort by 'Datetime'
df_sorted = df_extracted.sort_values('Datetime')
print("Data sorting completed.")

# Drop unnecessary columns
df_sorted = df_sorted.drop(columns=['Full_Date', 'Datetime'])

# Save the sorted data to a new CSV file
output_file = 'data/sorted_extracted_weather_data.csv'
try:
    df_sorted.to_csv(output_file, index=False)
    print(f"Extracted and sorted data has been saved to {output_file}")
except Exception as e:
    print(f"Error saving CSV file: {e}")
    exit(1)
