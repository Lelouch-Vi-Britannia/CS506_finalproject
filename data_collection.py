import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def scrape_weather_table(url, csv_filename='data/weather_data.csv'):
    # Make a request to the website
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table with class 'obs-history'
    table = soup.find('table', {'class': 'obs-history'})
    if not table:
        print("Weather observation table not found.")
        return

    # Extract the table headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]

    # Extract the table rows
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = tr.find_all('td')
        row = [cell.text.strip() for cell in cells]
        rows.append(row)

    # Ensure the number of columns in rows matches the number of headers
    min_columns = min(len(headers), len(max(rows, key=len)))
    headers = headers[:min_columns]
    rows = [row[:min_columns] for row in rows]

    # Create a DataFrame from the scraped data
    df_new = pd.DataFrame(rows, columns=headers)

    # Add current timestamp for tracking update time
    df_new['Scraped Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if the CSV file already exists
    if os.path.exists(csv_filename):
        # Load the existing data
        df_existing = pd.read_csv(csv_filename)

        # Ensure 'Date' and 'Time (edt)' columns are strings
        df_new['Date'] = df_new['Date'].astype(str)
        df_new['Time (edt)'] = df_new['Time (edt)'].astype(str)
        df_existing['Date'] = df_existing['Date'].astype(str)
        df_existing['Time (edt)'] = df_existing['Time (edt)'].astype(str)

        # Convert 'Date' and 'Time (edt)' to datetime for comparison
        df_new['Datetime'] = pd.to_datetime(df_new['Date'] + ' ' + df_new['Time (edt)'], format='%d %H:%M',
                                            errors='coerce')
        df_existing['Datetime'] = pd.to_datetime(df_existing['Date'] + ' ' + df_existing['Time (edt)'],
                                                 format='%d %H:%M', errors='coerce')

        # Filter only new rows that are not already in the existing data
        df_new_filtered = df_new[df_new['Datetime'] > df_existing['Datetime'].max()]

        # Exclude empty or all-NA entries before concatenation
        df_new_filtered = df_new_filtered.dropna(how='all')

        # Combine the existing and new filtered data
        df_combined = pd.concat([df_existing, df_new_filtered])
    else:
        # If no existing data, the new data becomes the combined dataset
        df_combined = df_new

    # Sort the data from oldest to newest
    df_combined = df_combined.sort_values(by=['Date', 'Time (edt)']).reset_index(drop=True)

    # Save the updated data to the CSV file
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df_combined.to_csv(csv_filename, index=False)
    print(f"Data successfully saved to {csv_filename}")


# URL to the National Weather Service page
url = 'https://forecast.weather.gov/data/obhistory/KBOS.html'
scrape_weather_table(url)