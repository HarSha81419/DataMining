import os
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Major Indian cities with coordinates
CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
}

# NASA POWER API base URL
NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M"
    "&community=RE"
    "&format=JSON"
)

def fetch_nasa_data(city, lat, lon, start_date, end_date):
    """Fetch daily NASA POWER data for given city and coordinates."""
    url = f"{NASA_URL}&latitude={lat}&longitude={lon}&start={start_date}&end={end_date}"
    try:
        print(f"📡 Fetching NASA data for {city} ({lat}, {lon}) from {start_date} to {end_date}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            print(f"⚠️ No data for {city}")
            return pd.DataFrame()

        df = pd.DataFrame(data["properties"]["parameter"])
        df = df.rename_axis("DATE").reset_index()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["CITY"] = city
        return df

    except Exception as e:
        print(f"❌ Failed for {city}: {e}")
        return pd.DataFrame()

def collect_all():
    """Fetch NASA data for all cities and save combined CSV."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=60)

    all_dfs = []
    for city, (lat, lon) in CITIES.items():
        df_city = fetch_nasa_data(city, lat, lon, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
        if not df_city.empty:
            all_dfs.append(df_city)

    if not all_dfs:
        print("⚠️ No city data fetched successfully.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(DATA_DIR, "combined_data.csv")
    combined.to_csv(output_path, index=False)
    print(f"✅ Saved combined NASA data for {len(all_dfs)} cities → {output_path}")
    return combined

if __name__ == "__main__":
    collect_all()
